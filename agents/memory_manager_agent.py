"""
Memory Manager Agent — I/O standardisation, parsing, and audit logging.

Responsibilities:
    1. **Input standardisation** — every prompt / payload sent to a feature
       agent passes through ``standardize_input`` first, which guarantees a
       JSON-serialisable dict with a fixed key order.
    2. **Output standardisation** — every raw LLM response is parsed by
       ``standardize_output`` into the canonical two-part envelope:

           {
               "part1_payload":   { "actionable": {...},
                                    "source_records": [...] },
               "part2_reasoning": "..."
           }

       If parsing fails the manager returns a best-effort envelope and
       records a ``parse_error`` so the UI can surface it.
    3. **Persistent logging** — every input/output and the final state are
       written to ``app_memory/<patient_id>/session_<ts>.jsonl`` (event
       stream) and ``…/session_<ts>.json`` (consolidated summary).

The MemoryManager is intentionally framework-agnostic: it knows nothing
about LangGraph and never touches the LLM directly. Each agent calls it
explicitly so the wrapping stays transparent and easy to unit-test.
"""

from __future__ import annotations

import json
import os
import re
import time
from copy import deepcopy
from typing import Any


# ── Helpers ──────────────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S | re.I)
_PART1_RE = re.compile(
    r"(?:^|\n)\s*#{0,4}\s*Part\s*1.*?\n(?P<body>.*?)"
    r"(?=\n\s*#{0,4}\s*Part\s*2|$)",
    re.S | re.I,
)
_PART2_RE = re.compile(
    r"(?:^|\n)\s*#{0,4}\s*Part\s*2.*?\n(?P<body>.*)$",
    re.S | re.I,
)


def _safe_json_default(obj: Any) -> str:
    try:
        return str(obj)
    except Exception:
        return "<unserialisable>"


def _to_jsonable(value: Any) -> Any:
    """Recursively coerce *value* into something json.dumps can handle."""
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# ── Memory Manager ───────────────────────────────────────────────────────────

class MemoryManager:
    """Per-pipeline-run audit + I/O standardiser."""

    EMPTY_PART1: dict = {"actionable": {}, "source_records": []}

    def __init__(
        self,
        patient_id: str | int,
        base_dir: str = "app_memory",
        session_id: str | None = None,
    ):
        self.patient_id = str(patient_id)
        self.session_id = session_id or time.strftime("%Y%m%d_%H%M%S")
        self.directory = os.path.join(base_dir, self.patient_id)
        os.makedirs(self.directory, exist_ok=True)
        self.log_path = os.path.join(
            self.directory, f"session_{self.session_id}.jsonl"
        )
        self.summary_path = os.path.join(
            self.directory, f"session_{self.session_id}.json"
        )
        self._summary: dict = {
            "patient_id": self.patient_id,
            "session_id": self.session_id,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "agent_outputs": {},
            "events": [],
        }

    # ── public API ─────────────────────────────────────────────────────────

    def session_metadata(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "session_id": self.session_id,
            "directory": self.directory,
            "log_path": self.log_path,
            "summary_path": self.summary_path,
        }

    def standardize_input(self, agent_name: str, payload: dict) -> dict:
        """Guarantee a JSON-serialisable dict before it reaches the agent."""
        clean = _to_jsonable(payload)
        ordered = {
            "agent": agent_name,
            "patient_id": self.patient_id,
            "received_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "payload": clean,
        }
        self._log("input", agent_name, ordered)
        return ordered

    def standardize_output(self, agent_name: str, raw_text: str) -> dict:
        """Parse a raw LLM response into the canonical two-part envelope."""
        envelope = self._parse_two_part(raw_text)
        envelope["agent_name"] = agent_name
        self._log("output", agent_name, envelope)
        self._summary["agent_outputs"][agent_name] = {
            "part1_payload": envelope.get("part1_payload", self.EMPTY_PART1),
            "part2_reasoning": envelope.get("part2_reasoning", ""),
            "raw": raw_text,
            "parse_error": envelope.get("parse_error"),
        }
        self._dump_summary()
        return envelope

    def record_skipped(self, agent_name: str, reason: str) -> dict:
        envelope = {
            "agent_name": agent_name,
            "skipped": True,
            "part1_payload": {
                "actionable": {"status": "skipped", "reason": reason},
                "source_records": [],
            },
            "part2_reasoning": (
                f"This agent was not activated by the Orchestrator. "
                f"Reason: {reason}"
            ),
        }
        self._log("skipped", agent_name, envelope)
        self._summary["agent_outputs"][agent_name] = envelope
        self._dump_summary()
        return envelope

    def record_event(self, agent_name: str, content: Any) -> None:
        self._log("event", agent_name, _to_jsonable(content))

    def record_agent_outcome(
        self, agent_name: str, outcome: dict, raw: str | None = None,
    ) -> None:
        """Persist a non-two-part agent outcome (e.g. orchestrator decision,
        diagnoses verdict) so the UI can find it under ``agent_outputs``."""
        clean = _to_jsonable(outcome)
        record = {"outcome": clean}
        if raw is not None:
            record["raw"] = raw
        self._summary["agent_outputs"][agent_name] = record
        self._log("outcome", agent_name, record)
        self._dump_summary()

    def finalise(self, final_state: dict | None = None) -> str:
        """Persist a trimmed snapshot of the final state and return the path."""
        if final_state is not None:
            trimmed: dict = {}
            for k, v in final_state.items():
                if k.endswith("_raw") or k == "visits_data":
                    continue
                trimmed[k] = _to_jsonable(v)
            self._summary["final_state"] = trimmed
        self._summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._dump_summary()
        return self.summary_path

    # ── parsing ────────────────────────────────────────────────────────────

    def _parse_two_part(self, raw_text: str) -> dict:
        """Best-effort extraction of part1_payload / part2_reasoning."""
        text = raw_text or ""

        # 1) Try strict JSON first (raw or fenced).
        for candidate in self._json_candidates(text):
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            part1 = obj.get("part1_payload") or obj.get("part1") or {}
            part2 = obj.get("part2_reasoning") or obj.get("part2") or ""
            if part1 or part2:
                return {
                    "part1_payload": self._coerce_part1(part1),
                    "part2_reasoning": str(part2),
                }

        # 2) Fall back to ``## Part 1`` / ``## Part 2`` markdown headings.
        m1 = _PART1_RE.search(text)
        m2 = _PART2_RE.search(text)
        if m1 or m2:
            part1_body = m1.group("body").strip() if m1 else ""
            part2_body = m2.group("body").strip() if m2 else ""
            part1_obj: dict
            try:
                part1_obj = json.loads(self._strip_fences(part1_body))
                if not isinstance(part1_obj, dict):
                    raise ValueError
            except Exception:
                part1_obj = {
                    "actionable": {"raw": part1_body},
                    "source_records": [],
                }
            return {
                "part1_payload": self._coerce_part1(part1_obj),
                "part2_reasoning": part2_body,
            }

        # 3) Total parse failure — wrap the raw text so nothing is lost.
        return {
            "part1_payload": {
                "actionable": {"raw_response": text[:2000]},
                "source_records": [],
            },
            "part2_reasoning": text,
            "parse_error": "Could not extract two-part schema; using raw text.",
        }

    @staticmethod
    def _json_candidates(text: str) -> list[str]:
        candidates: list[str] = []
        for fence in _FENCE_RE.findall(text):
            candidates.append(fence.strip())
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            candidates.append(stripped)
        return candidates

    @staticmethod
    def _strip_fences(text: str) -> str:
        m = _FENCE_RE.search(text)
        return m.group(1).strip() if m else text.strip()

    @classmethod
    def _coerce_part1(cls, part1: Any) -> dict:
        if not isinstance(part1, dict):
            return {"actionable": {"raw": part1}, "source_records": []}
        out = deepcopy(part1)
        out.setdefault("actionable", {})
        out.setdefault("source_records", [])
        if not isinstance(out["actionable"], (dict, list)):
            out["actionable"] = {"value": out["actionable"]}
        if not isinstance(out["source_records"], list):
            out["source_records"] = [str(out["source_records"])]
        return out

    # ── persistence ────────────────────────────────────────────────────────

    def _log(self, kind: str, agent_name: str, content: Any) -> None:
        entry = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "kind": kind,
            "agent": agent_name,
            "content": content,
        }
        self._summary["events"].append(entry)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=_safe_json_default) + "\n")
        except OSError:
            pass

    def _dump_summary(self) -> None:
        try:
            with open(self.summary_path, "w", encoding="utf-8") as f:
                json.dump(
                    self._summary, f, indent=2, default=_safe_json_default,
                )
        except OSError:
            pass


# ── Module-level helpers (used by tests + agents that don't hold a manager) ──

def empty_output(agent_name: str, reason: str = "Not activated") -> dict:
    return {
        "agent_name": agent_name,
        "skipped": True,
        "part1_payload": {
            "actionable": {"status": "skipped", "reason": reason},
            "source_records": [],
        },
        "part2_reasoning": "",
    }
