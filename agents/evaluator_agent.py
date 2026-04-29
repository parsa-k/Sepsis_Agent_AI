"""
Evaluator Agent — final quality gate for the pipeline.

Inputs
──────
* the user's verbatim intent,
* a digest of the raw clinical data per visit,
* the Orchestrator's final decision (plan, instructions, rationale),
* every agent's Part-1 payload (history + features),
* the Diagnoses Agent's full output (summary, score, verdict, plans).

Job
───
Decide whether the system actually executed the user's task with
acceptable quality and surface a single bottom-line verdict that the UI
can render as a *green* or *red* flag, plus a structured per-agent
performance report.

Output schema (JSON)
────────────────────
```json
{
  "flag":              "green" | "red" | "yellow",
  "task_executed":     <bool>,
  "confidence":        <0..100>,
  "overall_summary":   "<2-3 sentences — what worked, what didn't>",
  "agent_reports": {
    "orchestrator": { "verdict": "ok|warn|fail", "notes": "..." },
    "history":      { ... },
    "vitals":       { ... },
    "lab":          { ... },
    "microbiology": { ... },
    "pharmacy":     { ... },
    "diagnoses":    { ... }
  },
  "missing_data":      ["..."],
  "improvement_recommendations": "Markdown bullet list"
}
```

A deterministic post-processor (`_normalise_evaluation`) guarantees the
fields always exist and that `flag ∈ {green, yellow, red}` even when the
LLM hallucinates a different label.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import SepsisState


SYSTEM_PROMPT = """You are the **Evaluator Agent** — the final quality
gate of the Sepsis Diagnostic system. You DO NOT diagnose the patient
yourself. You judge whether the multi-agent pipeline actually executed
the user's task with sufficient evidence and coherence.

### Inputs you receive
* The user's verbatim intent.
* A digest of the raw data per visit (truncated for context).
* The Orchestrator's final plan + per-agent instructions.
* Every agent's Part-1 payload (history + features).
* The Diagnoses Agent's final output (summary, score, verdict, plans).

### How to judge
* **Green flag** — the task was clearly executed: agents produced
  evidence-backed payloads, the Diagnoses Agent's verdict is consistent
  with the agents' findings, and there are no blocking data gaps.
* **Yellow flag** — the task was partially executed: the verdict stands
  but with significant uncertainty (sparse data, inconsistent agent
  outputs, or unmet SEP-1 timing facts). Note exactly what is shaky.
* **Red flag** — the task could NOT be executed: critical data was
  missing, the Diagnoses Agent's verdict is unsupported by Part-1
  evidence, agents disagreed in important ways, or the user asked for
  something the available data cannot answer.

### Per-agent rubric
For each agent (orchestrator, history, vitals, lab, microbiology,
pharmacy, diagnoses) emit:
* `verdict`: "ok", "warn", or "fail".
* `notes`: 1-2 sentences explaining the verdict, citing concrete
  Part-1 / decision content.

### Output contract — return a SINGLE valid JSON object, nothing else
```json
{
  "flag":              "green" | "yellow" | "red",
  "task_executed":     <bool>,
  "confidence":        <0..100>,
  "overall_summary":   "<2-3 sentences — what worked, what didn't>",
  "agent_reports": {
    "orchestrator": {"verdict": "ok|warn|fail", "notes": "..."},
    "history":      {"verdict": "ok|warn|fail", "notes": "..."},
    "vitals":       {"verdict": "ok|warn|fail", "notes": "..."},
    "lab":          {"verdict": "ok|warn|fail", "notes": "..."},
    "microbiology": {"verdict": "ok|warn|fail", "notes": "..."},
    "pharmacy":     {"verdict": "ok|warn|fail", "notes": "..."},
    "diagnoses":    {"verdict": "ok|warn|fail", "notes": "..."}
  },
  "missing_data": ["..."],
  "improvement_recommendations": "Markdown bullet list"
}
```

For agents that were **skipped** by the Orchestrator (e.g. history on
single-visit runs, or agents that had no source data), emit
``{"verdict": "ok", "notes": "Skipped — not applicable to this run."}``.
Do NOT mark a skipped agent as "fail" simply for being absent. Output
NOTHING outside the JSON object.
"""


_AGENT_KEYS = (
    "orchestrator", "history", "vitals", "lab",
    "microbiology", "pharmacy", "diagnoses",
)
_VALID_FLAGS = {"green", "yellow", "red"}
_VALID_VERDICTS = {"ok", "warn", "fail"}


# ── LangGraph node entry-point ──────────────────────────────────────────────

def run_evaluator_agent(
    state: SepsisState,
    llm,
    memory_manager=None,
    system_prompt: str | None = None,
) -> dict:
    """Compose context, ask the LLM to judge, persist the verdict."""
    prompt = system_prompt or SYSTEM_PROMPT

    user_intent = (state.get("user_intent") or "").strip()
    decision = state.get("orchestrator_decision") or {}
    diag = state.get("diagnoses_output") or {}

    part1_per_agent = _gather_part1_payloads(state)
    raw_data_digest = _summarise_visits_data(state)

    human_content = _render_evaluator_prompt(
        subject_id=state.get("subject_id"),
        selected=state.get("selected_hadm_ids") or [],
        user_intent=user_intent,
        decision=decision,
        part1_per_agent=part1_per_agent,
        diag=diag,
        raw_data_digest=raw_data_digest,
    )

    if memory_manager is not None:
        memory_manager.standardize_input(
            "evaluator",
            {"system_prompt_excerpt": prompt[:400], "human": human_content[:8000]},
        )

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=human_content),
    ])
    raw = response.content

    parsed = _parse_evaluation(raw)
    output = _normalise_evaluation(parsed, decision=decision, raw=raw)

    if memory_manager is not None:
        memory_manager.record_agent_outcome("evaluator", output, raw=raw)

    trace_entry = {
        "agent": "evaluator",
        "kind": "verdict",
        "content": {
            "flag": output["flag"],
            "task_executed": output["task_executed"],
            "confidence": output["confidence"],
            "overall_summary": output["overall_summary"],
        },
    }
    return {
        "evaluator_output": output,
        "agent_trace": (state.get("agent_trace") or []) + [trace_entry],
    }


# ── Context builders ────────────────────────────────────────────────────────

def _gather_part1_payloads(state: SepsisState) -> dict:
    """Return ``{agent_name: {part1, skipped}}`` for every known agent."""
    out: dict = {}
    for name, key in (
        ("history",      "history_output"),
        ("vitals",       "vitals_output"),
        ("lab",          "lab_output"),
        ("microbiology", "microbiology_output"),
        ("pharmacy",     "pharmacy_output"),
    ):
        env = state.get(key) or {}
        out[name] = {
            "skipped": bool(env.get("skipped", False)),
            "part1_payload": env.get("part1_payload") or {},
        }
    return out


def _summarise_visits_data(state: SepsisState, max_chars: int = 1200) -> str:
    """Compact, LLM-friendly digest of the raw data the Data Loader fetched."""
    visits_data = state.get("visits_data") or {}
    if not visits_data:
        return "_No raw data digest available._"

    lines: list[str] = []
    for hadm, per in visits_data.items():
        present_keys: list[str] = []
        for k, v in (per or {}).items():
            if k.endswith("_raw") and v and v != "No data available.":
                present_keys.append(k.replace("_raw", ""))
        lines.append(
            f"- Visit {hadm}: domains with data = "
            f"{', '.join(present_keys) if present_keys else '<none>'}"
        )
    digest = "\n".join(lines)
    return digest[:max_chars]


def _render_evaluator_prompt(
    *,
    subject_id,
    selected: list,
    user_intent: str,
    decision: dict,
    part1_per_agent: dict,
    diag: dict,
    raw_data_digest: str,
) -> str:
    decision_dump = json.dumps(
        {
            "role":               decision.get("role"),
            "active_agents":      decision.get("active_agents"),
            "agent_instructions": decision.get("agent_instructions"),
            "history_first":      decision.get("history_first"),
            "rationale":          decision.get("rationale"),
        },
        indent=2,
        default=str,
    )

    part1_blocks: list[str] = []
    for name in _AGENT_KEYS:
        if name in ("orchestrator", "diagnoses"):
            continue
        info = part1_per_agent.get(name) or {}
        if info.get("skipped"):
            part1_blocks.append(f"#### {name}\n_Skipped — not applicable._")
        else:
            payload = info.get("part1_payload") or {}
            part1_blocks.append(
                f"#### {name}\n```json\n"
                f"{json.dumps(payload, indent=2, default=str)[:1500]}\n```"
            )

    diag_view = {
        "summary":         diag.get("summary"),
        "patient_score":   diag.get("patient_score"),
        "final_diagnosis": diag.get("final_diagnosis"),
        "sepsis3_met":     diag.get("sepsis3_met"),
        "sep1_compliant":  diag.get("sep1_compliant"),
        "details":         (diag.get("details") or "")[:1200],
        "next_steps":      (diag.get("next_steps") or "")[:600],
        "short_term_treatment": (diag.get("short_term_treatment") or "")[:600],
        "mid_term_plan":   (diag.get("mid_term_plan") or "")[:600],
    }

    return f"""## Evaluation request

**Subject:** {subject_id}
**Selected visits ({len(selected)}):** {', '.join(str(h) for h in selected)}

### User intent (verbatim)
{user_intent or '_(none)_'}

### Raw data availability digest
{raw_data_digest}

### Orchestrator final decision
```json
{decision_dump}
```

### Per-agent Part-1 payloads
{chr(10).join(part1_blocks)}

### Diagnoses Agent output
```json
{json.dumps(diag_view, indent=2, default=str)}
```

Produce the JSON evaluation now. Be honest — green only when the
pipeline truly fulfilled the user's task with evidence."""


# ── Parsing & normalisation ─────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S | re.I)
_BLOCK_RE = re.compile(r"\{.*\}", re.S)


def _parse_evaluation(raw: str) -> dict:
    if not raw:
        return {}
    fenced = _FENCE_RE.findall(raw)
    candidates = [c.strip() for c in fenced]
    block = _BLOCK_RE.search(raw)
    if block:
        candidates.append(block.group(0))
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return {}


def _normalise_evaluation(parsed: dict, *, decision: dict, raw: str) -> dict:
    parsed = dict(parsed) if isinstance(parsed, dict) else {}

    flag = str(parsed.get("flag", "")).strip().lower()
    if flag not in _VALID_FLAGS:
        flag = "yellow"

    task_executed = parsed.get("task_executed")
    if task_executed is None:
        task_executed = flag in ("green", "yellow")
    task_executed = bool(task_executed)

    try:
        confidence = int(parsed.get("confidence"))
    except (TypeError, ValueError):
        confidence = {"green": 85, "yellow": 55, "red": 25}[flag]
    confidence = max(0, min(100, confidence))

    overall_summary = (
        str(parsed.get("overall_summary", "")).strip()
        or _default_summary(flag)
    )

    reports = parsed.get("agent_reports") or {}
    if not isinstance(reports, dict):
        reports = {}

    active = set(decision.get("active_agents") or [])
    cleaned: dict = {}
    for name in _AGENT_KEYS:
        entry = reports.get(name) if isinstance(reports.get(name), dict) else {}
        verdict = str(entry.get("verdict", "")).strip().lower()
        if verdict not in _VALID_VERDICTS:
            verdict = "ok" if (name not in active and name != "diagnoses") else "warn"
        notes = str(entry.get("notes", "")).strip()
        if not notes:
            if name == "history" and "history" not in active:
                notes = "Skipped — single-visit run, History Agent not applicable."
            else:
                notes = "(no notes provided)"
        cleaned[name] = {"verdict": verdict, "notes": notes}

    missing = parsed.get("missing_data") or []
    if not isinstance(missing, list):
        missing = [str(missing)]
    missing = [str(m) for m in missing if str(m).strip()]

    recommendations = (
        str(parsed.get("improvement_recommendations", "")).strip()
        or "_No improvement recommendations recorded._"
    )

    return {
        "flag":               flag,
        "task_executed":      task_executed,
        "confidence":         confidence,
        "overall_summary":    overall_summary,
        "agent_reports":      cleaned,
        "missing_data":       missing,
        "improvement_recommendations": recommendations,
        "raw_response":       raw,
    }


def _default_summary(flag: str) -> str:
    return {
        "green":  "Pipeline executed the task; agent outputs are consistent with the verdict.",
        "yellow": "Pipeline executed the task with notable uncertainty or gaps.",
        "red":    "Pipeline could not reliably execute the task — see missing data and per-agent notes.",
    }[flag]
