"""
Orchestrator Agent — the dynamic planner of the Sepsis Diagnostic system.

Unlike the previous static pipeline, this orchestrator reads the user's
free-text instruction together with the per-visit data availability flags
and decides:

    * the **role of the run** (one-line mission statement),
    * which feature agents to **activate**
      (subset of vitals / lab / microbiology / pharmacy),
    * whether the **History Agent must run first** to extract a baseline
      (mandatory when the user selected more than one visit, otherwise
      forbidden — single-visit runs MUST skip the History Agent),
    * a short **per-agent instruction** that is injected into each
      activated feature agent's prompt.

The decision is returned as a JSON object on ``state['orchestrator_decision']``.
A deterministic fallback parser ensures the pipeline still runs even if
the LLM omits a field.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import SepsisState


SYSTEM_PROMPT = """You are the **Orchestrator** of a multi-agent Sepsis
Diagnostic system. You do NOT analyse clinical data yourself — you only
plan WHO should act and WHAT they should do.

### Sub-agents you can activate
| Agent          | Domain                                                |
|----------------|-------------------------------------------------------|
| vitals         | HR, BP/MAP, Temp, RR, SpO2, GCS                       |
| lab            | WBC, Lactate, Creatinine, Bilirubin, Platelets, Gas   |
| microbiology   | Cultures, organisms, sensitivities, infection source  |
| pharmacy       | Antibiotics, vasopressors, IV fluids, urine output    |
| history        | Prior admissions / chronic conditions / baselines     |

### Hard rules
1. If the user selected **exactly one** visit, the **history** agent MUST
   NOT appear in active_agents and ``history_first`` MUST be ``false``.
2. If the user selected **more than one** visit, the **history** agent
   MUST be activated FIRST (``history_first: true``) so its baseline can
   be propagated downstream.
3. Only activate agents whose data is actually present (the user-supplied
   availability flags tell you what is loaded).
4. Tailor each ``agent_instructions[<name>]`` to the user's intent — be
   concrete (e.g. "focus on lactate trend and SEP-1 3-hour bundle").

### Output contract — return a SINGLE valid JSON object, nothing else
```json
{
  "role": "<one-line mission>",
  "multi_visit": <bool>,
  "history_first": <bool>,
  "active_agents": ["vitals", "lab", "microbiology", "pharmacy"],
  "agent_instructions": {
    "vitals":       "...",
    "lab":          "...",
    "microbiology": "...",
    "pharmacy":     "...",
    "history":      "..."   // only when multi_visit
  },
  "rationale": "<2-4 sentences explaining why you picked this plan>"
}
```
"""


_FEATURE_AGENTS = ("vitals", "lab", "microbiology", "pharmacy")


# ── Public node entry-points ────────────────────────────────────────────────

def run_orchestrator(
    state: SepsisState,
    llm,
    memory_manager=None,
    system_prompt: str | None = None,
) -> dict:
    """Single-call planner. Produces ``state['orchestrator_decision']``."""
    prompt = system_prompt or SYSTEM_PROMPT

    selected = list(state.get("selected_hadm_ids") or [])
    if not selected and "hadm_id" in state:  # backwards-compat
        selected = [state["hadm_id"]]
    multi_visit = len(selected) > 1
    user_intent = (state.get("user_intent") or "").strip() or _DEFAULT_INTENT

    flags = state.get("available_data_flags", {})
    aggregated_flags = _aggregate_flags(flags, selected)

    human_content = _render_planning_prompt(
        subject_id=state.get("subject_id"),
        selected=selected,
        multi_visit=multi_visit,
        user_intent=user_intent,
        aggregated_flags=aggregated_flags,
        per_visit_flags=flags,
    )

    if memory_manager is not None:
        memory_manager.standardize_input(
            "orchestrator",
            {"system_prompt_excerpt": prompt[:400], "human": human_content},
        )

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=human_content),
    ])
    raw = response.content

    decision = _parse_decision(raw)
    decision = _enforce_rules(
        decision,
        multi_visit=multi_visit,
        aggregated_flags=aggregated_flags,
        user_intent=user_intent,
        raw=raw,
    )

    if memory_manager is not None:
        memory_manager.record_agent_outcome(
            "orchestrator", decision, raw=raw,
        )

    trace_entry = {
        "agent": "Orchestrator",
        "kind": "decision",
        "content": decision,
    }
    existing = state.get("agent_trace", []) or []

    return {
        "orchestrator_decision": decision,
        "agent_trace": existing + [trace_entry],
    }


# ── Routing helpers ─────────────────────────────────────────────────────────

def needs_history_first(state: SepsisState) -> str:
    """Conditional-edge function used by the LangGraph builder."""
    decision = state.get("orchestrator_decision") or {}
    if decision.get("history_first"):
        return "history"
    return "vitals"


def propagate_history_baseline(state: SepsisState) -> dict:
    """Tiny pass-through node that copies History Part-1 into ``history_baseline``.

    Runs immediately after the History Agent so downstream feature agents
    see a stable shape regardless of the History Agent's output schema.
    """
    history_out = state.get("history_output") or {}
    baseline = history_out.get("part1_payload") or {}
    trace_entry = {
        "agent": "Orchestrator",
        "kind": "propagate_baseline",
        "content": {"baseline_keys": list(baseline.get("actionable", {}).keys())},
    }
    existing = state.get("agent_trace", []) or []
    return {
        "history_baseline": baseline,
        "agent_trace": existing + [trace_entry],
    }


# ── Internal ────────────────────────────────────────────────────────────────

_DEFAULT_INTENT = (
    "Review the selected clinical records and evaluate for Sepsis-3 "
    "and SEP-1 compliance. Highlight any critical anomalies."
)


def _aggregate_flags(per_visit: dict, selected: list) -> dict:
    keys = ("vitals", "labs", "microbiology", "pharmacy", "icu", "diagnoses")
    out = {k: False for k in keys}
    for hadm in selected:
        v = per_visit.get(hadm, {}) or per_visit.get(str(hadm), {}) or {}
        for k in keys:
            if v.get(k):
                out[k] = True
    return out


def _render_planning_prompt(
    *,
    subject_id,
    selected: list,
    multi_visit: bool,
    user_intent: str,
    aggregated_flags: dict,
    per_visit_flags: dict,
) -> str:
    flags_lines = "\n".join(
        f"- {k}: {'PRESENT' if v else 'ABSENT'}"
        for k, v in aggregated_flags.items()
    )
    per_visit_lines = []
    for hadm in selected:
        v = per_visit_flags.get(hadm, {}) or per_visit_flags.get(str(hadm), {}) or {}
        active = ", ".join(k for k, val in v.items() if val) or "<none>"
        per_visit_lines.append(f"- Visit {hadm}: {active}")

    return f"""## Run plan request

**Subject:** {subject_id}
**Selected visits ({len(selected)}):** {', '.join(str(h) for h in selected)}
**Multi-visit?** {multi_visit}

### User intent (verbatim)
{user_intent}

### Aggregated data availability (across selected visits)
{flags_lines}

### Per-visit availability
{chr(10).join(per_visit_lines) if per_visit_lines else '- (none)'}

Produce the JSON plan now. Remember the hard rules — single visit ⇒ no
history agent; multiple visits ⇒ history first."""


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.S)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S | re.I)


def _parse_decision(raw: str) -> dict:
    if not raw:
        return {}
    fenced = _FENCE_RE.findall(raw)
    candidates = [c.strip() for c in fenced]
    block = _JSON_BLOCK_RE.search(raw)
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


def _enforce_rules(
    decision: dict,
    *,
    multi_visit: bool,
    aggregated_flags: dict,
    user_intent: str,
    raw: str,
) -> dict:
    decision = dict(decision) if isinstance(decision, dict) else {}

    decision.setdefault("role", _default_role(user_intent))
    decision["multi_visit"] = multi_visit
    decision["user_intent"] = user_intent

    active = decision.get("active_agents") or []
    if not isinstance(active, list):
        active = []
    active = [str(a).lower().strip() for a in active]

    if multi_visit:
        decision["history_first"] = True
        if "history" not in active:
            active = ["history"] + active
    else:
        decision["history_first"] = False
        active = [a for a in active if a != "history"]

    flag_to_agent = {
        "vitals": "vitals",
        "labs": "lab",
        "microbiology": "microbiology",
        "pharmacy": "pharmacy",
    }
    for flag, agent in flag_to_agent.items():
        if aggregated_flags.get(flag) and agent not in active:
            active.append(agent)

    valid = {"history", "vitals", "lab", "microbiology", "pharmacy"}
    seen: set = set()
    cleaned: list = []
    for a in active:
        if a in valid and a not in seen:
            cleaned.append(a)
            seen.add(a)
    decision["active_agents"] = cleaned

    instr = decision.get("agent_instructions") or {}
    if not isinstance(instr, dict):
        instr = {}
    for agent in cleaned:
        instr.setdefault(
            agent,
            f"Focus on data relevant to: {user_intent}",
        )
    decision["agent_instructions"] = instr

    decision.setdefault(
        "rationale",
        "Plan derived from user intent and available data flags.",
    )
    decision["raw_response"] = raw
    return decision


def _default_role(user_intent: str) -> str:
    short = user_intent.strip().split(".")[0][:120]
    return short or "Sepsis-3 + SEP-1 audit of the selected admission(s)."
