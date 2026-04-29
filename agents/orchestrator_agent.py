"""
Orchestrator Agent — the dynamic two-phase planner of the Sepsis Diagnostic
system.

Workflow
────────
Phase 1 — **Pre-plan** (`run_orchestrator_preplan`)
    Runs immediately after the Data Loader. Inputs:
        * the user's free-text intent,
        * selected visit IDs,
        * per-visit data-availability flags.
    Outputs a partial OrchestratorDecision that:
        * decides whether History must run first (multi-visit ⇒ yes,
          single-visit ⇒ no),
        * crafts a History-Agent instruction tailored to the user's
          intent (only when multi-visit).
    The post-history list of feature agents is left empty / provisional;
    that is finalised in Phase 2.

Phase 2 — **Re-plan** (`run_orchestrator_replan`)
    Runs AFTER the History Agent has produced its baseline (multi-visit
    runs) OR immediately after Phase 1 (single-visit runs). Inputs:
        * the user's intent,
        * data-availability flags,
        * the History Agent's Part-1 baseline (when present).
    Outputs the FINAL OrchestratorDecision:
        * `active_agents` — exact subset of {vitals, lab, microbiology,
          pharmacy} the system will run,
        * `agent_instructions[<name>]` — concrete, baseline-aware
          per-agent task strings that get injected into each feature
          agent's prompt,
        * `rationale` — short justification.

The two phases share state through `state['orchestrator_decision']`.
A deterministic fallback (`_enforce_rules`) ensures the pipeline always
runs even if the LLM produces malformed JSON.

The legacy `run_orchestrator` symbol is kept as an alias of the pre-plan
function for backwards compatibility with older tests / scripts.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import SepsisState


# ── System prompts ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the **Orchestrator (Phase 1 — pre-plan)** of a
multi-agent Sepsis Diagnostic system. You do NOT analyse clinical data
yourself — your only job in this phase is to decide whether the History
Agent must run first and, if so, craft a tight, intent-aware instruction
for it.

### Inputs you receive
* The **user intent** (free text — the primary task description).
* The **selected visits** (1 = single visit, 2+ = multi-visit).
* The **aggregated data-availability flags** across selected visits.

### Hard rules
1. If the user selected **exactly one** visit:
   - `history_first` MUST be `false`.
   - Do NOT include the History Agent.
   - Leave `agent_instructions.history` empty.
2. If the user selected **more than one** visit:
   - `history_first` MUST be `true`.
   - Provide a CONCRETE instruction in `agent_instructions.history`
     telling the History Agent EXACTLY which longitudinal facts to
     extract that are relevant to the user's intent. For example, if
     the user asks about CKD impact, ask History to surface the CKD
     stage trajectory and dialysis dependency from prior admissions.
3. Do not finalise the active feature agents yet — Phase 2 will pick
   those after seeing the baseline (or directly when single-visit).

### Output contract — return a SINGLE valid JSON object, nothing else
```json
{
  "role":          "<one-line mission statement>",
  "multi_visit":   <bool>,
  "history_first": <bool>,
  "agent_instructions": {
    "history": "<intent-aware instruction; empty for single-visit>"
  },
  "rationale": "<why this gating decision>"
}
```
"""


REPLAN_SYSTEM_PROMPT = """You are the **Orchestrator (Phase 2 — re-plan)**
of a multi-agent Sepsis Diagnostic system. Phase 1 has already happened.
The History Agent (when applicable) has produced a baseline summary that
is now part of your context. Your job in this phase is to:

1. Decide which feature agents (vitals / lab / microbiology / pharmacy)
   are actually needed to answer the user's intent.
2. Write a CONCRETE, BASELINE-AWARE instruction for each activated agent
   that:
     * is tightly scoped to the user's intent,
     * cites baseline facts when relevant ("baseline creatinine ~2.4 —
       evaluate AKI as deltas above this"),
     * tells the agent which records to focus on and which to ignore.
3. Only activate agents whose data is actually present (see
   availability flags).
4. Do NOT include `history` in active_agents at this phase — it has
   already run (or was correctly skipped).

### Output contract — return a SINGLE valid JSON object, nothing else
```json
{
  "role":          "<one-line mission statement, refined>",
  "active_agents": ["vitals", "lab", "microbiology", "pharmacy"],
  "agent_instructions": {
    "vitals":       "<concrete, baseline-aware task>",
    "lab":          "...",
    "microbiology": "...",
    "pharmacy":     "..."
  },
  "rationale": "<2-4 sentences explaining the picks and instruction logic>"
}
```
"""


_FEATURE_AGENTS = ("vitals", "lab", "microbiology", "pharmacy")


# ── Public node entry-points ────────────────────────────────────────────────

def run_orchestrator_preplan(
    state: SepsisState,
    llm,
    memory_manager=None,
    system_prompt: str | None = None,
) -> dict:
    """Phase 1 — gate the History Agent and craft its instruction."""
    prompt = system_prompt or SYSTEM_PROMPT

    selected = list(state.get("selected_hadm_ids") or [])
    if not selected and "hadm_id" in state:
        selected = [state["hadm_id"]]
    multi_visit = len(selected) > 1
    user_intent = (state.get("user_intent") or "").strip() or _DEFAULT_INTENT

    flags = state.get("available_data_flags", {})
    aggregated_flags = _aggregate_flags(flags, selected)

    human_content = _render_preplan_prompt(
        subject_id=state.get("subject_id"),
        selected=selected,
        multi_visit=multi_visit,
        user_intent=user_intent,
        aggregated_flags=aggregated_flags,
        per_visit_flags=flags,
    )

    if memory_manager is not None:
        memory_manager.standardize_input(
            "orchestrator_preplan",
            {"system_prompt_excerpt": prompt[:400], "human": human_content},
        )

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=human_content),
    ])
    raw = response.content

    parsed = _parse_decision(raw)
    decision = _enforce_preplan_rules(
        parsed,
        multi_visit=multi_visit,
        user_intent=user_intent,
        raw=raw,
    )

    if memory_manager is not None:
        memory_manager.record_agent_outcome(
            "orchestrator", decision, raw=raw,
        )

    trace_entry = {
        "agent": "Orchestrator",
        "kind": "preplan",
        "content": {
            "history_first": decision["history_first"],
            "history_instruction":
                decision.get("agent_instructions", {}).get("history", ""),
        },
    }
    existing = state.get("agent_trace", []) or []

    return {
        "orchestrator_decision": decision,
        "agent_trace": existing + [trace_entry],
    }


def run_orchestrator_replan(
    state: SepsisState,
    llm,
    memory_manager=None,
    system_prompt: str | None = None,
) -> dict:
    """Phase 2 — pick feature agents and write per-agent instructions."""
    prompt = system_prompt or REPLAN_SYSTEM_PROMPT

    selected = list(state.get("selected_hadm_ids") or [])
    if not selected and "hadm_id" in state:
        selected = [state["hadm_id"]]
    multi_visit = len(selected) > 1
    user_intent = (state.get("user_intent") or "").strip() or _DEFAULT_INTENT

    flags = state.get("available_data_flags", {})
    aggregated_flags = _aggregate_flags(flags, selected)

    history_baseline = state.get("history_baseline") or {}
    prior_decision = state.get("orchestrator_decision") or {}

    human_content = _render_replan_prompt(
        subject_id=state.get("subject_id"),
        selected=selected,
        multi_visit=multi_visit,
        user_intent=user_intent,
        aggregated_flags=aggregated_flags,
        per_visit_flags=flags,
        history_baseline=history_baseline,
        prior_decision=prior_decision,
    )

    if memory_manager is not None:
        memory_manager.standardize_input(
            "orchestrator_replan",
            {"system_prompt_excerpt": prompt[:400], "human": human_content},
        )

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=human_content),
    ])
    raw = response.content

    parsed = _parse_decision(raw)
    decision = _enforce_replan_rules(
        parsed,
        prior_decision=prior_decision,
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
        "kind": "replan",
        "content": {
            "active_agents": decision.get("active_agents", []),
            "agent_instructions": decision.get("agent_instructions", {}),
        },
    }
    existing = state.get("agent_trace", []) or []

    return {
        "orchestrator_decision": decision,
        "agent_trace": existing + [trace_entry],
    }


# Back-compat alias — older code/tests reference `run_orchestrator`.
run_orchestrator = run_orchestrator_preplan


# ── Routing helpers ─────────────────────────────────────────────────────────

def needs_history_first(state: SepsisState) -> str:
    """Conditional-edge function used by the LangGraph builder."""
    decision = state.get("orchestrator_decision") or {}
    if decision.get("history_first"):
        return "history"
    return "replan"


def propagate_history_baseline(state: SepsisState) -> dict:
    """Copy History Part-1 into ``history_baseline`` for downstream agents."""
    history_out = state.get("history_output") or {}
    baseline = history_out.get("part1_payload") or {}
    trace_entry = {
        "agent": "Orchestrator",
        "kind": "propagate_baseline",
        "content": {
            "baseline_keys": list(baseline.get("actionable", {}).keys())
        },
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


def _render_preplan_prompt(
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
    per_visit_lines: list[str] = []
    for hadm in selected:
        v = per_visit_flags.get(hadm, {}) or per_visit_flags.get(str(hadm), {}) or {}
        active = ", ".join(k for k, val in v.items() if val) or "<none>"
        per_visit_lines.append(f"- Visit {hadm}: {active}")

    return f"""## Phase 1 — pre-plan request

**Subject:** {subject_id}
**Selected visits ({len(selected)}):** {', '.join(str(h) for h in selected)}
**Multi-visit?** {multi_visit}

### User intent (verbatim)
{user_intent}

### Aggregated data availability (across selected visits)
{flags_lines}

### Per-visit availability
{chr(10).join(per_visit_lines) if per_visit_lines else '- (none)'}

Output the JSON for Phase 1 now. Remember: single visit ⇒
`history_first: false` and an empty history instruction; multi-visit ⇒
`history_first: true` AND a concrete history instruction tied to the
user intent."""


def _render_replan_prompt(
    *,
    subject_id,
    selected: list,
    multi_visit: bool,
    user_intent: str,
    aggregated_flags: dict,
    per_visit_flags: dict,
    history_baseline: dict,
    prior_decision: dict,
) -> str:
    flags_lines = "\n".join(
        f"- {k}: {'PRESENT' if v else 'ABSENT'}"
        for k, v in aggregated_flags.items()
    )
    per_visit_lines: list[str] = []
    for hadm in selected:
        v = per_visit_flags.get(hadm, {}) or per_visit_flags.get(str(hadm), {}) or {}
        active = ", ".join(k for k, val in v.items() if val) or "<none>"
        per_visit_lines.append(f"- Visit {hadm}: {active}")

    if history_baseline:
        baseline_block = (
            "### History baseline (Part 1 only — propagated by Phase 1)\n"
            f"```json\n{json.dumps(history_baseline, indent=2, default=str)}\n```"
        )
    else:
        baseline_block = (
            "### History baseline\n"
            "_None — single-visit run, History Agent was correctly skipped._"
        )

    role_hint = prior_decision.get("role") or ""

    return f"""## Phase 2 — re-plan request

**Subject:** {subject_id}
**Selected visits ({len(selected)}):** {', '.join(str(h) for h in selected)}
**Multi-visit?** {multi_visit}
**Phase-1 role hint:** {role_hint}

### User intent (verbatim)
{user_intent}

### Aggregated data availability (across selected visits)
{flags_lines}

### Per-visit availability
{chr(10).join(per_visit_lines) if per_visit_lines else '- (none)'}

{baseline_block}

Output the JSON for Phase 2 now: pick `active_agents` and write a
concrete, baseline-aware `agent_instructions` for each. Do NOT include
`history` — it has already run (or was correctly skipped)."""


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


def _enforce_preplan_rules(
    decision: dict,
    *,
    multi_visit: bool,
    user_intent: str,
    raw: str,
) -> dict:
    decision = dict(decision) if isinstance(decision, dict) else {}

    decision.setdefault("role", _default_role(user_intent))
    decision["multi_visit"]   = multi_visit
    decision["user_intent"]   = user_intent
    decision["history_first"] = bool(multi_visit)

    instr = decision.get("agent_instructions") or {}
    if not isinstance(instr, dict):
        instr = {}

    if multi_visit:
        if not str(instr.get("history", "")).strip():
            instr["history"] = (
                "Summarise prior admissions and chronic-disease burden, "
                "estimate baseline organ function, and surface the facts most "
                f"relevant to: {user_intent}"
            )
    else:
        instr.pop("history", None)

    decision["agent_instructions"] = instr
    decision["active_agents"] = []  # finalised in Phase 2
    decision.setdefault(
        "rationale",
        "Phase-1 gating derived from visit count and user intent.",
    )
    decision["raw_response_preplan"] = raw
    return decision


def _enforce_replan_rules(
    decision: dict,
    *,
    prior_decision: dict,
    multi_visit: bool,
    aggregated_flags: dict,
    user_intent: str,
    raw: str,
) -> dict:
    decision = dict(decision) if isinstance(decision, dict) else {}

    # Carry over fields from Phase 1 that Phase 2 should not override.
    decision.setdefault("role", prior_decision.get("role") or _default_role(user_intent))
    decision["multi_visit"]   = bool(multi_visit)
    decision["history_first"] = bool(multi_visit)
    decision["user_intent"]   = user_intent

    active = decision.get("active_agents") or []
    if not isinstance(active, list):
        active = []
    active = [str(a).lower().strip() for a in active]

    # Phase 2 cannot include history.
    active = [a for a in active if a != "history"]

    # Add data-present agents the LLM forgot.
    flag_to_agent = {
        "vitals": "vitals",
        "labs": "lab",
        "microbiology": "microbiology",
        "pharmacy": "pharmacy",
    }
    for flag, agent in flag_to_agent.items():
        if aggregated_flags.get(flag) and agent not in active:
            active.append(agent)

    valid = {"vitals", "lab", "microbiology", "pharmacy"}
    seen: set = set()
    cleaned: list = []
    for a in active:
        if a in valid and a not in seen:
            cleaned.append(a)
            seen.add(a)

    # Persist history in active_agents for downstream consumers (UI/audit)
    # — the graph already ran the History node; we mark it active for tracing.
    if multi_visit and "history" not in cleaned:
        cleaned = ["history"] + cleaned
    decision["active_agents"] = cleaned

    # Merge instructions from Phase 1 (history) and Phase 2 (features).
    prior_instr = (prior_decision.get("agent_instructions") or {})
    instr = decision.get("agent_instructions") or {}
    if not isinstance(instr, dict):
        instr = {}
    if multi_visit and "history" in prior_instr:
        instr.setdefault("history", prior_instr["history"])

    for agent in cleaned:
        if agent == "history":
            continue
        if not str(instr.get(agent, "")).strip():
            instr[agent] = (
                f"Focus on data relevant to: {user_intent}. "
                "Use any baseline context provided to interpret deltas."
            )
    decision["agent_instructions"] = instr

    decision.setdefault(
        "rationale",
        "Phase-2 picked feature agents from data flags + history baseline.",
    )
    decision["raw_response"] = raw
    return decision


def _default_role(user_intent: str) -> str:
    short = user_intent.strip().split(".")[0][:120]
    return short or "Sepsis-3 + SEP-1 audit of the selected admission(s)."


# ── Back-compat re-export of the legacy enforcement helper ─────────────────

def _enforce_rules(
    decision: dict,
    *,
    multi_visit: bool,
    aggregated_flags: dict,
    user_intent: str,
    raw: str,
) -> dict:
    """Legacy single-pass enforcement kept for older tests.

    Behaves like a combined Phase-1 + Phase-2 in one shot — useful when an
    LLM produces a fully-formed decision in a single call.
    """
    pre = _enforce_preplan_rules(
        decision,
        multi_visit=multi_visit,
        user_intent=user_intent,
        raw=raw,
    )
    post = _enforce_replan_rules(
        decision,
        prior_decision=pre,
        multi_visit=multi_visit,
        aggregated_flags=aggregated_flags,
        user_intent=user_intent,
        raw=raw,
    )
    # Merge: post-only fields win, but keep raw_response_preplan from pre.
    post.setdefault("raw_response_preplan", pre.get("raw_response_preplan", raw))
    return post
