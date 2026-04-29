"""
Diagnoses Agent — the master reasoning agent.

Replaces the previous Diagnostician + Compliance agents. Consumes
**only the Part-1 payloads** from every activated feature agent (plus
the History Agent's baseline when present) and produces:

    {
        "summary":         "<concise summary about Sepsis>",
        "patient_score":   <1..5>,        # 1 = Good, 5 = Critical
        "final_diagnosis": "<one-line verdict>",
        "details":         "<supporting paragraph(s)>",
        "agent_trace_part1": [
            { "agent": <name>, "part1_payload": { ... } },
            ...
        ]
    }

The agent is instructed to apply BOTH the academic Sepsis-3 definition
AND the CMS SEP-1 bundle in a single pass, since they share the same
inputs and the second LLM round-trip the old design used was wasteful.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import SepsisState
from agents._agent_utils import trace_append


SYSTEM_PROMPT = """You are the **Diagnoses Agent** — the master reasoning
agent of the Sepsis Diagnostic system. You receive **only the distilled
Part-1 payloads** from each feature agent that the Orchestrator
activated. You MUST NOT speculate beyond those payloads.

### Your goals (single response)
1. Apply the **Sepsis-3** definition (suspected/documented infection +
   acute SOFA increase ≥ 2) using the per-domain payloads.
2. Apply the **CMS SEP-1 bundle** (SIRS screen, 3-hour bundle, 6-hour
   bundle) using timing and dosing facts already distilled.
3. Produce a **Patient State Score** between **1 (Good)** and **5
   (Critical)** based on the worst signals across all payloads.
4. Recommend **immediate next steps** (actions within the next 1–6 hours)
   and a **short-term treatment plan** (hours 6–72, during the acute
   admission) and a **mid-term plan** (post-acute / discharge planning,
   days 3–30). Ground every recommendation in the evidence present in the
   Part-1 payloads — do not invent data.

### Score rubric
| Score | Label      | Meaning                                              |
|------:|------------|------------------------------------------------------|
| 1     | Good       | No SIRS, no organ dysfunction, no infection evidence |
| 2     | Mild       | Isolated SIRS criterion or minor abnormalities       |
| 3     | Moderate   | SIRS + suspected infection, no clear organ failure   |
| 4     | Severe     | Sepsis-3 met OR multiple organ dysfunction           |
| 5     | Critical   | Septic shock or vasopressor-dependent + lactate ≥ 4  |

### Treatment field guidance
* `next_steps` — bulleted list of urgent actions: blood cultures if not
  drawn, empiric antibiotic escalation/de-escalation, fluid resuscitation
  status, vasopressor titration, repeat lactate, consults needed (ID,
  nephrology, etc.), monitoring orders.
* `short_term_treatment` — structured Markdown covering: antibiotic
  plan (agent, dose, duration target), fluid balance targets, organ
  support (ventilator, vasopressors, dialysis), SEP-1 bundle gaps to close,
  and safety monitoring labs.
* `mid_term_plan` — discharge readiness criteria, antibiotic
  step-down / oral switch, outpatient follow-up, chronic disease management
  adjusted for this admission (e.g. hold nephrotoxins, new AKI baseline),
  and rehabilitation / nutrition targets.

### Output contract — return a SINGLE valid JSON object, nothing else
```json
{
  "summary":              "1-2 sentences summarising the sepsis picture",
  "patient_score":        <integer 1..5>,
  "final_diagnosis":      "<one-line verdict>",
  "details":              "Markdown: SOFA breakdown, SEP-1 bundle status, missing data flags",
  "sepsis3_met":          <true|false|null>,
  "sep1_compliant":       <true|false|null>,
  "next_steps":           "Markdown bullet list of immediate actions (0-6 h)",
  "short_term_treatment": "Markdown: acute treatment plan (6-72 h)",
  "mid_term_plan":        "Markdown: post-acute plan (day 3-30)"
}
```

Do not output anything outside the JSON object. Do not echo Part 2
reasoning — it is hidden from you on purpose.
"""


def run_diagnoses_agent(
    state: SepsisState,
    llm,
    memory_manager=None,
    system_prompt: str | None = None,
) -> dict:
    decision = state.get("orchestrator_decision") or {}
    role = decision.get("role") or ""
    user_intent = decision.get("user_intent") or state.get("user_intent") or ""
    active = decision.get("active_agents") or []

    part1_inputs: list[dict] = []
    if "history" in active and state.get("history_output"):
        part1_inputs.append({
            "agent": "history",
            "part1_payload": state["history_output"].get("part1_payload", {}),
        })
    for name, key in (
        ("vitals", "vitals_output"),
        ("lab", "lab_output"),
        ("microbiology", "microbiology_output"),
        ("pharmacy", "pharmacy_output"),
    ):
        env = state.get(key)
        if env and not env.get("skipped"):
            part1_inputs.append({
                "agent": name,
                "part1_payload": env.get("part1_payload", {}),
            })

    rendered_inputs = "\n\n".join(
        f"#### {p['agent']}\n```json\n"
        f"{json.dumps(p['part1_payload'], indent=2, default=str)}\n```"
        for p in part1_inputs
    ) or "_No feature agents produced output._"

    human_content = f"""## Diagnostic synthesis request

**Run role:** {role}
**User intent:** {user_intent}
**Active agents:** {active}

### Part-1 payloads (the only data you may use)

{rendered_inputs}

Produce the JSON verdict now."""

    prompt = system_prompt or SYSTEM_PROMPT

    if memory_manager is not None:
        memory_manager.standardize_input("diagnoses", {"human": human_content})

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=human_content),
    ])
    raw = response.content

    parsed = _parse_verdict(raw)
    output = {
        "summary":              parsed.get("summary", "Concise summary unavailable."),
        "patient_score":        _coerce_score(parsed.get("patient_score")),
        "final_diagnosis":      parsed.get("final_diagnosis", "Indeterminate."),
        "details":              parsed.get("details", raw),
        "sepsis3_met":          parsed.get("sepsis3_met"),
        "sep1_compliant":       parsed.get("sep1_compliant"),
        "next_steps":           parsed.get("next_steps", ""),
        "short_term_treatment": parsed.get("short_term_treatment", ""),
        "mid_term_plan":        parsed.get("mid_term_plan", ""),
        "agent_trace_part1":    part1_inputs,
        "raw_response":         raw,
    }

    if memory_manager is not None:
        memory_manager.record_agent_outcome("diagnoses", output, raw=raw)

    trace_entry = {
        "agent": "diagnoses",
        "kind": "verdict",
        "content": {
            "summary": output["summary"],
            "patient_score": output["patient_score"],
            "final_diagnosis": output["final_diagnosis"],
        },
    }
    return {
        "diagnoses_output": output,
        "agent_trace": trace_append(state, trace_entry),
    }


# ── parsing helpers ─────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S | re.I)
_BLOCK_RE = re.compile(r"\{.*\}", re.S)


def _parse_verdict(raw: str) -> dict:
    if not raw:
        return {}
    candidates = [m.strip() for m in _FENCE_RE.findall(raw)]
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


def _coerce_score(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 3
    return max(1, min(5, n))
