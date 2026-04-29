"""
Shared helpers for the five feature agents.

Encapsulates the boilerplate around:
  * building the human message from per-visit raw data,
  * injecting the orchestrator's per-agent instruction,
  * appending the history baseline (when present),
  * invoking the LLM,
  * piping the response through the MemoryManager.
"""

from __future__ import annotations

import json
from typing import Iterable

from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import SepsisState
from agents.memory_manager_agent import empty_output


TWO_PART_OUTPUT_INSTRUCTIONS = """
### Output format — STRICT, two parts only
You MUST respond with a single fenced ```json block whose object has exactly
these two top-level keys:

```json
{
  "part1_payload": {
    "actionable": { ... distilled, downstream-friendly facts ... },
    "source_records": [
      "<short reference to the patient record(s) that drove this fact>",
      ...
    ]
  },
  "part2_reasoning": "<your detailed reasoning, citing specific values and timestamps>"
}
```

Rules:
* `part1_payload.actionable` must be a compact dict — no prose paragraphs.
* `part1_payload.source_records` must list the specific records (e.g.
  ``"chartevents 2150-05-16 09:01 MAP=62"``) that justify each fact.
* `part2_reasoning` is the only place for long-form prose. It is hidden
  from downstream agents — never duplicate Part 1 content here.
* Output NOTHING outside the fenced JSON block.
"""


def visits_section(visits_data: dict, selected: list, key: str, label: str) -> str:
    """Format the per-visit raw text for one data domain."""
    if not selected:
        return f"### {label}\nNo visits selected."

    chunks: list[str] = []
    for hadm in selected:
        per = visits_data.get(hadm) or visits_data.get(str(hadm)) or {}
        raw = per.get(key) or "No data available."
        chunks.append(f"#### Visit {hadm}\n{raw}")
    return f"### {label}\n" + "\n\n".join(chunks)


def build_baseline_block(history_baseline: dict | None) -> str:
    if not history_baseline:
        return ""
    actionable = history_baseline.get("actionable") or {}
    src = history_baseline.get("source_records") or []
    return (
        "### Baseline context (from History Agent — Part 1 only)\n"
        f"```json\n{json.dumps(actionable, indent=2, default=str)}\n```\n"
        f"Source records: {src}\n"
    )


def orchestrator_instruction(state: SepsisState, agent_name: str) -> str:
    decision = state.get("orchestrator_decision") or {}
    instr = (decision.get("agent_instructions") or {}).get(agent_name, "")
    role = decision.get("role") or ""
    return (
        f"### Orchestrator role for this run\n{role}\n\n"
        f"### Your dynamic instruction\n{instr or '(none)'}\n"
    )


def is_active(state: SepsisState, agent_name: str) -> bool:
    decision = state.get("orchestrator_decision") or {}
    return agent_name in (decision.get("active_agents") or [])


def trace_append(state: SepsisState, entry: dict) -> list:
    return (state.get("agent_trace") or []) + [entry]


def selected_hadm_ids(state: SepsisState) -> list:
    sel = list(state.get("selected_hadm_ids") or [])
    if not sel and "hadm_id" in state:
        sel = [state["hadm_id"]]
    return sel


def standardise_or_raw(memory_manager, agent_name: str, raw_text: str) -> dict:
    if memory_manager is None:
        from agents.memory_manager_agent import MemoryManager
        return MemoryManager(patient_id="_anon")._parse_two_part(raw_text)
    return memory_manager.standardize_output(agent_name, raw_text)


def run_feature_agent(
    *,
    state: SepsisState,
    llm,
    memory_manager,
    agent_name: str,
    output_state_key: str,
    system_prompt: str,
    raw_data_sections: Iterable[tuple[str, str]],   # [(visits_data_key, label), ...]
    custom_system_prompt: str | None = None,
) -> dict:
    """Common run-loop. Returns the LangGraph state delta."""

    if not is_active(state, agent_name):
        out = empty_output(agent_name, "Not in orchestrator's active_agents list")
        if memory_manager is not None:
            memory_manager.record_skipped(agent_name, "not active")
        trace_entry = {
            "agent": agent_name,
            "kind": "skipped",
            "content": out["part1_payload"],
        }
        return {
            output_state_key: out,
            "agent_trace": trace_append(state, trace_entry),
        }

    selected = selected_hadm_ids(state)
    visits_data = state.get("visits_data") or {}

    sections = []
    for key, label in raw_data_sections:
        sections.append(visits_section(visits_data, selected, key, label))

    human_content = (
        f"## Patient #{state.get('subject_id')} — Visits {selected}\n\n"
        + orchestrator_instruction(state, agent_name)
        + build_baseline_block(state.get("history_baseline"))
        + "\n".join(sections)
        + "\n\nProvide your analysis in the required two-part JSON format now."
    )

    prompt = (custom_system_prompt or system_prompt).rstrip() + "\n" + TWO_PART_OUTPUT_INSTRUCTIONS

    if memory_manager is not None:
        memory_manager.standardize_input(agent_name, {"human": human_content})

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=human_content),
    ])
    envelope = standardise_or_raw(memory_manager, agent_name, response.content)

    trace_entry = {
        "agent": agent_name,
        "kind": "output",
        "content": envelope.get("part1_payload"),
    }
    return {
        output_state_key: envelope,
        "agent_trace": trace_append(state, trace_entry),
    }
