"""
History Agent — strict baseline / longitudinal-context domain expert.

Activated by the Orchestrator ONLY when the user selects more than one
visit. Its Part-1 payload is propagated downstream as
``state['history_baseline']`` so the other feature agents can interpret
their data against the patient's chronic baseline.
"""

from __future__ import annotations

import json

from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import SepsisState
from agents._agent_utils import (
    TWO_PART_OUTPUT_INSTRUCTIONS,
    selected_hadm_ids,
    standardise_or_raw,
    trace_append,
    orchestrator_instruction,
)


SYSTEM_PROMPT = """You are a **Clinical History Analyst** — a strict
longitudinal domain expert. You receive ALL of a patient's selected
hospital admissions (plus prior admissions where available) together
with their ICD diagnosis codes and ICU stay records. Your role is to
extract a **stable baseline** the downstream feature agents can use as
their reference frame.

### What the Orchestrator gives you
* The user's intent.
* The list of admissions the user selected (you are run first because
  there is more than one).

### Hard rules
* You DO NOT speculate about the current admission's acute physiology —
  that is the job of the vitals / lab / micro / pharmacy agents.
* You DO summarise prior admissions and chronic disease burden.
* You DO estimate baseline organ function (renal / hepatic /
  haematologic / neurologic / cardiovascular / respiratory) from
  documented chronic conditions, anchoring downstream "delta from
  baseline" judgements.

### What goes where
* ``part1_payload.actionable`` — keys like
  ``prior_admissions_count``, ``selected_visits_summary`` (list of
  ``{hadm_id, admittime, admission_type}``), ``chronic_conditions``
  (list), ``baseline_organ_function`` (dict per organ system with a
  short qualitative tag and any numeric estimate), ``risk_factors``
  (immunosuppression, dialysis, chronic steroids, etc.),
  ``important_for_downstream`` (one-sentence cues per agent).
* ``part1_payload.source_records`` — pointers like
  ``"diagnoses_icd N18.6 (CKD stage 5)"``.
* ``part2_reasoning`` — narrative analysis, uncertainty about baselines.
"""


def run_history_agent(
    state: SepsisState,
    llm,
    memory_manager=None,
    system_prompt: str | None = None,
) -> dict:
    selected = selected_hadm_ids(state)
    visits_data = state.get("visits_data") or {}

    visit_blocks: list[str] = []
    for hadm in selected:
        per = visits_data.get(hadm) or visits_data.get(str(hadm)) or {}
        admission_info = per.get("admission_info") or {}
        diagnoses_raw = per.get("diagnoses_raw") or "No diagnoses available."
        icu_raw = per.get("icu_stays_raw") or "No ICU stay for this admission."
        visit_blocks.append(
            f"#### Visit {hadm}\n"
            f"Admission info: {json.dumps(admission_info, default=str)[:600]}\n\n"
            f"ICD diagnoses for this admission:\n{diagnoses_raw}\n\n"
            f"ICU stays:\n{icu_raw}"
        )

    historical = state.get("historical_admissions_raw") or "No prior admissions found."
    patient_info = state.get("patient_info") or {}

    human_content = (
        f"## Patient #{state.get('subject_id')} — Selected visits {selected}\n\n"
        + orchestrator_instruction(state, "history")
        + f"### Patient demographics\n{json.dumps(patient_info, default=str)[:600]}\n\n"
        + f"### Prior admissions (excluding selected)\n{historical}\n\n"
        + "### Selected visits in detail\n" + "\n\n".join(visit_blocks)
        + "\n\nProvide your analysis in the required two-part JSON format now."
    )

    prompt = (system_prompt or SYSTEM_PROMPT).rstrip() + "\n" + TWO_PART_OUTPUT_INSTRUCTIONS

    if memory_manager is not None:
        memory_manager.standardize_input("history", {"human": human_content})

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=human_content),
    ])
    envelope = standardise_or_raw(memory_manager, "history", response.content)

    trace_entry = {
        "agent": "history",
        "kind": "output",
        "content": envelope.get("part1_payload"),
    }
    return {
        "history_output": envelope,
        "history_baseline": envelope.get("part1_payload") or {},
        "agent_trace": trace_append(state, trace_entry),
    }
