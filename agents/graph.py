"""
LangGraph wiring for the two-phase Sepsis Diagnostic system.

Flow
----
```
data_loader
    └─► orchestrator_preplan
            ├─► (multi-visit) ─► history ─► propagate_baseline ─► orchestrator_replan
            └─► (single visit) ───────────────────────────────► orchestrator_replan
                                              │
                                              ▼
            vitals ─► lab ─► microbiology ─► pharmacy
                                              │
                                              ▼
                                      diagnoses ─► evaluator ─► END
```

* Phase-1 Orchestrator (`orchestrator_preplan`) decides whether History
  must run first and crafts a tailored History-Agent instruction.
* The History Agent (multi-visit only) produces a Part-1 baseline, which
  `propagate_baseline` copies to `state['history_baseline']`.
* Phase-2 Orchestrator (`orchestrator_replan`) reads the baseline and the
  user intent, then finalises which feature agents to activate and
  writes their per-agent instructions.
* Feature agents read their dynamic instruction + baseline before
  emitting their two-part outputs.
* The Diagnoses Agent reads only Part-1 payloads.
* The Evaluator Agent gives a final green/yellow/red verdict on the run.

Each feature node is a no-op when ``orchestrator_decision.active_agents``
does not include it (handled inside `_agent_utils.run_feature_agent`).
The Memory Manager is constructed once per ``run_pipeline`` call and
injected into every agent via closure.
"""

from __future__ import annotations

import traceback

from langgraph.graph import StateGraph, END

from agents.state import SepsisState
from agents.memory_manager_agent import MemoryManager
from agents.orchestrator_agent import (
    run_orchestrator_preplan,
    run_orchestrator_replan,
    needs_history_first,
    propagate_history_baseline,
)
from agents.history_agent import run_history_agent
from agents.vitals_agent import run_vitals_agent
from agents.lab_agent import run_lab_agent
from agents.microbiology_agent import run_microbiology_agent
from agents.pharmacy_agent import run_pharmacy_agent
from agents.diagnoses_agent import run_diagnoses_agent
from agents.evaluator_agent import run_evaluator_agent

import db


# ── Data loader ─────────────────────────────────────────────────────────────

def make_data_loader(memory_manager: MemoryManager):
    def data_loader(state: SepsisState) -> dict:
        conn = db.get_conn()
        subject_id = state["subject_id"]
        selected = list(state.get("selected_hadm_ids") or [])
        if not selected and "hadm_id" in state:
            selected = [state["hadm_id"]]
        if not selected:
            patient_df = db.find_patient(conn, subject_id=subject_id)
            if patient_df.empty:
                conn.close()
                return {"error": "No admissions found for this subject."}
            selected = [int(patient_df.iloc[0]["hadm_id"])]

        primary_hadm = selected[0]
        admission_df = db.find_patient(conn, hadm_id=primary_hadm)
        patient_info = (
            admission_df.to_dict(orient="records")[0]
            if len(admission_df) else {}
        )

        visits_data: dict = {}
        flags: dict = {}
        for hadm in selected:
            adm_df = db.find_patient(conn, hadm_id=hadm)
            adm_info = adm_df.to_dict(orient="records")[0] if len(adm_df) else {}

            vitals_df = db.get_vitals(conn, subject_id, hadm)
            labs_df = db.get_labs(conn, subject_id, hadm)
            micro_df = db.get_microbiology(conn, subject_id, hadm)
            rx_df = db.get_prescriptions(conn, subject_id, hadm)
            icu_df = db.get_icu_stays(conn, subject_id, hadm)
            dx_df = db.get_diagnoses(conn, subject_id, hadm)
            input_df = db.get_input_events(conn, subject_id, hadm)
            output_df = db.get_output_events(conn, subject_id, hadm)

            visits_data[hadm] = {
                "admission_info": adm_info,
                "vitals_raw": _summarise(vitals_df),
                "labs_raw": _summarise(labs_df),
                "microbiology_raw": _summarise(micro_df),
                "prescriptions_raw": _summarise(rx_df),
                "icu_stays_raw": _summarise(icu_df),
                "diagnoses_raw": _summarise(dx_df),
                "input_events_raw": _summarise(input_df),
                "output_events_raw": _summarise(output_df),
            }
            flags[hadm] = {
                "vitals": not vitals_df.empty,
                "labs": not labs_df.empty,
                "microbiology": not micro_df.empty,
                "pharmacy": (not rx_df.empty)
                            or (not input_df.empty)
                            or (not output_df.empty),
                "icu": not icu_df.empty,
                "diagnoses": not dx_df.empty,
            }

        hist_df = db.get_historical_admissions(
            conn, subject_id, current_hadm_id=primary_hadm
        )
        conn.close()

        trace_entry = {
            "agent": "data_loader",
            "kind": "loaded",
            "content": {
                "subject_id": subject_id,
                "visits": selected,
                "row_counts": {
                    h: {k: (v != "No data available.") for k, v in d.items() if k.endswith("_raw")}
                    for h, d in visits_data.items()
                },
            },
        }
        memory_manager.record_event("data_loader", trace_entry["content"])

        return {
            "subject_id": subject_id,
            "selected_hadm_ids": selected,
            "patient_info": patient_info,
            "visits_data": visits_data,
            "available_data_flags": flags,
            "historical_admissions_raw": _summarise(hist_df),
            "memory_session": memory_manager.session_metadata(),
            "agent_trace": (state.get("agent_trace") or []) + [trace_entry],
        }
    return data_loader


def _summarise(df, max_rows: int = 80) -> str:
    if df is None or len(df) == 0:
        return "No data available."
    return df.head(max_rows).to_string(index=False)


# ── Node factories ──────────────────────────────────────────────────────────

def _make_node(fn, llm, memory_manager, prompt_key, custom_prompts):
    custom = (custom_prompts or {}).get(prompt_key)

    def node(state: SepsisState) -> dict:
        return fn(state, llm, memory_manager=memory_manager, system_prompt=custom)
    return node


# ── Graph builder ───────────────────────────────────────────────────────────

def build_graph(
    llm,
    memory_manager: MemoryManager,
    custom_prompts: dict | None = None,
):
    graph = StateGraph(SepsisState)

    graph.add_node("data_loader", make_data_loader(memory_manager))
    graph.add_node(
        "orchestrator_preplan",
        _make_node(run_orchestrator_preplan, llm, memory_manager,
                   "orchestrator", custom_prompts),
    )
    graph.add_node(
        "history",
        _make_node(run_history_agent, llm, memory_manager,
                   "history", custom_prompts),
    )
    graph.add_node("propagate_baseline", propagate_history_baseline)
    graph.add_node(
        "orchestrator_replan",
        _make_node(run_orchestrator_replan, llm, memory_manager,
                   "orchestrator_replan", custom_prompts),
    )
    graph.add_node(
        "vitals",
        _make_node(run_vitals_agent, llm, memory_manager,
                   "vitals", custom_prompts),
    )
    graph.add_node(
        "lab",
        _make_node(run_lab_agent, llm, memory_manager,
                   "lab", custom_prompts),
    )
    graph.add_node(
        "microbiology",
        _make_node(run_microbiology_agent, llm, memory_manager,
                   "microbiology", custom_prompts),
    )
    graph.add_node(
        "pharmacy",
        _make_node(run_pharmacy_agent, llm, memory_manager,
                   "pharmacy", custom_prompts),
    )
    graph.add_node(
        "diagnoses",
        _make_node(run_diagnoses_agent, llm, memory_manager,
                   "diagnoses", custom_prompts),
    )
    graph.add_node(
        "evaluator",
        _make_node(run_evaluator_agent, llm, memory_manager,
                   "evaluator", custom_prompts),
    )

    graph.set_entry_point("data_loader")
    graph.add_edge("data_loader", "orchestrator_preplan")

    # Phase-1 → (history?) → Phase-2
    graph.add_conditional_edges(
        "orchestrator_preplan",
        needs_history_first,
        {"history": "history", "replan": "orchestrator_replan"},
    )
    graph.add_edge("history", "propagate_baseline")
    graph.add_edge("propagate_baseline", "orchestrator_replan")

    # Phase-2 → feature agents (sequential, gated by active_agents)
    graph.add_edge("orchestrator_replan", "vitals")
    graph.add_edge("vitals", "lab")
    graph.add_edge("lab", "microbiology")
    graph.add_edge("microbiology", "pharmacy")
    graph.add_edge("pharmacy", "diagnoses")
    graph.add_edge("diagnoses", "evaluator")
    graph.add_edge("evaluator", END)

    return graph.compile()


# ── Public entry point ──────────────────────────────────────────────────────

def run_pipeline(
    llm,
    subject_id: int,
    selected_hadm_ids: list,
    user_intent: str = "",
    custom_prompts: dict | None = None,
    memory_manager: MemoryManager | None = None,
) -> SepsisState:
    """Execute the full pipeline and return the final state.

    ``selected_hadm_ids`` should always be a list (one or more visits).
    A new ``MemoryManager`` is created and used for the whole run unless
    one is supplied (useful for tests).
    """
    if memory_manager is None:
        memory_manager = MemoryManager(patient_id=subject_id)

    app = build_graph(llm, memory_manager, custom_prompts=custom_prompts)
    initial: SepsisState = {
        "subject_id": int(subject_id),
        "selected_hadm_ids": [int(h) for h in selected_hadm_ids],
        "user_intent": user_intent or "",
        "agent_trace": [],
    }
    try:
        result = app.invoke(initial)
    except Exception as exc:
        result = dict(initial)
        result["error"] = f"Pipeline failed: {exc}"
        result["agent_trace"] = (initial.get("agent_trace") or []) + [{
            "agent": "pipeline",
            "kind": "error",
            "content": f"{exc}\n{traceback.format_exc()}",
        }]
        memory_manager.record_event(
            "pipeline_error", {"error": str(exc),
                               "trace": traceback.format_exc()},
        )

    memory_manager.finalise(result)
    return result
