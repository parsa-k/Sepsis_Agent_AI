"""
LangGraph orchestrator — wires the Orchestrator and all specialised agents
into a state machine with a conditional reflection loop.

Flow:
  data_loader
    -> orchestrator_plan
    -> vitals -> lab -> microbiology -> pharmacy -> history
    -> diagnostician -> compliance
    -> (reflect? -> back to vitals | orchestrator_synthesize -> END)
"""

from __future__ import annotations
import traceback
from typing import Literal

from langgraph.graph import StateGraph, END

from agents.state import SepsisState
from agents.orchestrator_agent import run_orchestrator_plan, run_orchestrator_synthesize
from agents.vitals_agent import run_vitals_agent
from agents.lab_agent import run_lab_agent
from agents.microbiology_agent import run_microbiology_agent
from agents.pharmacy_agent import run_pharmacy_agent
from agents.history_agent import run_history_agent
from agents.diagnostician_agent import run_diagnostician_agent
from agents.compliance_agent import run_compliance_agent

import db


MAX_REFLECTIONS = 3


# ── Node: load data from DuckDB ─────────────────────────────────────────────

def data_loader(state: SepsisState) -> dict:
    conn = db.get_conn()
    subject_id = state["subject_id"]
    hadm_id = state["hadm_id"]

    admission_df = db.find_patient(conn, subject_id=subject_id, hadm_id=hadm_id)
    patient_info = admission_df.to_dict(orient="records")[0] if len(admission_df) else {}

    vitals_df = db.get_vitals(conn, subject_id, hadm_id)
    labs_df = db.get_labs(conn, subject_id, hadm_id)
    micro_df = db.get_microbiology(conn, subject_id, hadm_id)
    rx_df = db.get_prescriptions(conn, subject_id, hadm_id)
    icu_df = db.get_icu_stays(conn, subject_id, hadm_id)
    dx_all = db.get_diagnoses(conn, subject_id)
    hist_df = db.get_historical_admissions(conn, subject_id, hadm_id)
    input_df = db.get_input_events(conn, subject_id, hadm_id)
    output_df = db.get_output_events(conn, subject_id, hadm_id)

    def _summarise(df, max_rows=80):
        if df is None or df.empty:
            return "No data available."
        return df.head(max_rows).to_string(index=False)

    conn.close()

    trace_entry = {
        "agent": "Data Loader",
        "content": (
            f"Loaded data for subject {subject_id}, admission {hadm_id}.\n"
            f"  Vitals rows: {len(vitals_df)}\n"
            f"  Lab rows: {len(labs_df)}\n"
            f"  Microbiology rows: {len(micro_df)}\n"
            f"  Prescriptions: {len(rx_df)}\n"
            f"  ICU stays: {len(icu_df)}\n"
            f"  Diagnoses: {len(dx_all)}\n"
            f"  Historical admissions: {len(hist_df)}\n"
            f"  Input events: {len(input_df)}\n"
            f"  Output events: {len(output_df)}"
        ),
    }

    return {
        "patient_info": patient_info,
        "vitals_raw": _summarise(vitals_df),
        "labs_raw": _summarise(labs_df),
        "microbiology_raw": _summarise(micro_df),
        "prescriptions_raw": _summarise(rx_df),
        "icu_stays_raw": _summarise(icu_df),
        "diagnoses_raw": _summarise(dx_all),
        "historical_admissions_raw": _summarise(hist_df),
        "input_events_raw": _summarise(input_df),
        "output_events_raw": _summarise(output_df),
        "reflection_count": 0,
        "agent_trace": [trace_entry],
    }


# ── Node factories (inject LLM + optional custom prompt at build time) ───────

def make_orchestrator_plan_node(llm, custom_prompt=None):
    def node(state: SepsisState) -> dict:
        return run_orchestrator_plan(state, llm, system_prompt=custom_prompt)
    return node


def make_orchestrator_synth_node(llm, custom_prompt=None):
    def node(state: SepsisState) -> dict:
        return run_orchestrator_synthesize(state, llm, system_prompt=custom_prompt)
    return node


def make_vitals_node(llm, custom_prompt=None):
    def node(state: SepsisState) -> dict:
        return run_vitals_agent(state, llm, system_prompt=custom_prompt)
    return node


def make_lab_node(llm, custom_prompt=None):
    def node(state: SepsisState) -> dict:
        return run_lab_agent(state, llm, system_prompt=custom_prompt)
    return node


def make_microbiology_node(llm, custom_prompt=None):
    def node(state: SepsisState) -> dict:
        return run_microbiology_agent(state, llm, system_prompt=custom_prompt)
    return node


def make_pharmacy_node(llm, custom_prompt=None):
    def node(state: SepsisState) -> dict:
        return run_pharmacy_agent(state, llm, system_prompt=custom_prompt)
    return node


def make_history_node(llm, custom_prompt=None):
    def node(state: SepsisState) -> dict:
        return run_history_agent(state, llm, system_prompt=custom_prompt)
    return node


def make_diagnostician_node(llm, custom_prompt=None):
    def node(state: SepsisState) -> dict:
        return run_diagnostician_agent(state, llm, system_prompt=custom_prompt)
    return node


def make_compliance_node(llm, custom_prompt=None):
    def node(state: SepsisState) -> dict:
        return run_compliance_agent(state, llm, system_prompt=custom_prompt)
    return node


# ── Reflection node ──────────────────────────────────────────────────────────

def reflection_node(state: SepsisState) -> dict:
    count = state.get("reflection_count", 0) + 1
    missing = state.get("missing_data_queries", "")
    note = (
        f"Reflection #{count}: Academic Sepsis-3 = YES but SEP-1 = NO. "
        f"Re-examining chart for missing documentation.\n"
        f"Missing items flagged: {missing}"
    )
    trace_entry = {"agent": "Reflection Loop", "content": note}
    existing_trace = state.get("agent_trace", [])
    return {
        "reflection_count": count,
        "reflection_notes": note,
        "agent_trace": existing_trace + [trace_entry],
    }


# ── Conditional edge after compliance ────────────────────────────────────────

def should_reflect(state: SepsisState) -> Literal["reflect", "synthesize"]:
    sepsis3 = state.get("sepsis3_met")
    sep1 = state.get("sep1_met")
    count = state.get("reflection_count", 0)

    if sepsis3 is True and sep1 is False and count < MAX_REFLECTIONS:
        return "reflect"
    return "synthesize"


# ── Graph builder ────────────────────────────────────────────────────────────

def build_graph(llm, custom_prompts: dict | None = None):
    """
    Build the LangGraph state machine.

    custom_prompts: optional dict with keys matching agent names:
        'orchestrator', 'vitals', 'lab', 'microbiology', 'pharmacy',
        'history', 'diagnostician', 'compliance'
    """
    prompts = custom_prompts or {}

    graph = StateGraph(SepsisState)

    # Nodes
    graph.add_node("data_loader", data_loader)
    graph.add_node("orchestrator_plan", make_orchestrator_plan_node(llm, prompts.get("orchestrator")))
    graph.add_node("vitals", make_vitals_node(llm, prompts.get("vitals")))
    graph.add_node("lab", make_lab_node(llm, prompts.get("lab")))
    graph.add_node("microbiology", make_microbiology_node(llm, prompts.get("microbiology")))
    graph.add_node("pharmacy", make_pharmacy_node(llm, prompts.get("pharmacy")))
    graph.add_node("history", make_history_node(llm, prompts.get("history")))
    graph.add_node("diagnostician", make_diagnostician_node(llm, prompts.get("diagnostician")))
    graph.add_node("compliance", make_compliance_node(llm, prompts.get("compliance")))
    graph.add_node("reflect", reflection_node)
    graph.add_node("orchestrator_synthesize", make_orchestrator_synth_node(llm, prompts.get("orchestrator")))

    # Edges: linear flow
    graph.set_entry_point("data_loader")
    graph.add_edge("data_loader", "orchestrator_plan")
    graph.add_edge("orchestrator_plan", "vitals")
    graph.add_edge("vitals", "lab")
    graph.add_edge("lab", "microbiology")
    graph.add_edge("microbiology", "pharmacy")
    graph.add_edge("pharmacy", "history")
    graph.add_edge("history", "diagnostician")
    graph.add_edge("diagnostician", "compliance")

    # Conditional: reflect or synthesize
    graph.add_conditional_edges(
        "compliance",
        should_reflect,
        {"reflect": "reflect", "synthesize": "orchestrator_synthesize"},
    )

    # Reflection loops back to first feature agent
    graph.add_edge("reflect", "vitals")

    # Synthesis -> END
    graph.add_edge("orchestrator_synthesize", END)

    return graph.compile()


def run_pipeline(llm, subject_id: int, hadm_id: int, custom_prompts: dict | None = None) -> SepsisState:
    app = build_graph(llm, custom_prompts=custom_prompts)
    initial_state: SepsisState = {
        "subject_id": subject_id,
        "hadm_id": hadm_id,
        "agent_trace": [],
        "reflection_count": 0,
    }
    try:
        result = app.invoke(initial_state)
        return result
    except Exception as e:
        error_trace = {
            "agent": "Pipeline Error",
            "content": f"Error: {e}\n{traceback.format_exc()}",
        }
        initial_state["agent_trace"] = [error_trace]
        initial_state["final_summary"] = f"Pipeline failed: {e}"
        return initial_state
