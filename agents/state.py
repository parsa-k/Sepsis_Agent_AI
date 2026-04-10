"""
Central state schema for the Sepsis Diagnostic LangGraph workflow.

Every agent reads from / writes to this TypedDict.  The Orchestrator
coordinates the entire flow; five feature agents produce per-domain
analyses; the Diagnostician and Compliance agents produce final verdicts.
"""

from __future__ import annotations
from typing import TypedDict, Optional


class SepsisState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────────────
    subject_id: int
    hadm_id: int

    # ── Raw data from DuckDB (populated by data_loader) ──────────────────────
    patient_info: dict
    vitals_raw: str
    labs_raw: str
    microbiology_raw: str
    prescriptions_raw: str
    input_events_raw: str
    output_events_raw: str
    icu_stays_raw: str
    diagnoses_raw: str
    historical_admissions_raw: str

    # ── Orchestrator planning output ─────────────────────────────────────────
    orchestrator_plan: str

    # ── Feature agent outputs (one per clinical domain) ──────────────────────
    vitals_analysis: str
    lab_analysis: str
    microbiology_analysis: str
    pharmacy_analysis: str
    history_analysis: str

    # ── Diagnostician agent output ───────────────────────────────────────────
    sofa_score: str
    organ_dysfunction: str
    sepsis3_met: Optional[bool]
    sepsis3_reasoning: str

    # ── Compliance agent output ──────────────────────────────────────────────
    sirs_criteria: str
    lactate_timing: str
    sep1_met: Optional[bool]
    sep1_reasoning: str
    missing_data_queries: str

    # ── Reflection loop control ──────────────────────────────────────────────
    reflection_count: int
    reflection_notes: str

    # ── Final output (orchestrator synthesis) ────────────────────────────────
    final_academic_diagnosis: Optional[bool]
    final_sep1_compliance: Optional[bool]
    final_summary: str

    # ── Agent trace (for UI) ─────────────────────────────────────────────────
    agent_trace: list  # list[dict] — each entry is {"agent": str, "content": str}
