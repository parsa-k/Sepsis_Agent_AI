"""
Central state schema for the Sepsis Diagnostic LangGraph workflow.

The pipeline has been refactored around a strict **two-part output**
contract for every feature agent:

    part1_payload   — distilled, actionable facts + source records
                       (the ONLY thing propagated downstream)
    part2_reasoning — detailed rationale (kept for the UI / audit trail
                       but never injected into another agent's prompt)

The Orchestrator dynamically decides which feature agents to activate
based on the user's free-text intent and the per-visit data flags. If
the user selected more than one visit, the History Agent runs first to
extract a baseline that the Orchestrator then propagates to the
remaining feature agents.
"""

from __future__ import annotations
from typing import TypedDict, Optional, Any


# ── Canonical two-part agent output ──────────────────────────────────────────

class AgentOutput(TypedDict, total=False):
    """Strict envelope every feature agent must produce."""
    part1_payload: dict
    """
    {
        "actionable": { ... distilled facts ... },
        "source_records": [ "<short record reference>", ... ]
    }
    """

    part2_reasoning: str
    """Long-form explanation — never propagated downstream."""

    skipped: bool
    """True when the orchestrator did not activate this agent."""

    agent_name: str
    parse_error: Optional[str]


# ── Orchestrator decision envelope ───────────────────────────────────────────

class OrchestratorDecision(TypedDict, total=False):
    role: str
    """High-level mission statement (e.g. 'Sepsis-3 + SEP-1 audit')."""

    user_intent: str
    """Verbatim user instruction echoed back for downstream context."""

    multi_visit: bool
    history_first: bool
    """When True the History Agent runs before any other feature agent."""

    active_agents: list  # list[str]
    """Subset of {'vitals','lab','microbiology','pharmacy'} to run."""

    agent_instructions: dict  # dict[str, str]
    """Per-agent dynamic instructions injected into their human messages."""

    rationale: str
    raw_response: str


# ── Memory-manager session metadata ──────────────────────────────────────────

class MemorySession(TypedDict, total=False):
    patient_id: str
    session_id: str
    directory: str
    log_path: str
    summary_path: str


# ── Master pipeline state ────────────────────────────────────────────────────

class SepsisState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────────
    subject_id: int
    selected_hadm_ids: list   # list[int] — visit(s) the user picked
    user_intent: str          # free-text instruction to the Orchestrator

    # ── Loaded raw data ──────────────────────────────────────────────────
    patient_info: dict
    visits_data: dict
    """
    {
        <hadm_id>: {
            "vitals_raw": str, "labs_raw": str, "microbiology_raw": str,
            "prescriptions_raw": str, "input_events_raw": str,
            "output_events_raw": str, "icu_stays_raw": str,
            "diagnoses_raw": str, "admission_info": dict,
        }, ...
    }
    """
    available_data_flags: dict
    """
    {
        <hadm_id>: {
            "vitals": bool, "labs": bool, "microbiology": bool,
            "pharmacy": bool, "icu": bool, "diagnoses": bool,
        }, ...
    }
    """
    historical_admissions_raw: str

    # ── Orchestrator output ──────────────────────────────────────────────
    orchestrator_decision: dict   # OrchestratorDecision

    # ── History baseline (propagated downstream when multi-visit) ────────
    history_output: dict          # full AgentOutput
    history_baseline: dict        # = history_output['part1_payload']

    # ── Feature agent outputs (each is a full AgentOutput) ───────────────
    vitals_output: dict
    lab_output: dict
    microbiology_output: dict
    pharmacy_output: dict

    # ── Diagnoses Agent (master) output ──────────────────────────────────
    diagnoses_output: dict
    """
    {
        "summary":              str,
        "patient_score":        int,    # 1 (Good) … 5 (Critical)
        "final_diagnosis":      str,
        "details":              str,
        "sepsis3_met":          bool|None,
        "sep1_compliant":       bool|None,
        "next_steps":           str,    # 0–6 h actions (Markdown)
        "short_term_treatment": str,    # 6–72 h plan (Markdown)
        "mid_term_plan":        str,    # day 3–30 plan (Markdown)
        "agent_trace_part1": [
            {"agent": str, "part1_payload": dict}, ...
        ]
    }
    """

    # ── Evaluator Agent output (final quality gate) ──────────────────────
    evaluator_output: dict
    """
    {
        "flag":               "green" | "yellow" | "red",
        "task_executed":      bool,
        "confidence":         int,            # 0..100
        "overall_summary":    str,
        "agent_reports": {
            "<agent_name>": {"verdict": "ok|warn|fail", "notes": str},
            ...
        },
        "missing_data":       list[str],
        "improvement_recommendations": str   # Markdown
    }
    """

    # ── Memory manager + trace ───────────────────────────────────────────
    memory_session: dict  # MemorySession
    agent_trace: list     # list[{"agent": str, "kind": str, "content": Any}]

    # ── Convenience flags ────────────────────────────────────────────────
    error: Optional[str]
