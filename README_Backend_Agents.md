# Sepsis Diagnostic Agent — Backend & Agents Documentation

This document describes the backend data-access layer and the LangGraph
multi-agent system after the **two-phase Orchestrator + Evaluator
Agent** refactor.

## Overview

The backend is a **two-phase Orchestrator-controlled pipeline** that:

- consumes free-text user intent from the UI,
- supports one or many admissions per run,
- routes the History Agent conditionally (multi-visit only),
- standardises every feature-agent output into a strict two-part
  schema (`part1_payload` / `part2_reasoning`),
- logs every inter-agent exchange through a dedicated Memory Manager,
- emits a final diagnosis + treatment plan from a master Diagnoses
  Agent,
- runs an Evaluator Agent at the end as a quality gate that emits a
  green / yellow / red flag and a per-agent performance report.

## Data Access Layer (`db.py`)

The application reads **MIMIC-IV v3.1** (hundreds of GB of compressed
CSV) without loading anything into RAM:

- **DuckDB Integration** — runs SQL directly against `.csv.gz` files.
- **Optimised queries** — `get_sepsis_ready_stays()` and friends use
  multi-table CTEs to filter at the storage layer.
- **Domain loaders** — `get_vitals`, `get_labs`, `get_microbiology`,
  `get_prescriptions`, `get_icu_stays`, `get_diagnoses`,
  `get_input_events`, `get_output_events`, `get_historical_admissions`.

The data loader node calls these for every selected visit before the
Orchestrator plans.

## Multi-Agent Pipeline (`agents/graph.py`)

```text
data_loader
  └─► orchestrator_preplan        (Phase 1)
        ├─► (multi-visit) ─► history ─► propagate_baseline ─► orchestrator_replan (Phase 2)
        └─► (single-visit)─────────────────────────────────► orchestrator_replan (Phase 2)
                                              │
                                              ▼
                              vitals ─► lab ─► microbiology ─► pharmacy
                                              │
                                              ▼
                                 diagnoses ─► evaluator ─► END
```

Every feature node is a no-op when its name is not in
`orchestrator_decision.active_agents`.

### Agent State Schema (`agents/state.py`)

All nodes share `SepsisState` (TypedDict). Key groups:

- **Inputs** — `subject_id`, `selected_hadm_ids`, `user_intent`.
- **Loaded raw data** — `visits_data`, `available_data_flags`,
  `historical_admissions_raw`, `patient_info`.
- **Orchestration** — `orchestrator_decision` (carries Phase-1 and
  Phase-2 fields plus per-phase `raw_response*`).
- **Feature outputs** — `history_output`, `vitals_output`,
  `lab_output`, `microbiology_output`, `pharmacy_output` (each a full
  two-part `AgentOutput`).
- **Baseline propagation** — `history_baseline` (Part-1 of History).
- **Final reasoning** — `diagnoses_output` (summary, score,
  Sepsis-3 / SEP-1 verdicts, `next_steps`, `short_term_treatment`,
  `mid_term_plan`, Part-1 trace).
- **Quality gate** — `evaluator_output` (`flag`, `task_executed`,
  `confidence`, `agent_reports`, `missing_data`,
  `improvement_recommendations`).
- **Audit / logging** — `memory_session`, `agent_trace`.

### 1. Data Loader Node

Non-LLM node that:

- loads all selected admissions and the historical-admissions table,
- builds per-visit raw text summaries (`visits_data`),
- computes per-visit availability flags,
- records a `data_loader` trace entry through `MemoryManager`.

### 2. Orchestrator Agent — two phases (`agents/orchestrator_agent.py`)

#### Phase 1 — `run_orchestrator_preplan`

- Inputs: `user_intent`, `selected_hadm_ids`, `available_data_flags`.
- Hard rules enforced post-LLM:
  - single visit ⇒ `history_first = false`, History Agent excluded;
  - multi-visit ⇒ `history_first = true`, with a concrete
    intent-aware instruction in `agent_instructions["history"]`.
- Leaves `active_agents` empty — Phase 2 will fill it.

#### Phase 2 — `run_orchestrator_replan`

- Runs after History (multi-visit) or directly after Phase 1
  (single-visit).
- Inputs: user intent, data flags, `history_baseline`, prior decision.
- Picks the final feature agents and writes per-agent
  `agent_instructions` (concrete, baseline-aware).
- Cannot include `history` in the feature pick set, but preserves
  `"history"` at the head of `active_agents` (when it ran) so the UI
  and audit log can render it.

#### Routing helper

`needs_history_first` returns `"history"` when `history_first` is
true and `"replan"` otherwise; the LangGraph builder uses it to wire
Phase-1 → (history?) → Phase-2.

### 3. History Agent (`agents/history_agent.py`)

Multi-visit baseline extractor. Activated only when the Orchestrator
decides `history_first = true`. It receives the user's intent + the
Phase-1 instruction and emits a Part-1 baseline (chronic conditions,
baseline organ function, prior-admissions summary). The post-history
node `propagate_history_baseline` copies its `part1_payload` to
`state["history_baseline"]` so feature agents can read deltas against
baseline.

### 4. Feature Agents

Strict domain experts driven by `_agent_utils.run_feature_agent`:

- `vitals_agent.py`        — HR, BP/MAP, Temp, RR, SpO₂, GCS,
- `lab_agent.py`           — WBC, lactate, creatinine, bilirubin,
  platelets, blood-gas,
- `microbiology_agent.py`  — cultures, organisms, sensitivities,
  infection source,
- `pharmacy_agent.py`      — antibiotics, vasopressors, fluids,
  urine output,
- `history_agent.py`       — conditional, multi-visit only.

Each one receives:

- the Orchestrator's per-agent instruction (from Phase 2 for feature
  agents, Phase 1 for History),
- the History baseline block (when present),
- the relevant per-visit raw text only.

Each one returns the two-part envelope:

```json
{
  "part1_payload": {
    "actionable": {},
    "source_records": []
  },
  "part2_reasoning": "..."
}
```

Only `part1_payload` flows downstream. `part2_reasoning` is retained
in the audit log and the History UI.

### 5. Diagnoses Agent (master) (`agents/diagnoses_agent.py`)

Replaces the old Diagnostician + Compliance pair. Consumes **only
Part-1 payloads** from activated feature agents (history + features)
and outputs:

- `summary`,
- `patient_score` (1=Good … 5=Critical),
- `final_diagnosis`,
- `details` (Sepsis-3 + SEP-1 interpretation),
- `sepsis3_met`, `sep1_compliant`,
- `next_steps` — immediate actions (0–6 h),
- `short_term_treatment` — acute plan (6–72 h),
- `mid_term_plan` — post-acute plan (day 3–30),
- `agent_trace_part1` — Part-1-only trace.

Legacy modules removed: `agents/diagnostician_agent.py`,
`agents/compliance_agent.py`.

### 6. Evaluator Agent (`agents/evaluator_agent.py`) — NEW

The pipeline's final quality gate. Inputs:

- the user's verbatim intent,
- a raw-data digest per visit,
- the Orchestrator's final decision,
- every agent's Part-1 payload,
- the Diagnoses Agent's full output.

Output (strict JSON):

```json
{
  "flag":              "green | yellow | red",
  "task_executed":     true,
  "confidence":        88,
  "overall_summary":   "...",
  "agent_reports": {
    "orchestrator": {"verdict": "ok|warn|fail", "notes": "..."},
    "history":      {"verdict": "ok|warn|fail", "notes": "..."},
    "vitals":       {"verdict": "ok|warn|fail", "notes": "..."},
    "lab":          {"verdict": "ok|warn|fail", "notes": "..."},
    "microbiology": {"verdict": "ok|warn|fail", "notes": "..."},
    "pharmacy":     {"verdict": "ok|warn|fail", "notes": "..."},
    "diagnoses":    {"verdict": "ok|warn|fail", "notes": "..."}
  },
  "missing_data":      ["..."],
  "improvement_recommendations": "Markdown bullet list"
}
```

Skipped agents are reported as `verdict: "ok"` with a "Skipped — not
applicable" note rather than `fail`. A deterministic post-processor
(`_normalise_evaluation`) clamps the flag to one of `{green, yellow,
red}` and the confidence to `[0, 100]` even when the LLM hallucinates.

### 7. Memory Manager (`agents/memory_manager_agent.py`)

Backend component that:

- standardises inputs before each agent call (`standardize_input`),
- parses raw LLM responses into the two-part envelope
  (`standardize_output`),
- records non-two-part outcomes such as the Orchestrator decision and
  the Evaluator verdict (`record_agent_outcome`),
- normalises skipped agents (`record_skipped`),
- persists everything to:
  - `app_memory/<patient_id>/session_<ts>.jsonl` — append-only event
    stream,
  - `app_memory/<patient_id>/session_<ts>.json` — consolidated session
    summary.

This is the canonical source for backend conversation history and
audit reporting.

## Tests

`tests/` covers the full backend:

- `test_orchestrator.py` — Phase-1 + Phase-2 logic, enforcement rules,
  routing helper, baseline propagation, garbage-input recovery.
- `test_feature_agents.py` — every domain agent + the Diagnoses Agent,
  including baseline injection and the "Part-1 only" downstream rule.
- `test_evaluator.py` — Evaluator parsing, normalisation, end-to-end
  with a fake LLM, and the rule that Part-2 reasoning never reaches
  the Evaluator's prompt.
- `test_graph.py` — graph compiles, single-visit + multi-visit runs,
  Phase-1 → history → Phase-2 trace, evaluator output presence,
  session-file persistence.
- `test_memory_manager.py` — parser fall-backs and persistence.
- `test_imports.py` — every module imports clean and obsolete modules
  remain removed.

Latest run in conda env `AI`: **53/53 tests passing.**
