# Sepsis Diagnostic Agent - Backend & Agents Documentation

This document provides a detailed overview of the backend data architecture and the LangGraph-based multi-agent AI system powering the Sepsis Diagnostic Agent.

## Overview

The backend is now a **dynamic Orchestrator-controlled system** that:

- consumes user intent from the UI,
- supports one or multiple admissions per run,
- routes History conditionally,
- standardizes feature-agent output into a strict two-part schema,
- logs all inter-agent I/O through a dedicated Memory Manager,
- emits a single final diagnosis via a master Diagnoses Agent.

## Data Access Layer (`db.py`)

The application processes the massive **MIMIC-IV v3.1** dataset (hundreds of gigabytes) without crashing or requiring massive amounts of RAM.

- **DuckDB Integration**: The core of the data access layer. Instead of loading CSVs into Python memory (pandas), it executes SQL queries directly against compressed `.csv.gz` files.
- **Optimized Queries**: Functions in `db.py` (like `get_sepsis_ready_stays()`) use complex Common Table Expressions (CTEs) to perform single-pass scans across multiple large tables (labevents, chartevents, microbiologyevents, patients) to find viable patients and filter them by age (18-90) natively at the database level.
- **Data Loaders**: Domain extractors (`get_vitals()`, `get_labs()`, `get_microbiology()`, etc.) used by the graph's data-loader node for each selected visit.

## Multi-Agent Pipeline Architecture (`agents/graph.py`)

The system relies on **LangGraph** and currently executes this flow:

```text
data_loader
  -> orchestrator
    -> (if multi-visit) history -> propagate_baseline
    -> vitals -> lab -> microbiology -> pharmacy
    -> diagnoses
    -> END
```

There is no reflection loop in the current architecture.

### Agent State Schema (`agents/state.py`)
All nodes share `SepsisState` (TypedDict). Key groups:

- Inputs: `subject_id`, `selected_hadm_ids`, `user_intent`.
- Raw data: `visits_data` (per-visit domain summaries), `available_data_flags`.
- Orchestration: `orchestrator_decision`.
- Feature outputs: `history_output`, `vitals_output`, `lab_output`, `microbiology_output`, `pharmacy_output` (all two-part schema).
- Baseline propagation: `history_baseline` (Part 1 from History).
- Final reasoning: `diagnoses_output`.
- Audit/logging: `memory_session`, `agent_trace`.

### 1. Data Loader Node
A non-LLM node that:

- loads all selected admissions,
- builds per-visit raw summaries (`visits_data`),
- computes per-visit availability flags,
- records a loader trace entry through `MemoryManager`.

### 2. Orchestrator Agent (`agents/orchestrator_agent.py`)
The Orchestrator is now a **single dynamic planner**.

Inputs:

- user intent,
- selected visits,
- aggregated/per-visit data availability.

Output (`orchestrator_decision`):

- run role,
- `history_first` boolean,
- `active_agents` list,
- dynamic `agent_instructions` per active agent,
- rationale.

Hard routing rule:

- single visit -> history must not run,
- multiple visits -> history must run first.

### 3. Feature Agents
Feature agents are strict domain experts:

- `vitals_agent.py`
- `lab_agent.py`
- `microbiology_agent.py`
- `pharmacy_agent.py`
- `history_agent.py` (conditional; multi-visit only)

Each feature agent:

1. receives orchestrator instruction + relevant raw data (and baseline context where applicable),
2. returns standardized two-part output.

Two-part output contract:

```json
{
  "part1_payload": {
    "actionable": {},
    "source_records": []
  },
  "part2_reasoning": "..."
}
```

- Only `part1_payload` flows downstream.
- `part2_reasoning` is retained for audit/history UI.

### 4. Diagnoses Agent (Master) (`agents/diagnoses_agent.py`)

This replaces the old Diagnostician + Compliance pair.

It consumes **only Part 1 payloads** from activated feature agents and outputs:

- concise sepsis summary,
- patient state score (1-5),
- final diagnosis line,
- combined Sepsis-3 + SEP-1 interpretation,
- streamlined Part 1 trace.

Legacy modules removed:

- `agents/diagnostician_agent.py`
- `agents/compliance_agent.py`

### 5. Memory Manager (`agents/memory_manager_agent.py`)

New dedicated backend component responsible for:

- input standardization before feature-agent calls,
- output standardization/parsing to two-part format,
- skipped-agent normalization,
- event logging and session persistence to:
  - `app_memory/<patient_id>/session_<ts>.jsonl`
  - `app_memory/<patient_id>/session_<ts>.json`

This is the canonical source for backend conversation history and raw outputs.
