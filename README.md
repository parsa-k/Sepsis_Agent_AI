# Sepsis Diagnostic Agent

A LangGraph + Streamlit multi-agent system for sepsis analysis on MIMIC-IV, now fully refactored to a **dynamic orchestrator architecture** with:

- user-provided intent in the workspace,
- multi-visit routing with conditional History-first baseline extraction,
- strict two-part agent outputs (`Part 1 payload` + `Part 2 reasoning`),
- a new `MemoryManager` for I/O normalization and persistent run logs,
- a single master `Diagnoses Agent` replacing Diagnostician + Compliance.

---

## What Changed (Refactor Summary)

### 1) Dynamic orchestration instead of static chain

The Orchestrator no longer executes a fixed plan. It now reads:

- `user_intent` from the workspace text area,
- selected visits (`selected_hadm_ids`),
- per-visit data availability flags.

It returns a machine-readable decision:

- run role,
- active feature agents,
- per-agent dynamic instructions,
- whether `history_first` is required.

### 2) Conditional History routing

- **Single selected visit** -> History Agent is excluded.
- **Multiple selected visits** -> History Agent runs first, generates baseline Part 1 payload, and that baseline is propagated to downstream feature agents.

### 3) Two-part output contract for feature agents

Every feature agent output is normalized to:

```json
{
  "part1_payload": {
    "actionable": {},
    "source_records": []
  },
  "part2_reasoning": "..."
}
```

- Only `part1_payload` is passed downstream.
- `part2_reasoning` is retained for audit/review in Patient History.

### 4) New Memory Manager

`agents/memory_manager_agent.py` now:

- standardizes inter-agent input/output,
- parses and validates two-part responses,
- records events and outputs to:
  - `app_memory/<patient_id>/session_<timestamp>.jsonl` (event stream),
  - `app_memory/<patient_id>/session_<timestamp>.json` (session summary).

### 5) Master Diagnoses Agent

`agents/diagnoses_agent.py` replaces old Diagnostician + Compliance pair.
It consumes only Part 1 payloads from activated agents and produces:

- concise sepsis summary,
- patient state score (`1` good -> `5` critical),
- final diagnosis line,
- combined Sepsis-3 and SEP-1 assessment details,
- streamlined Part 1 trace.

---

## Current Pipeline

```text
data_loader
  -> orchestrator
    -> (if multi-visit) history -> propagate_baseline
    -> vitals -> lab -> microbiology -> pharmacy
    -> diagnoses
    -> END
```

No reflection loop exists in the current refactored flow.

---

## Current Project Structure (Relevant)

```text
agents/
  state.py
  graph.py
  memory_manager_agent.py
  orchestrator_agent.py
  history_agent.py
  vitals_agent.py
  lab_agent.py
  microbiology_agent.py
  pharmacy_agent.py
  diagnoses_agent.py
  _agent_utils.py

app/
  workspace.py
  history.py
  controller.py
  dashboard.py
  settings.py
  llm.py
  css.py
```

Removed:

- `agents/diagnostician_agent.py`
- `agents/compliance_agent.py`

---

## Workspace UX (Current)

`app/workspace.py` now includes:

- patient lookup (`subject_id` or `hadm_id`),
- **multi-visit selector** when multiple admissions exist,
- orchestrator instruction text area with default value:
  - `Review the selected clinical records and evaluate for Sepsis-3 and SEP-1 compliance. Highlight any critical anomalies.`
- results view that shows:
  - concise sepsis summary,
  - patient state score (1-5),
  - final diagnosis,
  - **Part 1 only** streamlined agent trace.

---

## Patient History UX (Current)

`app/history.py` now reads sessionized memory files from `app_memory/<patient_id>/`.

For each run/session it displays:

- run metadata (selected visits, active agents, score),
- per-agent Part 1 payload,
- per-agent **Part 2 reasoning** (fully visible),
- event stream from memory logs.

This is the canonical audit surface for long reasoning content.

---

## Setup & Run

```bash
conda activate AI
streamlit run app.py
```

---

## Test Status (Latest Refactor)

Implemented and passing in conda env `AI`:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

- 36 tests passed.
- Coverage includes memory manager parsing/logging, orchestrator routing rules, feature-agent two-part contract, diagnoses behavior, and graph-level single-vs-multi-visit execution.
