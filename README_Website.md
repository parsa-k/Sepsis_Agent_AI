# Sepsis Diagnostic Agent — Website & UI Documentation

This document describes the **current** Streamlit frontend after the
two-phase Orchestrator + Evaluator Agent refactor.

## Overview

The frontend is built with **Streamlit** and provides:

- patient discovery and chart exploration,
- prompt-level control of every agent (including both Orchestrator
  phases, the shared output-format template, and the new Evaluator),
- dynamic orchestration input from the user,
- clear separation between short actionable outputs (Part 1) and deep
  reasoning (Part 2),
- final pipeline-quality flag (green / yellow / red) from the new
  Evaluator Agent, surfaced on both Workspace and Patient History,
- sessionised historical audit from `app_memory/`.

## UI Architecture

The UI is modular and page-oriented:

- **`app.py`** — slim entry point. Sets the wide layout, injects the
  custom CSS, renders the sidebar navigation, and dispatches to a page
  module. Includes a "page-changed" guard that forces a clean rerun on
  navigation to prevent widget-state bleed between pages.
- **`app/`** — per-page modules and shared utilities.
  - **`css.py`** — `CUSTOM_CSS` constant: the "Medical Dark" design
    system (CSS variables, sidebar, nav buttons, metric cards, badges,
    tabs, expanders, dataframes, scrollbars).
  - **`secrets.py`** — `.secrets.json` load/save and session-state
    init.
  - **`llm.py`** — LLM instantiation, session caching, and 429
    fallback (Gemini family).

## Technologies

- **Streamlit** — reactive UI framework.
- **Plotly Express** — interactive charts.
- **Vanilla CSS** — injected via markdown for a polished theme.

## Application Pages

### 1. Settings & Configuration (`app/settings.py`)

LLM configuration:

- provider selection (Google Gemini default, OpenAI, Anthropic),
- model selection (override defaults),
- API-key management with `.secrets.json` persistence,
- "Save / Load Keys" buttons,
- connection test (Gemini fallback chain on 429).

### 2. Dataset Dashboard (`app/dashboard.py`)

MIMIC-IV exploration tool:

- metric cards (patients, admissions, mean age, mortality),
- gender distribution donut + top admission-types bar (Plotly,
  themed),
- paginated patient browser with filters (Patient Type, Admission
  Type, **Sepsis-ready** filter that requires complete sepsis-3 input
  data for adults 18–90),
- patient drill-down with 9 expanders (Demographics, Admissions, ICU,
  Diagnoses, Labs, Vitals, Prescriptions, Microbiology, Input
  Events) plus a Plotly vitals-over-time chart and completeness
  summary.

### 3. Agent Controller (`app/controller.py`)

Tab-per-agent prompt editor on a top toolbar. Each tab shows the
agent's description, an editable system prompt, "Save" and "Reset"
buttons, a live MODIFIED / SAVED indicator, and a side-by-side diff
expander against the default prompt.

Editable agents (in tab order):

1. **Orchestrator (Phase 1)** — pre-plan; gates History and crafts
   the History Agent's instruction.
2. **Orchestrator (Phase 2)** — re-plan; picks active feature agents
   and writes their per-agent baseline-aware instructions.
3. **History Agent** — multi-visit baseline extractor.
4. **Vitals Agent** — HR/BP/MAP/Temp/RR/SpO₂/GCS specialist.
5. **Lab Agent** — WBC/lactate/creatinine/bilirubin/platelets/gas.
6. **Microbiology Agent** — cultures, organisms, sensitivities.
7. **Pharmacy Agent** — antibiotics, vasopressors, fluids, urine.
8. **Diagnoses Agent** — master reasoning + Sepsis-3 / SEP-1 +
   immediate / short-term / mid-term treatment plans.
9. **Evaluator Agent** — final quality gate (green / yellow / red).
10. **Output Format (shared)** — the two-part output template
    appended to every feature agent's system prompt at runtime.

Edits are saved to session state (`prompt_*` keys) and to
`custom_prompts.json`. The graph reads them via the `custom_prompts`
parameter.

### 4. Agent Workspace (`app/workspace.py`)

Where the clinical pipeline is run on a selected patient.

- **Search** — by `subject_id` or `hadm_id`.
- **Patient banner** — quick demographics + admission summary.
- **Multi-visit selector** — pick one or more admissions for the run.
- **Orchestrator instruction input** — text area; default:
  > Review the selected clinical records and evaluate for Sepsis-3
  > and SEP-1 compliance. Highlight any critical anomalies.
- **Execution** — invokes the dynamic LangGraph (Phase 1 → optional
  History → Phase 2 → features → Diagnoses → Evaluator).
- **Results Display** (top to bottom):
  - **Evaluator flag banner** — green / yellow / red with confidence
    badge and one-line summary,
  - sepsis summary card,
  - Patient State Score (1=Good … 5=Critical),
  - final diagnosis,
  - **Treatment Recommendations** tabs: Immediate (0–6 h),
    Short-term (6–72 h), Mid-term (Day 3–30),
  - streamlined **Part 1 only** agent trace,
  - full **Evaluator report** — per-agent verdict cards
    (`ok` / `warn` / `fail`) with notes, plus expandable missing-data
    list and improvement recommendations.

Part 2 reasoning is intentionally hidden in this page.

### 5. Patient History (`app/history.py`)

Audit / review of previously executed pipeline runs. Top toolbar with
five tabs after picking a patient + session.

1. **Summary** —
   - Evaluator flag banner at the top,
   - run metadata cards (session id, started/finished, visits, active
     agents),
   - Patient State Score + Sepsis-3 / SEP-1 verdict badges,
   - sepsis summary, user intent, final diagnosis,
   - Treatment Recommendations (3 tabs),
   - Orchestrator plan (active agents, history-first flag,
     rationale),
   - full **Evaluator report** with per-agent cards.
2. **Raw Data** — full per-visit MIMIC-IV data fetched on demand
   from DuckDB (vitals, labs, micro, prescriptions, ICU stays,
   I/O events, ICD diagnoses).
3. **Visualize** — interactive Plotly charts over any plottable
   domain, with series and visit pickers.
4. **Agents Report** — every agent in canonical order
   (`orchestrator → history → vitals → lab → microbiology →
   pharmacy → diagnoses → evaluator`); each card has an inner
   three-tab view (`Part 1 / Part 2 / Raw LLM response`) for
   two-part agents, or the full Evaluator banner + report for the
   Evaluator card.
5. **Log** — the append-only event stream from
   `session_<ts>.jsonl` with kind / agent filters
   (`input / output / outcome / event / skipped / error`).

## Notes on UI / Backend Contract

- Workspace shows **Part 1 only** for context efficiency.
- History reveals **Part 2** for deep inspection.
- Multi-visit selection controls whether the History Agent is
  invoked: 1 visit ⇒ skipped; 2+ visits ⇒ History runs first.
- The Evaluator's flag is the user-facing summary of run quality and
  is rendered on both the Workspace and Patient History → Summary
  pages, plus inside History → Agents Report.
