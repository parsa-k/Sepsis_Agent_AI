# Sepsis Diagnostic Agent

An **Orchestrator-controlled multi-agent AI system** that reads a patient's chart from the **MIMIC-IV** clinical dataset, diagnoses **Sepsis** using the academic **Sepsis-3** criteria, and verifies whether the documentation satisfies US **CMS SEP-1** regulations.

The system uses **8 specialised LLM agents** — an Orchestrator that coordinates the workflow, five feature agents (each dedicated to a specific clinical data domain), and two reasoning agents (Diagnostician and Compliance). All agents are wired together in a **LangGraph state machine** with a cyclic reflection loop.

The codebase follows a **modular architecture**: the Streamlit UI is split into a thin `app.py` entry point backed by the `app/` package (7 focused modules), while all agent logic lives in `agents/` (one file per agent).

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Module Reference](#module-reference)
  - [app/ — UI Package](#app--ui-package)
  - [agents/ — Agent Package](#agents--agent-package)
  - [db.py — Data Access Layer](#dbpy--data-access-layer)
- [Dataset](#dataset)
- [Technology Stack](#technology-stack)
- [Setup & Installation](#setup--installation)
- [Running the Application](#running-the-application)
- [Application Sections](#application-sections)
  - [1. Settings & Configuration](#1-settings--configuration)
  - [2. Dataset Dashboard](#2-dataset-dashboard)
  - [3. Agent Controller](#3-agent-controller)
  - [4. Agent Workspace](#4-agent-workspace)
- [Agent Pipeline](#agent-pipeline)
  - [Data Loader](#data-loader)
  - [Orchestrator Agent](#orchestrator-agent)
  - [Vitals Agent](#vitals-agent)
  - [Lab Agent](#lab-agent)
  - [Microbiology Agent](#microbiology-agent)
  - [Pharmacy Agent](#pharmacy-agent)
  - [History Agent](#history-agent)
  - [Diagnostician Agent](#diagnostician-agent)
  - [Compliance Agent](#compliance-agent)
  - [Reflection Loop](#reflection-loop)
- [LangGraph State Schema](#langgraph-state-schema)
- [Clinical Background](#clinical-background)
- [Key Design Decisions](#key-design-decisions)
- [Example Usage](#example-usage)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     app.py  (slim entry point, ~57 lines)                    │
│         Bootstrap → Page Config → CSS → Sidebar → Page Dispatch             │
│                                                                              │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ app/settings │  │ app/dashboard │  │ app/controller│  │ app/workspace │  │
│  │  render()    │  │  render()     │  │  render()     │  │  render()     │  │
│  └──────┬───────┘  └──────┬────────┘  └──────┬────────┘  └───────┬──────┘  │
│         │                 │                   │                    │         │
│  ┌──────▼─────────────────▼───────────────────▼────────────────────▼──────┐  │
│  │  app/llm.py  (get_llm, Gemini fallback)    app/secrets.py (persist)   │  │
│  └──────┬─────────────────┬───────────────────┼────────────────────┬─────┘  │
│         │                 │                   │                    │         │
│  ┌──────▼─────────────────▼───────────────────▼────────────────────▼─────┐  │
│  │                    db.py  (DuckDB on .csv.gz — zero RAM load)         │  │
│  └──────┬─────────────────┬───────────────────┬────────────────────┬─────┘  │
│         │                 │                   │                    │         │
│  ┌──────▼─────────────────▼───────────────────▼────────────────────▼─────┐  │
│  │                   agents/graph.py  (LangGraph state machine)          │  │
│  │                                                                       │  │
│  │  data_loader ──► orchestrator (plan)                                  │  │
│  │                    │                                                   │  │
│  │       ┌────────────▼────────────────────────────┐                     │  │
│  │       │  vitals ► lab ► micro ► pharm ► history  │                    │  │
│  │       └────────────┬────────────────────────────┘                     │  │
│  │                    ▼                                                   │  │
│  │            diagnostician ──► compliance                               │  │
│  │                    │              │                                    │  │
│  │                    │    reflect ◄─┘  (if sep3=Y, sep1=N, count<3)     │  │
│  │                    │       │                                           │  │
│  │                    │       └──► vitals  (re-run all feature agents)    │  │
│  │                    ▼                                                   │  │
│  │          orchestrator (synthesis) ──► END                             │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │       LLM Provider  (Google Gemini / OpenAI / Anthropic Claude)     │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Agentic AI/
├── app.py                           # Slim entry point (~57 lines): config, CSS, sidebar, dispatch
├── db.py                            # DuckDB connection & all query functions
├── README.md                        # This file
├── .gitignore                       # Ignores .secrets.json, __pycache__, .env, mimiciv/
├── .secrets.json                    # Local API key store (auto-created, git-ignored)
│
├── app/                             # ── Streamlit UI package ──────────────────
│   ├── __init__.py                  # Package docstring
│   ├── css.py                       # CUSTOM_CSS constant (all stylesheet rules)
│   ├── secrets.py                   # API key persistence: load, save, init into session
│   ├── llm.py                       # LLM instantiation, Gemini model fallback logic
│   ├── settings.py                  # Settings & Configuration page (render)
│   ├── dashboard.py                 # MIMIC-IV Dashboard page (metrics, charts, patient browser)
│   ├── controller.py                # Agent Controller page (prompt editing for all 8 agents)
│   └── workspace.py                 # Agent Workspace page (search, run pipeline, view results)
│
├── agents/                          # ── LangGraph agent package ───────────────
│   ├── __init__.py
│   ├── state.py                     # SepsisState TypedDict (shared schema, ~30 fields)
│   ├── orchestrator_agent.py        # Central brain — planning & synthesis phases
│   ├── vitals_agent.py              # Vital signs analysis (chartevents)
│   ├── lab_agent.py                 # Lab results analysis (labevents)
│   ├── microbiology_agent.py        # Culture & infection analysis (microbiologyevents)
│   ├── pharmacy_agent.py            # Medications, fluids, urine output
│   ├── history_agent.py             # Clinical history & baseline organ function
│   ├── diagnostician_agent.py       # Sepsis-3 / SOFA scoring
│   ├── compliance_agent.py          # CMS SEP-1 bundle check
│   └── graph.py                     # LangGraph orchestrator + reflection loop
│
└── mimiciv/                         # ── MIMIC-IV dataset (git-ignored) ────────
    └── 3.1/
        ├── hosp/                    # Hospital module .csv.gz files
        │   ├── patients.csv.gz
        │   ├── admissions.csv.gz
        │   ├── labevents.csv.gz
        │   ├── d_labitems.csv.gz
        │   ├── diagnoses_icd.csv.gz
        │   ├── d_icd_diagnoses.csv.gz
        │   ├── prescriptions.csv.gz
        │   └── microbiologyevents.csv.gz
        └── icu/                     # ICU module .csv.gz files
            ├── chartevents.csv.gz
            ├── icustays.csv.gz
            ├── d_items.csv.gz
            ├── inputevents.csv.gz
            └── outputevents.csv.gz
```

---

## Module Reference

### app/ — UI Package

| Module             | Responsibility                                                                                                                   |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------|
| `app/css.py`       | Single `CUSTOM_CSS` constant containing all stylesheet rules (Inter font, metric cards, badges, sidebar theming, Material Icons protection). |
| `app/secrets.py`   | `load_saved()` / `save()` read/write `.secrets.json`; `init_into_session()` hydrates Streamlit session state on first run.       |
| `app/llm.py`       | `get_llm()` instantiates the active LangChain chat model from session state. `test_gemini_with_fallback()` tries multiple Gemini models on 429 errors. Exports `GEMINI_MODELS` and `MODEL_DEFAULTS`. |
| `app/settings.py`  | `render()` draws the Settings page: provider dropdown, model name input, API key fields, Save/Load buttons, connection test.     |
| `app/dashboard.py` | `render()` draws the Dashboard page: metric cards, Plotly charts, patient browser with ICU/admission/sepsis-ready filters, patient detail drill-down with 9 data sections and completeness summary. |
| `app/controller.py`| `render()` draws the Agent Controller: 8 expandable prompt editors with MODIFIED badges, status strip, reset buttons. `get_custom_prompts()` collects current prompts for the pipeline. |
| `app/workspace.py` | `render()` draws the Workspace: patient search, pipeline execution via `run_pipeline()`, verdict badges, agent trace expanders, synthesis summary. |

### agents/ — Agent Package

| Module                        | Responsibility                                                                                      |
|-------------------------------|-----------------------------------------------------------------------------------------------------|
| `agents/state.py`             | `SepsisState` TypedDict — the shared schema (~30 fields) that every graph node reads/writes.        |
| `agents/orchestrator_agent.py`| Planning phase (assess data, produce analysis plan) and synthesis phase (final structured report).   |
| `agents/vitals_agent.py`      | Analyses HR, BP/MAP, Temp, RR, SpO2, GCS from `chartevents`.                                       |
| `agents/lab_agent.py`         | Analyses WBC, Lactate, Creatinine, Bilirubin, Platelets, PaO2, Blood Gas from `labevents`.          |
| `agents/microbiology_agent.py`| Analyses cultures, organisms, sensitivities from `microbiologyevents`.                              |
| `agents/pharmacy_agent.py`    | Analyses antibiotics, vasopressors, IV fluids, urine output from `prescriptions`/`inputevents`/`outputevents`. |
| `agents/history_agent.py`     | Reviews prior admissions, ICD diagnoses, estimates baseline organ function.                          |
| `agents/diagnostician_agent.py`| Applies Sepsis-3 definition: infection evidence + acute SOFA >= 2 above baseline.                  |
| `agents/compliance_agent.py`  | Checks CMS SEP-1: SIRS screening, 3-hour and 6-hour bundle timing.                                 |
| `agents/graph.py`             | Builds the LangGraph `StateGraph` (11 nodes, conditional reflection edge), exposes `run_pipeline()`. |

### db.py — Data Access Layer

Central DuckDB interface. Queries compressed `.csv.gz` files directly — never loads full tables into Python memory. Key functions:

| Function                     | Purpose                                                                     |
|------------------------------|-----------------------------------------------------------------------------|
| `get_conn()`                 | Open a new DuckDB connection.                                               |
| `get_dashboard_metrics()`    | Aggregate statistics for the dashboard (patients, admissions, age, mortality). |
| `get_admission_types()`      | Distinct admission types for filter dropdowns.                              |
| `get_patients_filtered()`    | Paginated patient list with ICU, admission-type, and sepsis-ready filters.  |
| `get_patient_full_detail()`  | Full patient record across 9 data sections for the detail drill-down.       |
| `get_sepsis_ready_stays()`   | Single-pass CTE query checking 8 data-completeness requirements across 6 tables. |
| `find_patient()`             | Lookup by `subject_id` or `hadm_id`.                                        |
| `get_vitals()`, `get_labs()`, etc. | Per-domain queries used by the Data Loader node.                      |

---

## Dataset

**MIMIC-IV v3.1** — a freely-available, de-identified critical-care dataset from Beth Israel Deaconess Medical Center.

| Metric            | Value    |
|-------------------|----------|
| Total Patients    | 364,627  |
| Total Admissions  | 546,028  |
| Average Age       | 48.9     |
| Mortality Rate    | 2.2%     |

The dataset consists of large `.csv.gz` files (some multi-GB). DuckDB queries them directly without decompressing or loading into RAM.

---

## Technology Stack

| Layer          | Technology                                           |
|----------------|------------------------------------------------------|
| Frontend       | Streamlit (wide layout, custom CSS, Inter font)      |
| Data Querying  | DuckDB (SQL on compressed CSV)                       |
| AI Framework   | LangGraph + LangChain                                |
| LLM Providers  | Google Gemini (default), OpenAI, Anthropic Claude    |
| Visualisation  | Plotly (interactive pie + bar charts)                 |
| Package Mgmt   | Conda (environment name: `AI`)                       |
| Language        | Python 3.11+                                         |

---

## Setup & Installation

### 1. Activate the Conda environment

```bash
conda activate AI
```

### 2. Install dependencies

```bash
pip install streamlit duckdb langgraph langchain \
    langchain-openai langchain-anthropic langchain-google-genai \
    google-generativeai plotly
```

### 3. Verify installation

```bash
python -c "import streamlit, duckdb, langgraph, langchain, plotly; print('All imports OK')"
```

### 4. Ensure MIMIC-IV data exists

The application expects the dataset at `mimiciv/3.1/hosp/` and `mimiciv/3.1/icu/` relative to the project root:

```bash
ls mimiciv/3.1/hosp/patients.csv.gz
ls mimiciv/3.1/icu/chartevents.csv.gz
```

---

## Running the Application

```bash
conda activate AI
streamlit run app.py
```

The app will open at **http://localhost:8501**.

---

## Application Sections

### 1. Settings & Configuration

- **LLM Provider Dropdown** — defaults to **Google Gemini**; also supports OpenAI and Anthropic Claude.
- **Model Name** *(optional)* — leave blank to use the provider's default model (`gemini-2.5-flash`, `gpt-4o`, or `claude-sonnet-4-20250514`). Enter a custom model identifier to override.
- **API Key** — secure password input; stored in Streamlit session state.
- **Save / Load Keys** — persist API keys, provider, and model name to a local `.secrets.json` file so they survive across sessions. This file is git-ignored and never committed.
- **Connection Test** — sends a one-line test prompt to verify the key works. For Google Gemini, the test automatically falls back through multiple models (`gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.0-flash`, `gemini-2.0-flash-lite`) if the preferred model hits a 429 quota error, updating the active model to the first one that succeeds.

### 2. Dataset Dashboard

- **Metric Cards** — Total Patients, Total Admissions, Average Age, Mortality Rate (gradient-styled cards).
- **Gender Distribution** — interactive donut chart (Plotly).
- **Top Admission Types** — horizontal bar chart.
- **Patient Browser** with filters:
  - **Patient Type** — All / ICU patients only / Non-ICU patients only.
  - **Admission Type** — filter by any admission type in the dataset.
  - **Sepsis Data** — All / Sepsis-ready only (patients with complete data for Sepsis-3 diagnosis: blood cultures, antibiotics, PaO2, Platelets, Bilirubin, Creatinine, MAP, and GCS).
  - Paginated table (25 rows/page) showing subject ID, gender, age, death date, number of admissions, ICU stays, and last admission type.
- **Complete Patient Record** — click a row to expand a full view of that patient's data across **9 expandable sections**:
  - Demographics, Admissions, ICU Stays, Diagnoses (ICD), Laboratory Results, Vitals (Chart Events), Prescriptions, Microbiology Cultures, Input Events.
- **Data Completeness Summary** — green/orange panel listing which data sections are present and which are missing.

### 3. Agent Controller

- **Eight editable prompt panels** — one per agent (Orchestrator, Vitals, Lab, Microbiology, Pharmacy, History, Diagnostician, Compliance).
- Each panel shows the full system prompt in a text area; edits are saved to session state instantly.
- **MODIFIED badge** — appears when a prompt differs from its default.
- **Reset to Default** — per-agent or all-at-once.
- **Status summary strip** — two rows of four cards showing which agents use default vs. modified prompts.
- The Agent Workspace reads the latest prompts from the controller on every run via `controller.get_custom_prompts()`.

### 4. Agent Workspace

- **Search** — enter a `subject_id` or `hadm_id`.
- **Patient Banner** — displays demographics, admission type, and timestamps.
- **Run Agent** — triggers the full LangGraph pipeline using the latest prompts from the Agent Controller.
- **Results**:
  - **Final Verdict** — stylised YES/NO badges for Sepsis-3 and SEP-1.
  - **Agent Thought Process** — expandable traces showing each agent's reasoning (up to 10+ steps: Data Loader, Orchestrator Planning, Vitals, Lab, Microbiology, Pharmacy, History, Diagnostician, Compliance, and optionally Reflection iterations, then Orchestrator Synthesis).
  - **Reflection Count** — how many times the loop re-examined the chart.
  - **Summary** — final synthesised report from the Orchestrator.

---

## Agent Pipeline

### Data Loader

Not an LLM agent — queries DuckDB for the target patient/admission and produces text summaries:

| Data Source        | DuckDB Table              | Key Columns                              |
|--------------------|---------------------------|------------------------------------------|
| Demographics       | `patients` + `admissions` | age, gender, admit time, type, death time |
| Vitals             | `chartevents` + `d_items` | HR, BP, Temp, RR, SpO2, GCS              |
| Labs               | `labevents` + `d_labitems`| WBC, Lactate, Creatinine, Bilirubin, Platelets, BUN, Glucose, Electrolytes, Blood Gas, Bands |
| Microbiology       | `microbiologyevents`      | cultures, organisms, sensitivities       |
| Prescriptions      | `prescriptions`           | antibiotics, vasopressors, drugs         |
| ICU Stays          | `icustays`                | care unit, LOS, in/out times             |
| IV / Inputs        | `inputevents` + `d_items` | fluids, vasopressors, rates              |
| Output Events      | `outputevents` + `d_items`| urine output, drains                     |
| Diagnoses          | `diagnoses_icd` + `d_icd_diagnoses` | all ICD codes with descriptions |
| History            | `admissions` (prior)      | previous admission records               |

### Orchestrator Agent

**File:** `agents/orchestrator_agent.py`

The central brain that runs in two phases:

- **Planning Phase** (after data load): Receives raw data summaries, assesses which data domains are available vs. missing, and produces an analysis plan for downstream agents.
- **Synthesis Phase** (after all agents complete): Receives every agent's output, the Sepsis-3 verdict, and the SEP-1 verdict. Produces a structured final report with: Patient Overview, Data Availability Summary, Key Findings by Domain, Sepsis-3 Verdict, SEP-1 Compliance Verdict, and Clinical Notes.

### Vitals Agent

**File:** `agents/vitals_agent.py`

Analyses charted vital signs from `chartevents`:
- Latest values, trends, and critical flags for HR, BP/MAP, Temp, RR, SpO2, GCS.
- SOFA-relevant highlights: MAP for cardiovascular SOFA, GCS for neurologic SOFA.

### Lab Agent

**File:** `agents/lab_agent.py`

Analyses laboratory results from `labevents`:
- Latest values and trends for all sepsis-critical labs (WBC, Lactate, Creatinine, Bilirubin, Platelets, PaO2, Blood Gas, BUN, Glucose, Electrolytes, Bands, Troponin).
- SOFA bracket classification for each component lab.
- SIRS-relevant highlights for WBC, Bands, and pCO2.
- Lactate emphasis for SEP-1 bundle compliance.

### Microbiology Agent

**File:** `agents/microbiology_agent.py`

Analyses culture results from `microbiologyevents`:
- Culture summary (specimen types, dates, organisms).
- Infection assessment (documented vs. suspected).
- Antibiotic sensitivity profiles and resistance patterns.
- Blood culture timing relative to admission (SEP-1 bundle item 2).

### Pharmacy Agent

**File:** `agents/pharmacy_agent.py`

Analyses medications and fluid management from `prescriptions`, `inputevents`, and `outputevents`:
- Antibiotic summary (type, route, timing, broad-spectrum adequacy).
- Vasopressor summary (doses, duration — for cardiovascular SOFA).
- IV fluid resuscitation volumes (for SEP-1 6-hour bundle).
- Urine output trends and oliguria flags (for renal SOFA).

### History Agent

**File:** `agents/history_agent.py`

Reviews clinical history from `admissions`, `diagnoses_icd`, and `icustays`:
- Historical admissions summary.
- Chronic conditions (CKD, diabetes, COPD, heart failure, cirrhosis, etc.).
- Baseline organ function estimates for each SOFA system — critical for calculating delta-SOFA (acute change from baseline).

### Diagnostician Agent

**File:** `agents/diagnostician_agent.py`

Applies the **Sepsis-3** definition (Singer et al., JAMA 2016):

> Sepsis = suspected infection + acute organ dysfunction (SOFA >= 2 above baseline)

Uses outputs from all five feature agents to:
1. Score each SOFA component (Respiration, Coagulation, Liver, Cardiovascular, CNS, Renal).
2. Compute baseline SOFA from chronic conditions (History Agent).
3. Calculate delta-SOFA.
4. Assess infection presence (Microbiology + Pharmacy agents).
5. Conclude **Sepsis-3: YES** or **NO** with full reasoning.

### Compliance Agent

**File:** `agents/compliance_agent.py`

Checks the **CMS SEP-1** (Severe Sepsis / Septic Shock Early Management Bundle):

**SIRS Criteria (>= 2 required):**
| # | Criterion |
|---|-----------|
| 1 | Temperature > 38.3C or < 36.0C |
| 2 | Heart rate > 90 bpm |
| 3 | Respiratory rate > 20 or PaCO2 < 32 mmHg |
| 4 | WBC > 12K or < 4K or > 10% bands |

**3-Hour Bundle (from time zero):**
1. Lactate level measured
2. Blood cultures obtained before antibiotics
3. Broad-spectrum antibiotics administered

**6-Hour Bundle (if septic shock):**
4. 30 mL/kg crystalloid fluid resuscitation
5. Vasopressors if persistent hypotension
6. Repeat lactate if initial >= 2

### Reflection Loop

**Conditional edge in `agents/graph.py`:**

```
IF Sepsis-3 = YES AND SEP-1 = NO AND reflection_count < 3:
    -> Loop back to Vitals Agent (re-examine all clinical data)
ELSE:
    -> Orchestrator Synthesis -> END
```

This handles the common case where academic criteria are met but chart documentation doesn't clearly satisfy the regulatory bundle — the agents re-examine up to 3 times before the Orchestrator produces its final synthesis.

---

## LangGraph State Schema

Defined in `agents/state.py` as `SepsisState(TypedDict)`. All graph nodes read from and write to this shared state:

| Field Group           | Fields                                                                                       |
|-----------------------|----------------------------------------------------------------------------------------------|
| **Input**             | `subject_id`, `hadm_id`                                                                     |
| **Raw data** (from DuckDB) | `patient_info`, `vitals_raw`, `labs_raw`, `microbiology_raw`, `prescriptions_raw`, `input_events_raw`, `output_events_raw`, `icu_stays_raw`, `diagnoses_raw`, `historical_admissions_raw` |
| **Orchestrator plan** | `orchestrator_plan`                                                                          |
| **Feature analyses**  | `vitals_analysis`, `lab_analysis`, `microbiology_analysis`, `pharmacy_analysis`, `history_analysis` |
| **Diagnostician**     | `sofa_score`, `organ_dysfunction`, `sepsis3_met`, `sepsis3_reasoning`                        |
| **Compliance**        | `sirs_criteria`, `lactate_timing`, `sep1_met`, `sep1_reasoning`, `missing_data_queries`      |
| **Reflection**        | `reflection_count`, `reflection_notes`                                                       |
| **Final output**      | `final_academic_diagnosis`, `final_sep1_compliance`, `final_summary`                         |
| **UI trace**          | `agent_trace` — `list[dict]` of `{"agent": str, "content": str}`                            |

---

## Clinical Background

### Sepsis-3 (Academic Definition)

Published in JAMA 2016 by Singer et al., Sepsis-3 defines sepsis as *"life-threatening organ dysfunction caused by a dysregulated host response to infection."* It replaced older SIRS-based definitions with the SOFA score (Sequential Organ Failure Assessment), requiring a delta of >= 2 points from baseline to qualify as organ dysfunction.

### CMS SEP-1 (Regulatory Measure)

The Centers for Medicare & Medicaid Services (CMS) SEP-1 measure uses the older SIRS-based screening plus specific timed bundle interventions (lactate measurement, blood cultures, antibiotics, fluid resuscitation). Hospitals are scored on compliance with these time-sensitive actions. A patient can meet Sepsis-3 academically but fail SEP-1 if documentation of bundle compliance is incomplete.

This discrepancy between academic and regulatory definitions is precisely why both assessments are valuable — and why the reflection loop exists.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Modular UI package (`app/`)** | The Streamlit frontend is split into 7 focused modules behind a 57-line entry point. Each page is a standalone `render()` function — easy to find, edit, and test independently. |
| **Orchestrator-controlled architecture** | A central Orchestrator LLM plans the analysis, coordinates sub-agents, and synthesises the final report — providing coherent, user-facing communication. |
| **One agent per clinical domain** | Each data type (vitals, labs, microbiology, medications, history) gets a dedicated agent with a specialised prompt, enabling deep domain expertise and independent prompt tuning. |
| **DuckDB on `.csv.gz`** | MIMIC-IV files are multi-GB; DuckDB queries compressed files directly without loading into RAM, preventing memory overflow. |
| **Modular agent files** | Each agent has its own file with an editable `SYSTEM_PROMPT` — change clinical logic without touching the graph or UI. |
| **Agent Controller UI** | Live-edit all 8 agent prompts from the browser; no code changes needed to tweak agent behavior. |
| **LangGraph state machine** | Enables the cyclic reflection loop with conditional edges, which isn't possible with a simple sequential chain. |
| **Two-phase Orchestrator** | Planning phase flags missing data early; Synthesis phase produces a coherent final report rather than raw agent outputs. |
| **API key persistence** | Keys are saved to a local `.secrets.json` (git-ignored) and auto-loaded into session state on startup. Save/Load buttons in Settings give explicit control. |
| **Gemini model fallback** | On 429 quota errors, the connection test and LLM instantiation automatically try `gemini-2.5-flash` → `gemini-2.5-flash-lite` → `gemini-2.0-flash` → `gemini-2.0-flash-lite` before failing. |
| **Plotly charts** | Interactive, publication-quality visualisations that work natively in Streamlit's wide layout. |
| **Sepsis-ready patient filter** | DuckDB query checks 8 data-completeness requirements across 6 tables in a single scan, cached for 1 hour. |
| **Default to Gemini** | Google Gemini is the default LLM provider; model name is optional (auto-fills sensible defaults per provider). |
| **CSS isolation** | All custom CSS lives in `app/css.py` as a single constant. Material Symbols font is explicitly protected from the Inter font override. |

---

## Example Usage

1. **Open the app** → navigate to **Settings** → select your LLM provider (defaults to Gemini) → enter your API key → click **Save Keys** then **Test Connection**.
2. Go to **Dashboard** → use the filters to narrow patients by ICU status, admission type, or sepsis-readiness → click a patient row to view their complete record across 9 data sections.
3. *(Optional)* Go to **Agent Controller** → customise any of the 8 agent prompts (Orchestrator, Vitals, Lab, Microbiology, Pharmacy, History, Diagnostician, Compliance).
4. Go to **Agent Workspace** → enter `10000032` as `subject_id` → click **Run Sepsis Diagnostic Agent**.
5. Watch the 10-step pipeline execute → review each agent's reasoning in the expandable traces → see the final Sepsis-3 and SEP-1 verdicts with the Orchestrator's synthesis report.
