# Sepsis Diagnostic Agent - Website & UI Documentation

This document describes the **current** Streamlit frontend after the dynamic multi-visit refactor.

## Overview

The frontend is built with **Streamlit** and provides:

- patient discovery and chart exploration,
- prompt-level agent control,
- dynamic orchestration input from the user,
- clear separation between short actionable outputs (Part 1) and deep reasoning (Part 2),
- sessionized historical audit from `app_memory/`.

## UI Architecture

The UI is modular and page-oriented:

- **`app.py`**: The main entry point. It is extremely slim (~57 lines). It handles bootstrapping the application, setting up the page configuration (wide layout, custom title/icon), injecting custom CSS, rendering the sidebar navigation, and dispatching the user to the selected page module.
- **`app/` Package**: Contains all the logic for individual pages and shared UI utilities.
  - **`css.py`**: Houses the `CUSTOM_CSS` constant. It contains all stylesheet rules, bringing a custom Inter font, styling metric cards, badges, and ensuring Streamlit's default components look polished and modern.
  - **`secrets.py`**: Manages API key persistence (loading/saving `.secrets.json`) and session state initialization.
  - **`llm.py`**: Handles LLM instantiation, session caching, and fallback logic for API rate limits (e.g., trying multiple Gemini models on a 429 error).

## Technologies Used

- **Streamlit**: Core framework for building the reactive web app.
- **Plotly Express**: Used for building interactive, publication-quality visualizations natively within Streamlit's wide layout.
- **Vanilla CSS**: Injected via markdown to enforce high-quality visual aesthetics.

## Application Pages

### 1. Settings & Configuration (`app/settings.py`)
Handles LLM API configuration.
- **LLM Provider Selection**: Dropdown to select between Google Gemini (default), OpenAI, and Anthropic Claude.
- **Model Selection**: Allows overriding the default models (e.g., `gemini-2.5-flash`, `gpt-4o`) with specific ones.
- **API Key Management**: Secure text input for API keys.
- **Persistence**: "Save Keys" and "Load Keys" buttons to persist credentials locally into `.secrets.json`.
- **Connection Testing**: A button to dispatch a simple test prompt. It features built-in fallback logic (specifically for Gemini) to automatically downshift models if rate limits are encountered.

### 2. Dataset Dashboard (`app/dashboard.py`)
A comprehensive tool for exploring the MIMIC-IV dataset and finding suitable patient records for analysis.
- **Metric Cards**: High-level statistics (Total Patients, Total Admissions, Average Age, Mortality Rate) styled with custom CSS gradients.
- **Visualizations**: 
  - Gender Distribution (Plotly donut chart).
  - Top Admission Types (Plotly horizontal bar chart).
- **Patient Browser**: An interactive, paginated table to browse patients with powerful filters:
  - Patient Type: All, ICU-only, Non-ICU.
  - Admission Type: Emergency, Urgent, Elective, etc.
  - **Sepsis Data Filter**: A specialized filter that queries the database to find only "Sepsis-ready" patients—those with complete sets of data (blood cultures, antibiotics, PaO2, Platelets, Bilirubin, Creatinine, MAP, and GCS) required for a full Sepsis-3 assessment, filtered to adults aged 18-90.
- **Patient Detail Drill-Down**: Clicking a patient row expands a detailed view of their clinical record, organized into 9 expandable sections (Demographics, Admissions, ICU Stays, Diagnoses, Labs, Vitals, Prescriptions, Microbiology, Input Events). It includes an interactive **Vital Signs Visualization** chart (via Plotly) to dynamically plot selected vitals over time, alongside a "Completeness Summary" panel highlighting missing data.

### 3. Agent Controller (`app/controller.py`)
The control center for customizing prompt behavior of the **current** agents.
- Provides prompt editors for: Orchestrator, History, Vitals, Lab, Microbiology, Pharmacy, and Diagnoses (master).
- **Live Updates**: Edits are saved to the session state instantly.
- **Visual Indicators**: "MODIFIED" badges appear next to prompts that have been altered from their defaults.
- **Reset functionality**: Options to reset individual agents or all agents back to their default clinical instructions.

### 4. Agent Workspace (`app/workspace.py`)
The execution environment where the clinical pipeline is run on a specific patient.
- **Search**: Allows searching for a patient via `subject_id` or `hadm_id`.
- **Patient Banner**: Displays a quick summary of the selected patient.
- **Multi-Visit Selector**: If multiple admissions exist, users can select one or more visits to include in the run.
- **Orchestrator Instruction Input**: A text area where users directly define intent for the Orchestrator. Default:
  - `Review the selected clinical records and evaluate for Sepsis-3 and SEP-1 compliance. Highlight any critical anomalies.`
- **Execution**: Runs dynamic LangGraph flow with selected visits + user intent.
- **Results Display**:
  - **Concise Sepsis Summary**.
  - **Patient State Score** (`1` Good -> `5` Critical).
  - **Final Diagnosis** from the Diagnoses Agent.
  - **Streamlined Agent Trace** using **Part 1 payloads only**.
  - Part 2 reasoning is intentionally hidden in this page.

### 5. Patient History (`app/history.py`)
A dedicated view for reviewing previously executed pipeline runs.
- **Sessionized History**: Reads from `app_memory/<patient_id>/session_*.json`.
- **Patient + Session Selection**: Select patient folder, then specific run.
- **Visualizations**: Includes the same interactive **Vital Signs Visualization** found in the Dashboard, pulling the raw DataFrame from the database on-the-fly.
- **Per-Agent Audit View**:
  - Part 1 payload (actionable + source records),
  - Part 2 reasoning (full detailed reasoning),
  - raw LLM output and event stream.
- This is the canonical page for long-form reasoning inspection.

## Notes on Current UI/Backend Contract

- Workspace shows Part 1 only for context efficiency.
- History reveals Part 2 for deep inspection.
- Multi-visit selection controls whether History Agent is invoked:
  - single visit -> no History Agent,
  - multiple visits -> History Agent runs first and baseline is propagated downstream.
