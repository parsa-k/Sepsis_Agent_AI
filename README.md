# Sepsis Diagnostic Agent

A LangGraph + Streamlit multi-agent system for sepsis analysis on
MIMIC-IV. The system was refactored to a **dynamic two-phase
orchestrator** plus an **Evaluator quality gate**, with strict two-part
agent outputs and a persistent run-log Memory Manager.

Key features:

- **Two-phase Orchestrator** — Phase 1 gates the History Agent based on
  visit count and crafts a tailored history instruction; Phase 2 runs
  after history (or directly for single-visit) and finalises which
  feature agents to activate, with concrete baseline-aware instructions.
- **User-provided intent** in the workspace, fed verbatim to the
  Orchestrator.
- **Conditional History routing** — single-visit runs skip History;
  multi-visit runs activate History first to produce a Part-1 baseline
  that the Phase-2 re-plan and the feature agents consume.
- **Two-part agent outputs** — every feature agent emits
  `part1_payload` (actionable, propagated downstream) +
  `part2_reasoning` (full reasoning, kept for audit only).
- **Master Diagnoses Agent** — single agent that produces summary,
  Patient State Score (1–5), final diagnosis, Sepsis-3 + SEP-1
  verdicts, and immediate / short-term / mid-term treatment plans.
- **Evaluator Agent** — final quality gate. Compares user intent +
  raw-data digest + agent Part-1 payloads + the Diagnoses verdict and
  emits a green / yellow / red flag plus an audit report.
- **Memory Manager** for I/O standardisation, parsing, and persistent
  run logs in `app_memory/<patient_id>/`.

---

## What Changed (Latest Refactor — Two-Phase Orchestrator + Evaluator)

### 1) Two-phase Orchestrator

Phase 1 — `agents.orchestrator_agent.run_orchestrator_preplan`:

- Reads `user_intent`, `selected_hadm_ids`, `available_data_flags`.
- Decides whether History must run first (multi-visit ⇒ yes,
  single-visit ⇒ no, enforced).
- Crafts an intent-aware History-Agent instruction when multi-visit.
- Does NOT pick feature agents at this phase.

Phase 2 — `agents.orchestrator_agent.run_orchestrator_replan`:

- Runs after History (multi-visit) or directly after Phase 1
  (single-visit).
- Reads the History `part1_payload` baseline (if any) plus the user
  intent and data flags.
- Picks the final `active_agents` set and writes per-agent
  `agent_instructions` that are concrete and baseline-aware.
- Cannot include `history` in the feature pick set; history (when run)
  is preserved at the head of `active_agents` for traceability only.

Both phases share `state['orchestrator_decision']` and persist their
raw responses for audit (`raw_response_preplan`, `raw_response`).

### 2) Evaluator Agent (final quality gate)

`agents/evaluator_agent.py` runs at the end of every pipeline. It
reads:

- the user's verbatim intent,
- a raw-data digest per visit,
- the Orchestrator's final decision,
- every agent's Part-1 payload,
- the Diagnoses Agent's full output.

It produces a strict JSON envelope:

```json
{
  "flag":              "green | yellow | red",
  "task_executed":     true,
  "confidence":        88,
  "overall_summary":   "...",
  "agent_reports": {
    "orchestrator": {"verdict": "ok|warn|fail", "notes": "..."},
    "history":      {"verdict": "ok",   "notes": "Skipped — single-visit run."},
    "vitals":       {"verdict": "ok",   "notes": "..."},
    "lab":          {"verdict": "warn", "notes": "..."},
    "microbiology": {"verdict": "ok",   "notes": "..."},
    "pharmacy":     {"verdict": "ok",   "notes": "..."},
    "diagnoses":    {"verdict": "ok",   "notes": "..."}
  },
  "missing_data":      ["..."],
  "improvement_recommendations": "Markdown bullet list"
}
```

The flag is rendered as a coloured banner on both the Workspace and
Patient History pages.

### 3) Two-part output contract for feature agents

```json
{
  "part1_payload": {
    "actionable": {},
    "source_records": []
  },
  "part2_reasoning": "..."
}
```

Only `part1_payload` is propagated downstream. `part2_reasoning` is
hidden from the Diagnoses Agent and the Evaluator on purpose (saves
context, prevents prose drift), but kept in the audit log.

### 4) Diagnoses Agent — full treatment plan

`agents/diagnoses_agent.py` consumes only Part-1 payloads and produces:

- concise sepsis summary,
- Patient State Score (1=Good → 5=Critical),
- final diagnosis line,
- Sepsis-3 + SEP-1 verdicts,
- `next_steps` — immediate actions (0–6 h),
- `short_term_treatment` — acute plan (6–72 h),
- `mid_term_plan` — post-acute plan (day 3–30).

### 5) Memory Manager

Runtime artifacts written to `app_memory/<patient_id>/`:

- `session_<ts>.jsonl` — append-only event stream
  (input, output, outcome, event, skipped, error),
- `session_<ts>.json` — consolidated session summary including every
  agent's outcome and the trimmed final state.

---

## Current Pipeline

```text
data_loader
  └─► orchestrator_preplan   (Phase 1: gate History + craft its instruction)
        ├─► (multi-visit)  ─► history ─► propagate_baseline ─► orchestrator_replan
        └─► (single-visit) ─────────────────────────────────► orchestrator_replan
                                                  │
                                                  ▼
                              vitals ─► lab ─► microbiology ─► pharmacy
                                                  │
                                                  ▼
                                    diagnoses ─► evaluator ─► END
```

There is no reflection loop. Each feature node is a no-op when it is
not in `active_agents`.

---

## Current Project Structure

```text
agents/
  state.py
  graph.py
  memory_manager_agent.py
  orchestrator_agent.py        # two-phase planner
  history_agent.py
  vitals_agent.py
  lab_agent.py
  microbiology_agent.py
  pharmacy_agent.py
  diagnoses_agent.py
  evaluator_agent.py           # NEW — final quality gate
  _agent_utils.py

app/
  workspace.py     # evaluator flag at top, two-part trace, treatment tabs
  history.py       # tabbed: Summary, Raw Data, Visualize, Agents Report, Log
  controller.py    # tab-per-agent prompt editor (incl. Phase-2 + Evaluator)
  dashboard.py
  settings.py
  llm.py
  css.py

tests/
  test_orchestrator.py    # Phase 1 + Phase 2 + enforcement rules
  test_feature_agents.py  # five domain agents + diagnoses
  test_evaluator.py       # NEW — evaluator parsing, normalisation, e2e
  test_graph.py           # full pipeline incl. evaluator + replan
  test_memory_manager.py
  test_imports.py
```

Removed earlier (kept here as a marker):

- `agents/diagnostician_agent.py`
- `agents/compliance_agent.py`

---

## Workspace UX

`app/workspace.py`:

- patient lookup (`subject_id` or `hadm_id`),
- multi-visit selector when multiple admissions exist,
- orchestrator instruction text area (default:
  "*Review the selected clinical records and evaluate for Sepsis-3 and
  SEP-1 compliance. Highlight any critical anomalies.*"),
- results view:
  - **Evaluator flag banner** (green / yellow / red) at the top with a
    confidence badge and overall summary,
  - sepsis summary,
  - Patient State Score (1–5),
  - final diagnosis,
  - **Treatment Recommendations** tabs: Immediate / Short-term /
    Mid-term,
  - streamlined **Part 1 only** agent trace,
  - full Evaluator agent-by-agent verdict cards.

---

## Patient History UX

`app/history.py` reads `app_memory/<patient_id>/session_*.json` and
shows five tabs after selecting a patient + session:

1. **Summary** — Evaluator flag banner, run metadata, score / verdict
   badges, sepsis summary, treatment plans, orchestrator plan, full
   Evaluator report.
2. **Raw Data** — full per-visit MIMIC-IV data fetched on demand from
   DuckDB (vitals, labs, micro, prescriptions, ICU stays, input /
   output events, ICD diagnoses).
3. **Visualize** — interactive Plotly charts over any plottable
   domain (vitals / labs / inputs / outputs), grouped by series and
   visit.
4. **Agents Report** — every agent in canonical order, including the
   Evaluator with its full report; for two-part agents an inner three-
   tab view (`Part 1 / Part 2 / Raw LLM response`).
5. **Log** — the full append-only event stream with kind / agent
   filters.

---

## Setup & Run

```bash
conda activate AI
streamlit run app.py
```

---

## Test Status

Latest run in conda env `AI`:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

- **53 tests passing**, including:
  - two-phase orchestrator (Phase 1 + Phase 2 + enforcement),
  - feature agents two-part contract,
  - history baseline propagation,
  - diagnoses payload-only consumption,
  - evaluator parsing / normalisation / end-to-end,
  - graph compilation,
  - graph integration: single visit + multi-visit + replan trace +
    evaluator output,
  - memory manager parsing / logging / persistence,
  - module import sanity + obsolete-module removal.
