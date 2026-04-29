"""Patient History page — five-tab view of a selected pipeline session.

Tabs
────
  Summary       Run metadata, intent, sepsis summary, final diagnosis.
  Raw Data      Full MIMIC-IV data for every selected visit (DuckDB).
  Visualize     Interactive Plotly charts for any plottable data domain.
  Agents Report Per-agent Part 1 (actionable) and Part 2 (reasoning).
  Log           Raw event stream from the Memory Manager.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st

import db


MEMORY_DIR = "app_memory"

_AGENT_ORDER = (
    "orchestrator", "history", "vitals", "lab",
    "microbiology", "pharmacy", "diagnoses", "evaluator",
)

_AGENT_ICONS = {
    "orchestrator": "🧠",
    "history":      "📋",
    "vitals":       "💓",
    "lab":          "🔬",
    "microbiology": "🦠",
    "pharmacy":     "💊",
    "diagnoses":    "🩻",
    "evaluator":    "✅",
}

_FLAG_PALETTE = {
    "green":  ("#059669", "✅", "Task executed successfully"),
    "yellow": ("#D97706", "⚠️", "Task executed with caveats"),
    "red":    ("#DC2626", "🛑", "Task could not be executed reliably"),
}

_VERDICT_COLORS = {
    "ok":   "#059669",
    "warn": "#D97706",
    "fail": "#DC2626",
}

_SCORE_PALETTE = {
    1: ("#059669", "Good"),
    2: ("#65A30D", "Mild"),
    3: ("#D97706", "Moderate"),
    4: ("#EA580C", "Severe"),
    5: ("#DC2626", "Critical"),
}


# ── Entry point ──────────────────────────────────────────────────────────────

def render():
    st.markdown("# Patient History")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if not os.path.isdir(MEMORY_DIR):
        st.info("No patient history found. Run a patient in the Agent Workspace first.")
        return

    patient_dirs = _list_patient_dirs()
    legacy_files = _list_legacy_files()

    if not patient_dirs and not legacy_files:
        st.info("No patient history found. Run a patient in the Agent Workspace first.")
        return

    # ── patient + session selectors ───────────────────────────────────────
    if patient_dirs:
        sel_c1, sel_c2 = st.columns(2)
        with sel_c1:
            patient_id = st.selectbox(
                "Patient (subject ID)",
                options=patient_dirs,
                key="hist_patient_id",
            )

        sessions = _list_sessions(patient_id)
        if not sessions:
            st.warning(
                f"No saved sessions for patient `{patient_id}`. "
                "Run a pipeline from the Agent Workspace first."
            )
        else:
            with sel_c2:
                session_file = st.selectbox(
                    "Session (newest first)",
                    options=sessions,
                    format_func=lambda f: (
                        f.replace("session_", "").replace(".json", "")
                    ),
                    key="hist_session_file",
                )

            session_path = os.path.join(MEMORY_DIR, patient_id, session_file)
            session = _read_json(session_path)
            if session:
                final_state = session.get("final_state") or {}
                subject_id_int = _to_int(final_state.get("subject_id"))
                selected_hadm_ids = [
                    int(h) for h in (final_state.get("selected_hadm_ids") or [])
                    if _to_int(h) is not None
                ]
                patient_info = final_state.get("patient_info") or {}
                diag = final_state.get("diagnoses_output") or {}

                _render_banner(
                    patient_id, selected_hadm_ids, patient_info, diag,
                    session.get("started_at"), session.get("session_id"),
                )

                st.markdown("")  # spacing

                (
                    t_summary, t_raw, t_viz, t_agents, t_log
                ) = st.tabs([
                    "Summary",
                    "Raw Data",
                    "Visualize",
                    "Agents Report",
                    "Log",
                ])

                with t_summary:
                    _tab_summary(session, final_state)

                with t_raw:
                    _tab_raw_data(subject_id_int, selected_hadm_ids, patient_id)

                with t_viz:
                    _tab_visualize(subject_id_int, selected_hadm_ids, patient_id)

                with t_agents:
                    _tab_agents(session)

                with t_log:
                    _tab_log(session)

    # ── legacy single-file runs ───────────────────────────────────────────
    if legacy_files:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        with st.expander("Legacy single-file runs (pre-refactor)", expanded=False):
            for f in legacy_files:
                with st.expander(f.replace(".json", ""), expanded=False):
                    legacy = _read_json(os.path.join(MEMORY_DIR, f))
                    if legacy:
                        st.json(legacy)


# ── Banner ───────────────────────────────────────────────────────────────────

def _render_banner(
    patient_id, selected_hadm_ids, patient_info, diag,
    started_at, session_id,
):
    score = max(1, min(5, _to_int(diag.get("patient_score")) or 3))
    color, slabel = _SCORE_PALETTE.get(score, ("#888", "—"))

    visits_str = ", ".join(str(h) for h in selected_hadm_ids) or "—"
    gender = patient_info.get("gender", "N/A")
    age = patient_info.get("anchor_age", "N/A")

    st.markdown(
        f'<div style="background:linear-gradient(135deg,#1C2E45,#243555);'
        f'color:#C8D8F0;border-radius:14px;'
        f'padding:1rem 1.5rem;margin:0.5rem 0;display:flex;align-items:center;'
        f'gap:1.5rem;border:1px solid rgba(79,142,247,0.2);'
        f'box-shadow:0 4px 20px rgba(13,27,42,0.2);">'
        f'<div style="flex:1;line-height:1.8;">'
        f'<span style="font-size:1.05rem;font-weight:700;color:#EAF0FF;">'
        f'Patient {patient_id}</span>'
        f'&ensp;<span style="opacity:0.65;font-size:0.85rem;">'
        f'Gender: {gender} &nbsp;|&nbsp; Age: {age}</span><br>'
        f'<span style="font-size:0.82rem;opacity:0.6;">'
        f'Visits: {visits_str} &nbsp;|&nbsp; '
        f'Session: {session_id or "—"} &nbsp;|&nbsp; {started_at or ""}'
        f'</span>'
        f'</div>'
        f'<div style="background:{color};color:white;border-radius:10px;'
        f'padding:0.5rem 1.25rem;font-weight:700;text-align:center;'
        f'min-width:110px;white-space:nowrap;box-shadow:0 2px 10px {color}55;">'
        f'Score {score}/5<br>'
        f'<span style="font-size:0.78rem;font-weight:500;">{slabel}</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Summary
# ══════════════════════════════════════════════════════════════════════════════

def _tab_summary(session: dict, final_state: dict):
    diag        = final_state.get("diagnoses_output") or {}
    evaluation  = final_state.get("evaluator_output") or _outcome_from_summary(
        session, "evaluator",
    )
    decision    = final_state.get("orchestrator_decision") or {}
    selected    = final_state.get("selected_hadm_ids") or []
    user_intent = final_state.get("user_intent") or "(none)"

    score = max(1, min(5, _to_int(diag.get("patient_score")) or 3))
    color, slabel = _SCORE_PALETTE.get(score, ("#888", "—"))

    # ── Evaluator flag (top) ─────────────────────────────────────────────
    if evaluation:
        _render_evaluator_flag(evaluation)

    # ── run summary cards ────────────────────────────────────────────────
    st.markdown("#### Run Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f"**Session**\n\n`{session.get('session_id', '—')}`"
    )
    c2.markdown(
        f"**Started / Finished**\n\n"
        f"`{session.get('started_at', '—')}` → `{session.get('finished_at', '—')}`"
    )
    c3.markdown(
        f"**Visits ({len(selected)})**\n\n"
        f"`{', '.join(str(s) for s in selected) or '—'}`"
    )
    c4.markdown(
        f"**Active agents**\n\n"
        f"`{', '.join(decision.get('active_agents', [])) or '—'}`"
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── score + verdict badges ────────────────────────────────────────────
    score_col, verdict_col = st.columns([1, 2])
    with score_col:
        st.markdown(
            f'<div style="background:{color};color:white;border-radius:16px;'
            f'padding:1.5rem;text-align:center;'
            f'box-shadow:0 6px 20px {color}44;">'
            f'<div style="font-size:0.72rem;font-weight:600;letter-spacing:0.08em;'
            f'opacity:0.88;margin-bottom:0.3rem;">PATIENT STATE SCORE</div>'
            f'<div style="font-size:3.2rem;font-weight:800;line-height:1;">'
            f'{score}</div>'
            f'<div style="font-size:0.72rem;opacity:0.7;">out of 5</div>'
            f'<div style="font-size:1rem;font-weight:600;margin-top:0.4rem;">'
            f'{slabel}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with verdict_col:
        s3 = diag.get("sepsis3_met")
        sep1 = diag.get("sep1_compliant")

        s3_color = (
            "#DC2626" if s3 is True
            else "#059669" if s3 is False
            else "#64748B"
        )
        s3_text = (
            "Sepsis-3: YES" if s3 is True
            else "Sepsis-3: NO" if s3 is False
            else "Sepsis-3: Indeterminate"
        )
        sep1_color = (
            "#059669" if sep1 is True
            else "#DC2626" if sep1 is False
            else "#64748B"
        )
        sep1_text = (
            "SEP-1: Compliant" if sep1 is True
            else "SEP-1: Non-compliant" if sep1 is False
            else "SEP-1: Indeterminate"
        )

        st.markdown(
            f'<div style="display:flex;gap:0.75rem;margin-bottom:1rem;">'
            f'<span style="background:{s3_color};color:white;border-radius:8px;'
            f'padding:0.45rem 1rem;font-weight:600;font-size:0.9rem;">'
            f'{s3_text}</span>'
            f'<span style="background:{sep1_color};color:white;border-radius:8px;'
            f'padding:0.45rem 1rem;font-weight:600;font-size:0.9rem;">'
            f'{sep1_text}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if diag.get("final_diagnosis"):
            st.markdown(
                f'<div style="background:#EEF3FC;border-radius:10px;'
                f'padding:0.75rem 1rem;font-weight:600;color:#1A2C50;'
                f'border:1px solid #D0DEFA;">'
                f'Final Diagnosis: {_he(diag["final_diagnosis"])}'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── user intent ───────────────────────────────────────────────────────
    st.markdown("#### User Intent")
    st.code(user_intent, language="text")

    # ── sepsis summary ────────────────────────────────────────────────────
    if diag.get("summary"):
        st.markdown("#### Sepsis Summary")
        st.info(diag["summary"])

    # ── diagnostic details ────────────────────────────────────────────────
    if diag.get("details"):
        st.markdown("#### Diagnostic Details")
        with st.expander("Full diagnostic details from Diagnoses Agent", expanded=True):
            st.markdown(diag["details"])

    # ── treatment recommendations ─────────────────────────────────────────
    next_steps           = (diag.get("next_steps") or "").strip()
    short_term_treatment = (diag.get("short_term_treatment") or "").strip()
    mid_term_plan        = (diag.get("mid_term_plan") or "").strip()

    if next_steps or short_term_treatment or mid_term_plan:
        st.markdown("#### Treatment Recommendations")
        tx_tabs = st.tabs([
            "⚡  Immediate (0–6 h)",
            "🏥  Short-term (6–72 h)",
            "📅  Mid-term (Day 3–30)",
        ])
        with tx_tabs[0]:
            st.markdown(next_steps) if next_steps else st.caption("Not recorded.")
        with tx_tabs[1]:
            st.markdown(short_term_treatment) if short_term_treatment else st.caption("Not recorded.")
        with tx_tabs[2]:
            st.markdown(mid_term_plan) if mid_term_plan else st.caption("Not recorded.")

    # ── orchestrator role / rationale ─────────────────────────────────────
    if decision.get("role") or decision.get("rationale"):
        st.markdown("#### Orchestrator Plan")
        if decision.get("role"):
            st.markdown(
                f'<div style="background:#EEF3FC;border-radius:8px;'
                f'padding:0.6rem 1rem;margin-bottom:0.5rem;'
                f'border:1px solid #D0DEFA;color:#1A2C50;">'
                f'<strong>Role:</strong> {_he(decision["role"])}'
                f'</div>',
                unsafe_allow_html=True,
            )
        hist_first = decision.get("history_first", False)
        st.markdown(
            f"History Agent first: **{'Yes' if hist_first else 'No'}** &ensp;"
            f"(multiple visits selected = history-first routing)"
        )
        if decision.get("rationale"):
            with st.expander("Orchestrator rationale"):
                st.markdown(decision["rationale"])

    # ── evaluator full report ─────────────────────────────────────────────
    if evaluation:
        st.markdown("#### Evaluator Report")
        _render_evaluator_report(evaluation)


# ── Evaluator helpers (shared by Summary tab + Agents Report tab) ───────────

def _render_evaluator_flag(evaluation: dict):
    flag = (evaluation.get("flag") or "yellow").lower()
    color, emoji, headline = _FLAG_PALETTE.get(flag, _FLAG_PALETTE["yellow"])
    overall    = evaluation.get("overall_summary") or ""
    confidence = evaluation.get("confidence", 0)

    st.markdown(
        f"<div style='background:linear-gradient(135deg,{color}E6,{color});"
        f"color:white;padding:1rem 1.25rem;border-radius:14px;"
        f"margin-bottom:1rem;display:flex;align-items:center;gap:1rem;"
        f"box-shadow:0 6px 20px {color}55;'>"
        f"<div style='font-size:2rem;line-height:1;'>{emoji}</div>"
        f"<div style='flex:1;'>"
        f"<div style='font-size:0.72rem;font-weight:600;opacity:0.85;"
        f"letter-spacing:0.08em;'>EVALUATOR VERDICT</div>"
        f"<div style='font-size:1.05rem;font-weight:700;line-height:1.3;'>"
        f"{headline}</div>"
        f"<div style='font-size:0.88rem;opacity:0.92;margin-top:0.25rem;'>"
        f"{_he(overall)}</div>"
        f"</div>"
        f"<div style='background:rgba(255,255,255,0.2);border-radius:10px;"
        f"padding:0.5rem 0.85rem;text-align:center;min-width:78px;'>"
        f"<div style='font-size:0.65rem;opacity:0.85;'>CONFIDENCE</div>"
        f"<div style='font-size:1.5rem;font-weight:800;line-height:1;'>"
        f"{confidence}%</div></div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_evaluator_report(evaluation: dict):
    reports = evaluation.get("agent_reports") or {}
    canonical = (
        "orchestrator", "history", "vitals", "lab",
        "microbiology", "pharmacy", "diagnoses",
    )

    cols = st.columns(min(4, max(1, len(canonical))))
    for i, name in enumerate(canonical):
        entry   = reports.get(name) or {}
        verdict = (entry.get("verdict") or "warn").lower()
        notes   = entry.get("notes") or ""
        color   = _VERDICT_COLORS.get(verdict, "#64748B")
        with cols[i % len(cols)]:
            st.markdown(
                f"<div style='background:#FFFFFF;border:1px solid #D0DEFA;"
                f"border-left:4px solid {color};border-radius:10px;"
                f"padding:0.65rem 0.85rem;margin-bottom:0.5rem;'>"
                f"<div style='font-weight:700;color:#1A2C50;font-size:0.9rem;'>"
                f"{name}</div>"
                f"<div style='font-size:0.72rem;font-weight:700;color:{color};"
                f"letter-spacing:0.06em;text-transform:uppercase;"
                f"margin:0.15rem 0 0.3rem 0;'>{verdict}</div>"
                f"<div style='font-size:0.82rem;color:#3D5A8A;line-height:1.4;'>"
                f"{_he(notes)}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    missing = evaluation.get("missing_data") or []
    if missing:
        with st.expander("Missing or ambiguous data flagged by Evaluator"):
            for item in missing:
                st.markdown(f"- {item}")

    recs = evaluation.get("improvement_recommendations") or ""
    if recs.strip() and not recs.strip().startswith("_"):
        with st.expander("Improvement recommendations", expanded=False):
            st.markdown(recs)


def _outcome_from_summary(session: dict, agent_name: str) -> dict:
    """Pull a non-two-part outcome (e.g. evaluator) from agent_outputs."""
    record = (session.get("agent_outputs") or {}).get(agent_name) or {}
    return record.get("outcome") or {}


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Raw Data
# ══════════════════════════════════════════════════════════════════════════════

def _tab_raw_data(
    subject_id: Optional[int],
    selected_hadm_ids: list,
    patient_id: str,
):
    if subject_id is None:
        st.warning("Subject ID not available in this session.")
        return
    if not selected_hadm_ids:
        st.warning("No visit IDs found in session.")
        return

    st.markdown(
        f"Patient **{subject_id}** — visits `{', '.join(str(h) for h in selected_hadm_ids)}`"
    )

    visit_options = [str(h) for h in selected_hadm_ids]
    if len(visit_options) > 1:
        visit_sel = st.selectbox(
            "Visit to inspect",
            options=["All visits"] + visit_options,
            key=f"raw_visit_{patient_id}",
        )
        hadms = (
            selected_hadm_ids
            if visit_sel == "All visits"
            else [int(visit_sel)]
        )
    else:
        hadms = selected_hadm_ids

    # Explicit load trigger to avoid auto-querying on every tab switch
    load_key = f"raw_loaded_{patient_id}"
    if st.button("Load data from MIMIC-IV", key=f"load_raw_{patient_id}", type="primary"):
        st.session_state[load_key] = True

    if not st.session_state.get(load_key):
        st.caption(
            "Click **Load data from MIMIC-IV** above to query DuckDB. "
            "Results are cached — subsequent tab switches are instant."
        )
        return

    for hadm_id in hadms:
        if len(hadms) > 1:
            st.markdown(f"### Visit {hadm_id}")

        _raw_section("Vitals (chartevents)", _q_vitals, subject_id, hadm_id)
        _raw_section("Labs (labevents)",     _q_labs,   subject_id, hadm_id)
        _raw_section("Microbiology",         _q_micro,  subject_id, hadm_id)
        _raw_section("Prescriptions",        _q_rx,     subject_id, hadm_id)
        _raw_section("ICU Stays",            _q_icu,    subject_id, hadm_id)
        _raw_section("Input Events (IV/drips)", _q_inputs,  subject_id, hadm_id)
        _raw_section("Output Events (urine/drains)", _q_outputs, subject_id, hadm_id)
        _raw_section("Diagnoses (ICD)",      _q_dx,     subject_id, hadm_id)


def _raw_section(label: str, loader_fn, subject_id: int, hadm_id: int):
    df = loader_fn(subject_id, hadm_id)
    count = f" ({len(df):,} rows)" if not df.empty else " — no data"
    with st.expander(f"**{label}**{count}", expanded=False):
        if df.empty:
            st.info(f"No {label.lower()} for visit {hadm_id}.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)


# ── DuckDB loaders (all cached) ──────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def _q_vitals(sid: int, hid: int) -> pd.DataFrame:
    c = db.get_conn(); df = db.get_vitals(c, sid, hid); c.close(); return df

@st.cache_data(ttl=600, show_spinner=False)
def _q_labs(sid: int, hid: int) -> pd.DataFrame:
    c = db.get_conn(); df = db.get_labs(c, sid, hid); c.close(); return df

@st.cache_data(ttl=600, show_spinner=False)
def _q_micro(sid: int, hid: int) -> pd.DataFrame:
    c = db.get_conn(); df = db.get_microbiology(c, sid, hid); c.close(); return df

@st.cache_data(ttl=600, show_spinner=False)
def _q_rx(sid: int, hid: int) -> pd.DataFrame:
    c = db.get_conn(); df = db.get_prescriptions(c, sid, hid); c.close(); return df

@st.cache_data(ttl=600, show_spinner=False)
def _q_icu(sid: int, hid: int) -> pd.DataFrame:
    c = db.get_conn(); df = db.get_icu_stays(c, sid, hid); c.close(); return df

@st.cache_data(ttl=600, show_spinner=False)
def _q_inputs(sid: int, hid: int) -> pd.DataFrame:
    c = db.get_conn(); df = db.get_input_events(c, sid, hid); c.close(); return df

@st.cache_data(ttl=600, show_spinner=False)
def _q_outputs(sid: int, hid: int) -> pd.DataFrame:
    c = db.get_conn(); df = db.get_output_events(c, sid, hid); c.close(); return df

@st.cache_data(ttl=600, show_spinner=False)
def _q_dx(sid: int, hid: int) -> pd.DataFrame:
    c = db.get_conn(); df = db.get_diagnoses(c, sid, hid); c.close(); return df


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Visualize
# ══════════════════════════════════════════════════════════════════════════════

_VIZ_SOURCES = {
    "Vitals (HR, BP, Temp, RR, SpO2, GCS)": {
        "loader": _q_vitals,
        "time_col": "charttime",
        "value_col": "valuenum",
        "label_col": "label",
    },
    "Labs (Lactate, Creatinine, WBC, …)": {
        "loader": _q_labs,
        "time_col": "charttime",
        "value_col": "valuenum",
        "label_col": "label",
    },
    "Output Events (urine, drains)": {
        "loader": _q_outputs,
        "time_col": "charttime",
        "value_col": "value",
        "label_col": "label",
    },
    "Input Events (IV fluids, vasopressors)": {
        "loader": _q_inputs,
        "time_col": "starttime",
        "value_col": "amount",
        "label_col": "label",
    },
}


def _tab_visualize(
    subject_id: Optional[int],
    selected_hadm_ids: list,
    patient_id: str,
):
    if subject_id is None:
        st.warning("Subject ID not available in this session.")
        return
    if not selected_hadm_ids:
        st.warning("No visit IDs found in session.")
        return

    st.markdown("#### Configure Chart")
    cfg_c1, cfg_c2, cfg_c3 = st.columns([2, 1, 1])

    with cfg_c1:
        data_source = st.selectbox(
            "Data source",
            options=list(_VIZ_SOURCES.keys()),
            key=f"viz_src_{patient_id}",
        )

    src_cfg = _VIZ_SOURCES[data_source]
    loader = src_cfg["loader"]
    time_col = src_cfg["time_col"]
    value_col = src_cfg["value_col"]
    label_col = src_cfg["label_col"]

    # Load data for all selected visits, tagged with hadm_id
    frames = []
    with st.spinner("Loading from MIMIC-IV…"):
        for hid in selected_hadm_ids:
            df = loader(subject_id, hid)
            if not df.empty:
                df = df.copy()
                df["visit"] = str(hid)
                frames.append(df)

    if not frames:
        st.info("No data available for the selected source and visits.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=[time_col, value_col])
    combined[time_col] = pd.to_datetime(combined[time_col], errors="coerce")
    combined = combined.dropna(subset=[time_col])
    combined[value_col] = pd.to_numeric(combined[value_col], errors="coerce")
    combined = combined.dropna(subset=[value_col])

    if combined.empty:
        st.warning("No plottable numeric values found.")
        return

    all_labels = sorted(combined[label_col].dropna().unique().tolist())

    with cfg_c2:
        selected_labels = st.multiselect(
            "Series to plot",
            options=all_labels,
            default=all_labels[:3] if all_labels else [],
            key=f"viz_labels_{patient_id}",
        )

    multi_visit = len(selected_hadm_ids) > 1
    with cfg_c3:
        color_by = st.selectbox(
            "Color by",
            options=["label", "visit"] if multi_visit else ["label"],
            key=f"viz_color_{patient_id}",
        )

    if not selected_labels:
        st.info("Select at least one series from the list above.")
        return

    plot_df = combined[combined[label_col].isin(selected_labels)].copy()

    st.markdown("#### Chart")
    if plot_df.empty:
        st.warning("No data for the selected series.")
        return

    fig = px.line(
        plot_df.sort_values(time_col),
        x=time_col,
        y=value_col,
        color=label_col if color_by == "label" else "visit",
        line_dash=label_col if color_by == "visit" and multi_visit else None,
        title=f"{data_source} — Patient {subject_id}",
        markers=True,
        color_discrete_sequence=[
            "#4F8EF7", "#7C3AED", "#10B981", "#D97706",
            "#EF4444", "#0EA5E9", "#EC4899", "#84CC16",
        ],
    )
    fig.update_layout(
        font_family="Inter",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Time",
        yaxis_title="Value",
        legend_title="",
        hovermode="x unified",
        height=500,
        title_font_color="#1A2C50",
        xaxis=dict(gridcolor="rgba(208,222,250,0.4)"),
        yaxis=dict(gridcolor="rgba(208,222,250,0.4)"),
    )
    fig.update_traces(marker_size=6, line_width=2)
    st.plotly_chart(fig, use_container_width=True)

    # Data table toggle
    if st.checkbox("Show underlying data table", key=f"viz_tbl_{patient_id}"):
        display_cols = [time_col, label_col, value_col, "visit"]
        available_cols = [c for c in display_cols if c in plot_df.columns]
        st.dataframe(
            plot_df[available_cols].sort_values(time_col),
            use_container_width=True,
            hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Agents Report
# ══════════════════════════════════════════════════════════════════════════════

def _tab_agents(session: dict):
    agent_outputs = session.get("agent_outputs") or {}

    if not agent_outputs:
        st.info("No agent outputs captured in this session.")
        return

    st.markdown(
        "Each agent card shows **Part 1** (the actionable payload passed "
        "downstream) and **Part 2** (the detailed reasoning, hidden in the "
        "live Workspace but fully visible here for audit)."
    )

    seen: set = set()
    for name in _AGENT_ORDER:
        if name in agent_outputs:
            _render_agent_card(name, agent_outputs[name])
            seen.add(name)

    for name, payload in agent_outputs.items():
        if name not in seen:
            _render_agent_card(name, payload)


def _render_agent_card(agent_name: str, payload: dict):
    icon = _AGENT_ICONS.get(agent_name, "🤖")
    skipped = payload.get("skipped", False)

    # Detect orchestrator / diagnoses style (has "outcome" key)
    is_outcome_style = "outcome" in payload and "part1_payload" not in payload

    header = f"{icon} **{agent_name}**"
    if skipped:
        header += " — *skipped*"
    elif is_outcome_style:
        header += " — *decision / verdict*"

    with st.expander(header, expanded=not skipped):
        if skipped:
            reason = (
                (payload.get("part1_payload") or {})
                .get("actionable", {})
                .get("reason", "Agent not activated")
            )
            st.warning(f"Agent skipped — {reason}")
            return

        if is_outcome_style:
            outcome = payload.get("outcome") or {}

            # Special-case: evaluator gets a rich rendering
            if agent_name == "evaluator" and outcome:
                _render_evaluator_flag(outcome)
                _render_evaluator_report(outcome)
            else:
                st.markdown("**Outcome**")
                st.json(outcome)

            if payload.get("raw"):
                with st.expander("Raw LLM response"):
                    st.code(payload["raw"], language="text")
            return

        # Standard two-part output
        part1 = payload.get("part1_payload") or {}
        part2 = payload.get("part2_reasoning") or ""
        parse_err = payload.get("parse_error")

        tab_p1, tab_p2, tab_raw = st.tabs([
            "Part 1 — Actionable (downstream)",
            "Part 2 — Reasoning (audit only)",
            "Raw LLM response",
        ])

        with tab_p1:
            actionable = (
                part1.get("actionable") if isinstance(part1, dict) else part1
            )
            sources = (
                part1.get("source_records", []) if isinstance(part1, dict) else []
            )
            if actionable:
                st.json(actionable)
            else:
                st.caption("No actionable payload recorded.")
            if sources:
                st.markdown("**Source records**")
                st.code("\n".join(str(s) for s in sources), language="text")
            if parse_err:
                st.warning(f"Parser warning: {parse_err}")

        with tab_p2:
            if part2:
                st.markdown(part2)
            else:
                st.caption("No Part 2 reasoning recorded.")

        with tab_raw:
            raw = payload.get("raw") or ""
            if raw:
                st.code(raw, language="markdown")
            else:
                st.caption("Raw LLM response not available.")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 — Log
# ══════════════════════════════════════════════════════════════════════════════

_KIND_COLORS = {
    "input":   "#4F8EF7",
    "output":  "#059669",
    "outcome": "#7C3AED",
    "event":   "#D97706",
    "skipped": "#64748B",
    "error":   "#DC2626",
}


def _tab_log(session: dict):
    events = session.get("events") or []

    if not events:
        st.info("No events recorded for this session.")
        return

    st.markdown(
        f"**{len(events)}** events recorded across "
        f"**{len({e.get('agent') for e in events})}** agents."
    )

    # Filter controls
    f_c1, f_c2 = st.columns(2)
    all_agents = sorted({e.get("agent", "") for e in events if e.get("agent")})
    all_kinds  = sorted({e.get("kind", "") for e in events if e.get("kind")})

    with f_c1:
        filter_agents = st.multiselect(
            "Filter by agent",
            options=all_agents,
            default=[],
            key="log_filter_agents",
            placeholder="All agents",
        )
    with f_c2:
        filter_kinds = st.multiselect(
            "Filter by kind",
            options=all_kinds,
            default=[],
            key="log_filter_kinds",
            placeholder="All kinds",
        )

    filtered = [
        ev for ev in events
        if (not filter_agents or ev.get("agent") in filter_agents)
        and (not filter_kinds or ev.get("kind") in filter_kinds)
    ]

    st.caption(f"Showing {len(filtered)} of {len(events)} events.")
    st.markdown("")

    for idx, ev in enumerate(filtered):
        kind  = ev.get("kind", "?")
        agent = ev.get("agent", "?")
        ts    = ev.get("ts", "")
        badge_color = _KIND_COLORS.get(kind, "#888")

        header_html = (
            f'<span style="background:{badge_color};color:white;border-radius:6px;'
            f'padding:2px 8px;font-size:0.75rem;font-weight:600;">{kind}</span>'
            f'&ensp;<code>{agent}</code>'
            f'&ensp;<span style="color:#888;font-size:0.8rem;">{ts}</span>'
        )
        with st.expander(
            label=f"{kind.upper():8s}  {agent}  {ts}",
            expanded=False,
        ):
            st.markdown(header_html, unsafe_allow_html=True)
            content = ev.get("content")
            if isinstance(content, dict):
                st.json(content)
            elif content is not None:
                st.code(str(content), language="text")
            else:
                st.caption("(empty)")


# ── Filesystem helpers ────────────────────────────────────────────────────────

def _list_patient_dirs() -> list[str]:
    out = []
    for entry in sorted(os.listdir(MEMORY_DIR)):
        if os.path.isdir(os.path.join(MEMORY_DIR, entry)):
            out.append(entry)
    return out


def _list_legacy_files() -> list[str]:
    out = []
    for entry in sorted(os.listdir(MEMORY_DIR)):
        if (
            os.path.isfile(os.path.join(MEMORY_DIR, entry))
            and entry.endswith(".json")
        ):
            out.append(entry)
    return out


def _list_sessions(patient_id: str) -> list[str]:
    folder = os.path.join(MEMORY_DIR, patient_id)
    if not os.path.isdir(folder):
        return []
    sessions = [
        f for f in os.listdir(folder)
        if f.startswith("session_") and f.endswith(".json")
    ]
    sessions.sort(
        key=lambda f: os.path.getmtime(os.path.join(folder, f)),
        reverse=True,
    )
    return sessions


def _read_json(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        st.error(f"Could not read session file `{path}`.")
        return None


# ── Tiny utilities ────────────────────────────────────────────────────────────

def _to_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _he(text: str) -> str:
    """HTML-escape a string for safe markdown injection."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
