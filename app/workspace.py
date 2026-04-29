"""Agent Workspace page — patient search, visit picker, run pipeline, view results."""

from __future__ import annotations

import json
import streamlit as st

import db
from agents.graph import run_pipeline
from app.llm import get_llm_with_fallback
from app.controller import get_custom_prompts


DEFAULT_INTENT = (
    "Review the selected clinical records and evaluate for Sepsis-3 "
    "and SEP-1 compliance. Highlight any critical anomalies."
)


# ── Public entry-point ──────────────────────────────────────────────────────

def render():
    st.markdown("# Sepsis Diagnostic Agent Workspace")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 1. Patient lookup")
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        search_input = st.text_input(
            "Enter Subject ID or Admission ID (hadm_id)",
            placeholder="e.g. 10000032 or 29079034",
            key="ws_search_input",
        )
    with search_col2:
        search_type = st.selectbox(
            "Search by", ["subject_id", "hadm_id"], key="ws_search_type",
        )

    if not search_input:
        return

    try:
        search_val = int(search_input.strip())
    except ValueError:
        st.error("Please enter a valid numeric ID.")
        return

    subject_id, primary_hadm_id, primary_row, all_admissions = (
        _resolve_patient(search_val, search_type)
    )
    if subject_id is None:
        return

    _render_patient_banner(subject_id, primary_hadm_id, primary_row)

    st.markdown("### 2. Select hospital visit(s)")
    selected_hadms = _render_visit_picker(all_admissions, primary_hadm_id)

    st.markdown("### 3. Tell the Orchestrator what to do")
    user_intent = st.text_area(
        "Instruction for the Orchestrator",
        value=st.session_state.get("ws_user_intent", DEFAULT_INTENT),
        height=110,
        key="ws_user_intent",
        help=(
            "This text is fed directly to the Orchestrator. It will pick "
            "which feature agents to activate and tailor their per-agent "
            "instructions to your request."
        ),
    )

    st.markdown("### 4. Run")
    run_disabled = len(selected_hadms) == 0
    if st.button(
        f"Run Sepsis Diagnostic Agent on {len(selected_hadms)} visit(s)",
        use_container_width=True,
        type="primary",
        disabled=run_disabled,
    ):
        _run_pipeline(subject_id, selected_hadms, user_intent)

    if "last_result" in st.session_state:
        _render_results(st.session_state["last_result"])


# ── Patient lookup ──────────────────────────────────────────────────────────

def _resolve_patient(search_val: int, search_type: str):
    conn = db.get_conn()
    if search_type == "hadm_id":
        primary_df = db.find_patient(conn, hadm_id=search_val)
    else:
        primary_df = db.find_patient(conn, subject_id=search_val)

    if primary_df.empty:
        conn.close()
        st.warning("No admission found for this ID.")
        return None, None, None, None

    primary_row = primary_df.iloc[0]
    subject_id = int(primary_row["subject_id"])
    primary_hadm_id = int(primary_row["hadm_id"])

    all_adm = conn.execute(f"""
        SELECT a.hadm_id, a.admittime, a.dischtime, a.admission_type,
               a.admission_location, a.discharge_location,
               a.hospital_expire_flag
        FROM read_csv_auto('{db.PATHS['admissions']}') a
        WHERE a.subject_id = {subject_id}
        ORDER BY a.admittime
    """).fetchdf()
    conn.close()

    return subject_id, primary_hadm_id, primary_row, all_adm


def _render_patient_banner(subject_id, hadm_id, row):
    st.markdown(
        f'<div style="background:#b0bbcf;border-radius:12px;'
        f'padding:1rem 1.5rem;margin:1rem 0;">'
        f'<strong>Patient:</strong> {subject_id} &nbsp;|&nbsp;'
        f'<strong>Primary admission:</strong> {hadm_id} &nbsp;|&nbsp;'
        f'<strong>Gender:</strong> {row.get("gender", "N/A")} &nbsp;|&nbsp;'
        f'<strong>Age:</strong> {row.get("anchor_age", "N/A")} &nbsp;|&nbsp;'
        f'<strong>Admit:</strong> {row.get("admittime", "N/A")} &nbsp;|&nbsp;'
        f'<strong>Type:</strong> {row.get("admission_type", "N/A")}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_visit_picker(all_admissions, primary_hadm_id) -> list:
    """Show every admission for the patient and let the user pick one or more."""
    if all_admissions is None or len(all_admissions) == 0:
        st.warning("No admissions on record.")
        return []

    admissions_list = all_admissions.to_dict(orient="records")

    if len(admissions_list) == 1:
        only = admissions_list[0]
        st.info(
            f"Only one admission on record — visit "
            f"**{int(only['hadm_id'])}** "
            f"({only.get('admittime')}, {only.get('admission_type')}). "
            f"The History Agent will be **skipped** for single-visit runs."
        )
        return [int(only["hadm_id"])]

    st.caption(
        f"This patient has **{len(admissions_list)} admissions**. Pick one or "
        f"more — selecting two or more will activate the **History Agent** to "
        f"build a cross-visit baseline before the feature agents run."
    )

    options = [int(r["hadm_id"]) for r in admissions_list]

    def _label(hadm_id: int) -> str:
        rec = next(r for r in admissions_list if int(r["hadm_id"]) == hadm_id)
        return (
            f"{hadm_id} — {rec.get('admittime')} "
            f"({rec.get('admission_type')}, "
            f"discharge: {rec.get('discharge_location')})"
        )

    default = [primary_hadm_id] if primary_hadm_id in options else options[:1]
    return st.multiselect(
        "Hospital admissions",
        options=options,
        default=default,
        format_func=_label,
        key="ws_selected_hadms",
    )


# ── Pipeline runner ─────────────────────────────────────────────────────────

def _run_pipeline(subject_id: int, selected_hadm_ids: list, user_intent: str):
    llm, _model_used, _ = get_llm_with_fallback()
    if llm is None:
        st.error("Please configure your LLM API key in the Settings tab first.")
        st.stop()

    custom_prompts = get_custom_prompts()

    with st.status("Running Sepsis Diagnostic Pipeline...", expanded=True) as status:
        st.write(
            f"Subject {subject_id} • visits {selected_hadm_ids} • "
            f"{'multi-visit (history-first)' if len(selected_hadm_ids) > 1 else 'single visit'}"
        )
        result = run_pipeline(
            llm=llm,
            subject_id=subject_id,
            selected_hadm_ids=selected_hadm_ids,
            user_intent=user_intent,
            custom_prompts=custom_prompts or None,
        )
        status.update(label="Pipeline complete!", state="complete")

    st.session_state["last_result"] = result


# ── Results rendering ───────────────────────────────────────────────────────

def _render_results(result: dict):
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    diag = result.get("diagnoses_output") or {}
    summary = diag.get("summary") or "Concise summary unavailable."
    score = diag.get("patient_score") or 3
    final_diagnosis = diag.get("final_diagnosis") or "Indeterminate."

    st.markdown("### Final Result")

    st.markdown(
        f"<div style='background:#1e2a3a;color:#fff;padding:1rem 1.25rem;"
        f"border-radius:12px;'>"
        f"<div style='font-size:0.85rem;opacity:0.8;'>Sepsis summary</div>"
        f"<div style='font-size:1.05rem;'>{_html_escape(summary)}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    score_col, diag_col = st.columns([1, 2])
    with score_col:
        _render_patient_score(score)
    with diag_col:
        st.markdown("**Final diagnosis**")
        st.markdown(
            f"<div style='background:#b0bbcf;padding:0.85rem 1rem;"
            f"border-radius:10px;font-weight:600;'>"
            f"{_html_escape(final_diagnosis)}</div>",
            unsafe_allow_html=True,
        )

    if diag.get("details"):
        with st.expander("Diagnostic details"):
            st.markdown(diag["details"])

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Streamlined agent trace (Part 1 only)")
    _render_orchestrator_decision(result.get("orchestrator_decision") or {})
    _render_part1_trace(result, diag.get("agent_trace_part1") or [])


def _render_orchestrator_decision(decision: dict):
    if not decision:
        return
    cols = st.columns(3)
    cols[0].markdown(
        f"**Role**\n\n{_html_escape(decision.get('role', '—'))}",
    )
    cols[1].markdown(
        f"**Active agents**\n\n`{', '.join(decision.get('active_agents', [])) or '—'}`"
    )
    cols[2].markdown(
        f"**History first?**\n\n{'Yes' if decision.get('history_first') else 'No'}"
    )
    if decision.get("rationale"):
        with st.expander("Orchestrator rationale"):
            st.markdown(decision["rationale"])


def _render_patient_score(score: int):
    palette = {
        1: ("#1e8e3e", "Good"),
        2: ("#7cb342", "Mild"),
        3: ("#f2994a", "Moderate"),
        4: ("#e67e22", "Severe"),
        5: ("#c0392b", "Critical"),
    }
    color, label = palette.get(int(score or 3), ("#888", "—"))
    st.markdown(
        f"<div style='background:{color};color:white;border-radius:12px;"
        f"padding:1rem;text-align:center;'>"
        f"<div style='font-size:0.8rem;opacity:0.85;'>Patient State Score</div>"
        f"<div style='font-size:2.5rem;font-weight:700;'>{int(score or 3)}/5</div>"
        f"<div style='font-size:0.95rem;'>{label}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_part1_trace(result: dict, fallback_trace: list):
    """Display Part 1 only — Part 2 reasoning is intentionally hidden here."""
    if fallback_trace:
        ordered = fallback_trace
    else:
        ordered = []
        for name, key in (
            ("history", "history_output"),
            ("vitals", "vitals_output"),
            ("lab", "lab_output"),
            ("microbiology", "microbiology_output"),
            ("pharmacy", "pharmacy_output"),
        ):
            env = result.get(key)
            if env and not env.get("skipped"):
                ordered.append({
                    "agent": name,
                    "part1_payload": env.get("part1_payload", {}),
                })

    if not ordered:
        st.info("No active agents produced a payload for this run.")
        return

    for entry in ordered:
        agent = entry["agent"]
        payload = entry.get("part1_payload") or {}
        with st.expander(f"**{agent}** — Part 1 (Actionable)", expanded=True):
            actionable = payload.get("actionable", payload)
            sources = payload.get("source_records", [])
            st.markdown("**Actionable**")
            st.json(actionable)
            if sources:
                st.markdown("**Source records**")
                st.code("\n".join(str(s) for s in sources), language="text")

    st.caption(
        "Part 2 (detailed reasoning) is intentionally hidden in this view to "
        "save context. Open it from the **Patient History** tab."
    )


# ── Tiny utilities ──────────────────────────────────────────────────────────

def _html_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
