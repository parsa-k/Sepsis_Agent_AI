"""Agent Workspace page — search patients, run the pipeline, view results."""

import streamlit as st

import db
from agents.graph import run_pipeline
from app.llm import get_llm
from app.controller import get_custom_prompts


def render():
    st.markdown("# 🤖 Sepsis Diagnostic Agent Workspace")
    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True,
    )

    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        search_input = st.text_input(
            "Enter Subject ID or Admission ID (hadm_id)",
            placeholder="e.g. 10000032 or 29079034",
        )
    with search_col2:
        search_type = st.selectbox("Search by", ["subject_id", "hadm_id"])

    if not search_input:
        return

    try:
        search_val = int(search_input.strip())
    except ValueError:
        st.error("Please enter a valid numeric ID.")
        return

    subject_id, hadm_id, row = _resolve_patient(search_val, search_type)
    if subject_id is None:
        return

    _render_patient_banner(subject_id, hadm_id, row)

    if st.button(
        "🚀 Run Sepsis Diagnostic Agent",
        use_container_width=True,
        type="primary",
    ):
        _run_pipeline(subject_id, hadm_id)

    if "last_result" in st.session_state:
        _render_results(st.session_state["last_result"])


# ── Private helpers ──────────────────────────────────────────────────────────

def _resolve_patient(search_val: int, search_type: str):
    conn = db.get_conn()
    if search_type == "hadm_id":
        admission_df = db.find_patient(conn, hadm_id=search_val)
    else:
        admission_df = db.find_patient(conn, subject_id=search_val)
    conn.close()

    if admission_df.empty:
        st.warning("No admission found for this ID.")
        return None, None, None

    row = admission_df.iloc[0]
    return int(row["subject_id"]), int(row["hadm_id"]), row


def _render_patient_banner(subject_id, hadm_id, row):
    st.markdown(
        f'<div style="background:#b0bbcf;border-radius:12px;'
        f'padding:1rem 1.5rem;margin:1rem 0;">'
        f'<strong>Patient:</strong> {subject_id} &nbsp;|&nbsp;'
        f'<strong>Admission:</strong> {hadm_id} &nbsp;|&nbsp;'
        f'<strong>Gender:</strong> {row.get("gender", "N/A")} &nbsp;|&nbsp;'
        f'<strong>Age:</strong> {row.get("anchor_age", "N/A")} &nbsp;|&nbsp;'
        f'<strong>Admit:</strong> {row.get("admittime", "N/A")} &nbsp;|&nbsp;'
        f'<strong>Type:</strong> {row.get("admission_type", "N/A")}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _run_pipeline(subject_id: int, hadm_id: int):
    llm = get_llm()
    if llm is None:
        st.error(
            "Please configure your LLM API key in the Settings tab first."
        )
        st.stop()

    custom_prompts = get_custom_prompts()

    with st.status(
        "Running Sepsis Diagnostic Pipeline...", expanded=True,
    ) as status:
        st.write("Querying DuckDB for patient data...")
        result = run_pipeline(
            llm, subject_id, hadm_id,
            custom_prompts=custom_prompts or None,
        )
        status.update(label="Pipeline complete!", state="complete")

    st.session_state["last_result"] = result


def _render_results(result: dict):
    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Final Verdict")
    v1, v2 = st.columns(2)

    with v1:
        sepsis3 = result.get("final_academic_diagnosis")
        if sepsis3 is True:
            badge, text = "badge-yes", "YES — Sepsis Detected"
        elif sepsis3 is False:
            badge, text = "badge-no", "NO — No Sepsis"
        else:
            badge, text = "badge-unknown", "INDETERMINATE"
        st.markdown("**Academic Sepsis-3 Diagnosis**")
        st.markdown(
            f'<span class="{badge}">{text}</span>',
            unsafe_allow_html=True,
        )

    with v2:
        sep1 = result.get("final_sep1_compliance")
        if sep1 is True:
            badge, text = "badge-no", "YES — Compliant"
        elif sep1 is False:
            badge, text = "badge-yes", "NO — Not Compliant"
        else:
            badge, text = "badge-unknown", "INDETERMINATE"
        st.markdown("**CMS SEP-1 Compliance**")
        st.markdown(
            f'<span class="{badge}">{text}</span>',
            unsafe_allow_html=True,
        )

    if result.get("reflection_count", 0) > 0:
        st.info(
            f"Reflection loop ran {result['reflection_count']} time(s)."
        )

    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Agent Thought Process")

    trace = result.get("agent_trace", [])
    for i, entry in enumerate(trace):
        agent_name = entry.get("agent", f"Step {i + 1}")
        content = entry.get("content", "")
        with st.expander(
            f"**{agent_name}**",
            expanded=(i == len(trace) - 1),
        ):
            st.markdown(content)

    if result.get("final_summary"):
        st.markdown(
            '<div class="section-divider"></div>',
            unsafe_allow_html=True,
        )
        st.markdown("### Summary")
        st.markdown(result["final_summary"])
