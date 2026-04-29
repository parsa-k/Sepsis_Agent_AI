"""Patient History page — browse `app_memory/` and inspect each agent's
Part 1 (actionable) and Part 2 (detailed reasoning)."""

from __future__ import annotations

import json
import os
import streamlit as st

import db


MEMORY_DIR = "app_memory"


def render():
    st.markdown("# Patient History")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if not os.path.isdir(MEMORY_DIR):
        st.info(
            "No patient history found. Run a patient in the Agent Workspace first."
        )
        return

    patient_dirs = _list_patient_dirs()
    legacy_files = _list_legacy_files()

    if not patient_dirs and not legacy_files:
        st.info(
            "No patient history found. Run a patient in the Agent Workspace first."
        )
        return

    if patient_dirs:
        st.markdown("### 1. Pick a patient")
        patient_id = st.selectbox(
            "Patients on file",
            options=patient_dirs,
            key="hist_patient_id",
        )

        st.markdown("### 2. Pick a session")
        sessions = _list_sessions(patient_id)
        if not sessions:
            st.warning(
                f"No saved sessions for patient `{patient_id}` yet. "
                f"Run a pipeline from the Agent Workspace."
            )
        else:
            session_file = st.selectbox(
                "Sessions",
                options=sessions,
                format_func=lambda f: f.replace("session_", "").replace(".json", ""),
                key="hist_session_file",
            )
            session_path = os.path.join(MEMORY_DIR, patient_id, session_file)
            session = _read_json(session_path)
            if session:
                _render_session(patient_id, session)

    if legacy_files:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        with st.expander("Legacy single-file runs (pre-refactor)"):
            for f in legacy_files:
                with st.expander(f.replace(".json", "")):
                    legacy = _read_json(os.path.join(MEMORY_DIR, f))
                    if legacy:
                        st.json(legacy)


# ── File-system helpers ─────────────────────────────────────────────────────

def _list_patient_dirs() -> list[str]:
    out = []
    for entry in sorted(os.listdir(MEMORY_DIR)):
        full = os.path.join(MEMORY_DIR, entry)
        if os.path.isdir(full):
            out.append(entry)
    return out


def _list_legacy_files() -> list[str]:
    out = []
    for entry in sorted(os.listdir(MEMORY_DIR)):
        full = os.path.join(MEMORY_DIR, entry)
        if os.path.isfile(full) and entry.endswith(".json"):
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
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        st.error(f"Could not read `{path}`.")
        return None


# ── Session renderer ────────────────────────────────────────────────────────

def _render_session(patient_id: str, session: dict):
    final_state = session.get("final_state") or {}
    diag = final_state.get("diagnoses_output") or {}
    decision = final_state.get("orchestrator_decision") or {}
    selected = final_state.get("selected_hadm_ids") or []
    user_intent = final_state.get("user_intent") or "(none)"

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    _render_patient_banner(patient_id, selected, final_state.get("patient_info") or {})

    st.markdown("### Run summary")
    cols = st.columns(4)
    cols[0].markdown(f"**Session**\n\n`{session.get('session_id', '—')}`")
    cols[1].markdown(f"**Visits**\n\n`{', '.join(str(s) for s in selected) or '—'}`")
    cols[2].markdown(
        f"**Patient State Score**\n\n`{diag.get('patient_score', '—')} / 5`"
    )
    cols[3].markdown(
        f"**Active agents**\n\n`{', '.join(decision.get('active_agents', [])) or '—'}`"
    )

    st.markdown("**User intent**")
    st.code(user_intent, language="text")

    if diag.get("summary"):
        st.markdown("**Sepsis summary**")
        st.info(diag["summary"])
    if diag.get("final_diagnosis"):
        st.markdown(f"**Final diagnosis:** {diag['final_diagnosis']}")
    if diag.get("details"):
        with st.expander("Diagnoses Agent — full details"):
            st.markdown(diag["details"])

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Per-agent outputs")
    st.caption(
        "Open each agent to see its **Part 1 (actionable)** payload AND its "
        "**Part 2 (detailed reasoning)** — the latter is hidden in the live "
        "Workspace view but kept here for audit."
    )
    _render_agent_outputs(session.get("agent_outputs") or {}, final_state)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Live event log")
    with st.expander("Raw event stream (input/output/event entries)"):
        events = session.get("events") or []
        st.write(f"{len(events)} entries")
        for ev in events:
            st.markdown(
                f"- `{ev.get('ts')}` **{ev.get('agent')}** — _{ev.get('kind')}_"
            )
            with st.expander("Show payload"):
                st.json(ev.get("content") or {})

    if selected:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### Quick vitals chart (primary visit)")
        try:
            conn = db.get_conn()
            vitals_df = db.get_vitals(conn, int(patient_id), int(selected[0]))
            conn.close()
            if not vitals_df.empty:
                from app.dashboard import render_vitals_chart
                render_vitals_chart(vitals_df)
            else:
                st.caption("No charted vitals to plot.")
        except (ValueError, Exception) as exc:  # noqa: BLE001
            st.caption(f"(Vitals chart unavailable: {exc})")


def _render_patient_banner(patient_id, selected, patient_info: dict):
    st.markdown(
        f'<div style="background:#b0bbcf;border-radius:12px;'
        f'padding:1rem 1.5rem;margin:0.5rem 0;">'
        f'<strong>Patient:</strong> {patient_id} &nbsp;|&nbsp;'
        f'<strong>Visits:</strong> {", ".join(str(s) for s in selected) or "—"} &nbsp;|&nbsp;'
        f'<strong>Gender:</strong> {patient_info.get("gender", "N/A")} &nbsp;|&nbsp;'
        f'<strong>Age:</strong> {patient_info.get("anchor_age", "N/A")}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_agent_outputs(agent_outputs: dict, final_state: dict):
    """Render every agent we have a record for, in canonical order."""
    canonical_order = (
        "orchestrator", "history", "vitals", "lab",
        "microbiology", "pharmacy", "diagnoses",
    )
    seen = set()
    for name in canonical_order:
        if name in agent_outputs:
            _render_one_agent(name, agent_outputs[name])
            seen.add(name)

    for name, payload in agent_outputs.items():
        if name not in seen:
            _render_one_agent(name, payload)

    if not agent_outputs:
        st.info("No per-agent outputs were captured for this session.")


def _render_one_agent(agent_name: str, payload: dict):
    skipped = payload.get("skipped", False)
    title = f"**{agent_name}**"
    if skipped:
        title += " — _skipped_"

    with st.expander(title, expanded=False):
        if skipped:
            st.warning(
                payload.get("part2_reasoning")
                or "This agent was not activated."
            )
            return

        part1 = payload.get("part1_payload") or {}
        part2 = payload.get("part2_reasoning") or ""
        parse_err = payload.get("parse_error")

        st.markdown("#### Part 1 — Actionable payload (propagated downstream)")
        actionable = part1.get("actionable") if isinstance(part1, dict) else part1
        sources = part1.get("source_records") if isinstance(part1, dict) else []
        st.json(actionable or {})
        if sources:
            st.markdown("**Source records**")
            st.code("\n".join(str(s) for s in sources), language="text")

        st.markdown("#### Part 2 — Detailed reasoning (hidden in Workspace)")
        if part2:
            st.markdown(part2)
        else:
            st.caption("(No Part 2 reasoning recorded.)")

        if parse_err:
            st.warning(f"Parser warning: {parse_err}")

        with st.expander("Raw LLM response"):
            st.code(payload.get("raw") or "(none)", language="markdown")
