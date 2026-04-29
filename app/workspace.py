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
        f'<div style="background:linear-gradient(135deg,#1C2E45,#243555);'
        f'color:#C8D8F0;border-radius:12px;padding:1rem 1.5rem;margin:1rem 0;'
        f'border:1px solid rgba(79,142,247,0.2);'
        f'box-shadow:0 4px 16px rgba(13,27,42,0.18);">'
        f'<strong style="color:#fff;">Patient:</strong> {subject_id} &nbsp;|&nbsp;'
        f'<strong style="color:#fff;">Primary admission:</strong> {hadm_id} &nbsp;|&nbsp;'
        f'<strong style="color:#fff;">Gender:</strong> {row.get("gender", "N/A")} &nbsp;|&nbsp;'
        f'<strong style="color:#fff;">Age:</strong> {row.get("anchor_age", "N/A")} &nbsp;|&nbsp;'
        f'<strong style="color:#fff;">Admit:</strong> {row.get("admittime", "N/A")} &nbsp;|&nbsp;'
        f'<strong style="color:#fff;">Type:</strong> {row.get("admission_type", "N/A")}'
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

    diag       = result.get("diagnoses_output") or {}
    evaluation = result.get("evaluator_output") or {}
    summary    = diag.get("summary") or "Concise summary unavailable."
    score      = diag.get("patient_score") or 3
    final_diagnosis = diag.get("final_diagnosis") or "Indeterminate."

    # ── Evaluator flag (green / yellow / red) — shown FIRST ──────────────
    if evaluation:
        _render_evaluator_flag(evaluation)

    st.markdown("### Final Result")

    st.markdown(
        f"<div style='background:linear-gradient(135deg,#1C2E45,#243555);"
        f"color:#C8D8F0;padding:1rem 1.25rem;border-radius:12px;"
        f"border:1px solid rgba(79,142,247,0.2);"
        f"box-shadow:0 4px 16px rgba(13,27,42,0.15);'>"
        f"<div style='font-size:0.78rem;font-weight:600;letter-spacing:0.07em;"
        f"color:#4F8EF7;margin-bottom:0.35rem;'>SEPSIS SUMMARY</div>"
        f"<div style='font-size:1.02rem;color:#EAF0FF;line-height:1.55;'>"
        f"{_html_escape(summary)}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    score_col, diag_col = st.columns([1, 2])
    with score_col:
        _render_patient_score(score)
    with diag_col:
        st.markdown("**Final diagnosis**")
        st.markdown(
            f"<div style='background:#EEF3FC;padding:0.85rem 1rem;"
            f"border-radius:10px;font-weight:600;color:#1A2C50;"
            f"border:1px solid #D0DEFA;'>"
            f"{_html_escape(final_diagnosis)}</div>",
            unsafe_allow_html=True,
        )

    if diag.get("details"):
        with st.expander("Diagnostic details (SOFA / SEP-1 breakdown)"):
            st.markdown(diag["details"])

    # ── Treatment recommendations ─────────────────────────────────────────
    next_steps           = (diag.get("next_steps") or "").strip()
    short_term_treatment = (diag.get("short_term_treatment") or "").strip()
    mid_term_plan        = (diag.get("mid_term_plan") or "").strip()

    if next_steps or short_term_treatment or mid_term_plan:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### Treatment Recommendations")

        tx_tabs = st.tabs([
            "⚡  Immediate (0–6 h)",
            "🏥  Short-term (6–72 h)",
            "📅  Mid-term (Day 3–30)",
        ])

        with tx_tabs[0]:
            if next_steps:
                st.markdown(next_steps)
            else:
                st.caption("No immediate actions recorded.")

        with tx_tabs[1]:
            if short_term_treatment:
                st.markdown(short_term_treatment)
            else:
                st.caption("No short-term plan recorded.")

        with tx_tabs[2]:
            if mid_term_plan:
                st.markdown(mid_term_plan)
            else:
                st.caption("No mid-term plan recorded.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Streamlined agent trace (Part 1 only)")
    _render_orchestrator_decision(result.get("orchestrator_decision") or {})
    _render_part1_trace(result, diag.get("agent_trace_part1") or [])

    # ── Evaluator full report ────────────────────────────────────────────
    if evaluation:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        _render_evaluator_report(evaluation)


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
        1: ("#059669", "Good"),
        2: ("#65A30D", "Mild"),
        3: ("#D97706", "Moderate"),
        4: ("#EA580C", "Severe"),
        5: ("#DC2626", "Critical"),
    }
    color, label = palette.get(int(score or 3), ("#64748B", "—"))
    st.markdown(
        f"<div style='background:{color};color:white;border-radius:14px;"
        f"padding:1.25rem 1rem;text-align:center;"
        f"box-shadow:0 4px 16px {color}44;'>"
        f"<div style='font-size:0.72rem;font-weight:600;letter-spacing:0.08em;"
        f"opacity:0.88;margin-bottom:0.3rem;'>PATIENT STATE SCORE</div>"
        f"<div style='font-size:2.8rem;font-weight:800;line-height:1;'>"
        f"{int(score or 3)}/5</div>"
        f"<div style='font-size:0.9rem;font-weight:600;margin-top:0.3rem;'>"
        f"{label}</div>"
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


# ── Evaluator rendering ─────────────────────────────────────────────────────

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


def _render_evaluator_flag(evaluation: dict):
    flag = (evaluation.get("flag") or "yellow").lower()
    color, emoji, headline = _FLAG_PALETTE.get(flag, _FLAG_PALETTE["yellow"])
    overall = evaluation.get("overall_summary") or ""
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
        f"{_html_escape(overall)}</div>"
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
    st.markdown("### Evaluator Report")
    reports = evaluation.get("agent_reports") or {}
    canonical = (
        "orchestrator", "history", "vitals", "lab",
        "microbiology", "pharmacy", "diagnoses",
    )

    cols = st.columns(min(4, max(1, len(canonical))))
    for i, name in enumerate(canonical):
        entry = reports.get(name) or {}
        verdict = (entry.get("verdict") or "warn").lower()
        notes = entry.get("notes") or ""
        color = _VERDICT_COLORS.get(verdict, "#64748B")
        with cols[i % len(cols)]:
            st.markdown(
                f"<div style='background:#FFFFFF;border:1px solid #D0DEFA;"
                f"border-left:4px solid {color};border-radius:10px;"
                f"padding:0.65rem 0.85rem;margin-bottom:0.5rem;'>"
                f"<div style='font-weight:700;color:#1A2C50;font-size:0.9rem;'>"
                f"{name}</div>"
                f"<div style='font-size:0.72rem;font-weight:700;"
                f"color:{color};letter-spacing:0.06em;text-transform:uppercase;"
                f"margin:0.15rem 0 0.3rem 0;'>{verdict}</div>"
                f"<div style='font-size:0.82rem;color:#3D5A8A;line-height:1.4;'>"
                f"{_html_escape(notes)}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    missing = evaluation.get("missing_data") or []
    if missing:
        with st.expander("Missing or ambiguous data flagged by Evaluator"):
            for item in missing:
                st.markdown(f"- {item}")

    recs = evaluation.get("improvement_recommendations") or ""
    if recs and recs.strip() and recs.strip().startswith("_") is False:
        with st.expander("Improvement recommendations", expanded=False):
            st.markdown(recs)


# ── Tiny utilities ──────────────────────────────────────────────────────────

def _html_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
