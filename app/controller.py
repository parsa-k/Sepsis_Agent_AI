"""Agent Controller page — live-edit system prompts for all 8 agents."""

import streamlit as st

from agents.orchestrator_agent import SYSTEM_PROMPT as DEFAULT_ORCHESTRATOR
from agents.vitals_agent import SYSTEM_PROMPT as DEFAULT_VITALS
from agents.lab_agent import SYSTEM_PROMPT as DEFAULT_LAB
from agents.microbiology_agent import SYSTEM_PROMPT as DEFAULT_MICROBIOLOGY
from agents.pharmacy_agent import SYSTEM_PROMPT as DEFAULT_PHARMACY
from agents.history_agent import SYSTEM_PROMPT as DEFAULT_HISTORY
from agents.diagnostician_agent import SYSTEM_PROMPT as DEFAULT_DIAGNOSTICIAN
from agents.compliance_agent import SYSTEM_PROMPT as DEFAULT_COMPLIANCE


AGENT_DEFS = [
    {
        "key": "prompt_orchestrator",
        "label": "Orchestrator",
        "icon": "🧠",
        "description": (
            "Central brain — plans analysis, coordinates sub-agents, "
            "and synthesises the final report."
        ),
        "default": DEFAULT_ORCHESTRATOR,
    },
    {
        "key": "prompt_vitals",
        "label": "Vitals Agent",
        "icon": "💓",
        "description": (
            "Analyses charted vital signs: HR, BP/MAP, Temp, RR, "
            "SpO2, GCS from chartevents."
        ),
        "default": DEFAULT_VITALS,
    },
    {
        "key": "prompt_lab",
        "label": "Lab Agent",
        "icon": "🔬",
        "description": (
            "Analyses lab results: WBC, Lactate, Creatinine, "
            "Bilirubin, Platelets, PaO2, Blood Gas."
        ),
        "default": DEFAULT_LAB,
    },
    {
        "key": "prompt_microbiology",
        "label": "Microbiology Agent",
        "icon": "🦠",
        "description": (
            "Analyses cultures, organisms, sensitivities — "
            "determines infection evidence."
        ),
        "default": DEFAULT_MICROBIOLOGY,
    },
    {
        "key": "prompt_pharmacy",
        "label": "Pharmacy Agent",
        "icon": "💊",
        "description": (
            "Analyses antibiotics, vasopressors, IV fluids, "
            "and urine output."
        ),
        "default": DEFAULT_PHARMACY,
    },
    {
        "key": "prompt_history",
        "label": "History Agent",
        "icon": "📋",
        "description": (
            "Reviews prior admissions and ICD diagnoses to identify "
            "chronic conditions and baseline organ function."
        ),
        "default": DEFAULT_HISTORY,
    },
    {
        "key": "prompt_diagnostician",
        "label": "Diagnostician Agent",
        "icon": "🩻",
        "description": (
            "Applies Sepsis-3 criteria — calculates SOFA score and "
            "determines if organ dysfunction is acute."
        ),
        "default": DEFAULT_DIAGNOSTICIAN,
    },
    {
        "key": "prompt_compliance",
        "label": "Compliance Agent",
        "icon": "📑",
        "description": (
            "Checks CMS SEP-1 bundle compliance — SIRS screening, "
            "3-hour and 6-hour bundle timing."
        ),
        "default": DEFAULT_COMPLIANCE,
    },
]

_PROMPT_KEYS = [
    ("prompt_orchestrator", "orchestrator"),
    ("prompt_vitals", "vitals"),
    ("prompt_lab", "lab"),
    ("prompt_microbiology", "microbiology"),
    ("prompt_pharmacy", "pharmacy"),
    ("prompt_history", "history"),
    ("prompt_diagnostician", "diagnostician"),
    ("prompt_compliance", "compliance"),
]


def get_custom_prompts() -> dict:
    """Collect current prompts from session state for the pipeline."""
    prompts = {}
    for session_key, name in _PROMPT_KEYS:
        if session_key in st.session_state:
            prompts[name] = st.session_state[session_key]
    return prompts


def render():
    st.markdown("# 🎛️ Agent Controller")
    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Edit the system prompts for each agent below. Changes are "
        "saved to your session instantly — the **Agent Workspace** will "
        "use whatever prompts you see here on the next run. Click "
        "**Reset to Default** to restore the original prompt for any agent."
    )

    _init_defaults()
    _render_prompt_editors()
    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True,
    )
    _render_status_strip()
    _render_reset_all_button()


# ── Private helpers ──────────────────────────────────────────────────────────

def _init_defaults():
    for agent in AGENT_DEFS:
        if agent["key"] not in st.session_state:
            st.session_state[agent["key"]] = agent["default"]


def _render_prompt_editors():
    for agent in AGENT_DEFS:
        with st.expander(
            f'{agent["icon"]}  **{agent["label"]}** — '
            f'{agent["description"]}',
            expanded=False,
        ):
            current_val = st.session_state[agent["key"]]
            is_modified = (
                current_val.strip() != agent["default"].strip()
            )

            if is_modified:
                st.markdown(
                    '<span style="background:#f2994a;color:white;'
                    'padding:2px 10px;border-radius:10px;'
                    'font-size:0.75rem;font-weight:600;">MODIFIED</span>',
                    unsafe_allow_html=True,
                )

            new_val = st.text_area(
                f'System prompt for {agent["label"]}',
                value=current_val,
                height=350,
                key=f'_ta_{agent["key"]}',
                label_visibility="collapsed",
            )
            st.session_state[agent["key"]] = new_val

            btn_cols = st.columns([1, 1, 4])
            with btn_cols[0]:
                if st.button(
                    "Reset to Default",
                    key=f'_reset_{agent["key"]}',
                    use_container_width=True,
                ):
                    st.session_state[agent["key"]] = agent["default"]
                    st.rerun()
            with btn_cols[1]:
                st.markdown(
                    f"<small style='color:#888;padding-top:0.5rem;"
                    f"display:block;'>{len(new_val):,} chars</small>",
                    unsafe_allow_html=True,
                )


def _render_status_strip():
    for row_start in range(0, len(AGENT_DEFS), 4):
        row_agents = AGENT_DEFS[row_start: row_start + 4]
        cols = st.columns(len(row_agents))
        for i, agent in enumerate(row_agents):
            with cols[i]:
                cur = st.session_state[agent["key"]]
                is_mod = cur.strip() != agent["default"].strip()
                color = "#f2994a" if is_mod else "#11998e"
                label = "Modified" if is_mod else "Default"
                st.markdown(
                    f'<div style="background:#657a8e;border-radius:12px;'
                    f'padding:1rem;text-align:center;'
                    f'border-top:3px solid {color};">'
                    f'<div style="font-size:1.5rem;">'
                    f'{agent["icon"]}</div>'
                    f'<div style="font-weight:600;font-size:0.85rem;'
                    f'margin:0.25rem 0;">{agent["label"]}</div>'
                    f'<div style="color:{color};font-weight:600;'
                    f'font-size:0.8rem;">{label}</div></div>',
                    unsafe_allow_html=True,
                )


def _render_reset_all_button():
    any_modified = any(
        st.session_state.get(a["key"], a["default"]).strip()
        != a["default"].strip()
        for a in AGENT_DEFS
    )
    if any_modified:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(
            "Reset ALL Prompts to Defaults",
            type="secondary",
            use_container_width=True,
        ):
            for agent in AGENT_DEFS:
                st.session_state[agent["key"]] = agent["default"]
            st.rerun()
