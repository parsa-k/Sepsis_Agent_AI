"""Agent Controller page — live-edit system prompts for all 8 agents."""

import streamlit as st

from agents.orchestrator_agent import SYSTEM_PROMPT as DEFAULT_ORCHESTRATOR
from agents.vitals_agent import SYSTEM_PROMPT as DEFAULT_VITALS
from agents.lab_agent import SYSTEM_PROMPT as DEFAULT_LAB
from agents.microbiology_agent import SYSTEM_PROMPT as DEFAULT_MICROBIOLOGY
from agents.pharmacy_agent import SYSTEM_PROMPT as DEFAULT_PHARMACY
from agents.history_agent import SYSTEM_PROMPT as DEFAULT_HISTORY
from agents.diagnoses_agent import SYSTEM_PROMPT as DEFAULT_DIAGNOSES


AGENT_DEFS = [
    {
        "key": "prompt_orchestrator",
        "label": "Orchestrator",
        "icon": "🧠",
        "description": (
            "Dynamic planner — reads user intent + data flags and decides "
            "which feature agents to activate (and whether History runs first)."
        ),
        "default": DEFAULT_ORCHESTRATOR,
    },
    {
        "key": "prompt_history",
        "label": "History Agent",
        "icon": "📋",
        "description": (
            "Multi-visit baseline extractor. Activated by the Orchestrator "
            "ONLY when the user selects more than one visit."
        ),
        "default": DEFAULT_HISTORY,
    },
    {
        "key": "prompt_vitals",
        "label": "Vitals Agent",
        "icon": "💓",
        "description": (
            "Strict vital-signs domain expert: HR, BP/MAP, Temp, RR, SpO2, GCS."
        ),
        "default": DEFAULT_VITALS,
    },
    {
        "key": "prompt_lab",
        "label": "Lab Agent",
        "icon": "🔬",
        "description": (
            "Strict labs domain expert: WBC, Lactate, Creatinine, Bilirubin, "
            "Platelets, blood-gas."
        ),
        "default": DEFAULT_LAB,
    },
    {
        "key": "prompt_microbiology",
        "label": "Microbiology Agent",
        "icon": "🦠",
        "description": (
            "Strict infection-evidence domain expert: cultures, organisms, "
            "sensitivities, MDRO flags."
        ),
        "default": DEFAULT_MICROBIOLOGY,
    },
    {
        "key": "prompt_pharmacy",
        "label": "Pharmacy Agent",
        "icon": "💊",
        "description": (
            "Strict pharmacy/fluids domain expert: antibiotics, vasopressors, "
            "crystalloid totals, urine output."
        ),
        "default": DEFAULT_PHARMACY,
    },
    {
        "key": "prompt_diagnoses",
        "label": "Diagnoses Agent (master)",
        "icon": "🩻",
        "description": (
            "Master reasoning agent. Consumes only Part 1 payloads from the "
            "active feature agents and emits the final summary, Patient State "
            "Score (1–5), Sepsis-3 verdict, and SEP-1 compliance verdict."
        ),
        "default": DEFAULT_DIAGNOSES,
    },
]

_PROMPT_KEYS = [
    ("prompt_orchestrator", "orchestrator"),
    ("prompt_history", "history"),
    ("prompt_vitals", "vitals"),
    ("prompt_lab", "lab"),
    ("prompt_microbiology", "microbiology"),
    ("prompt_pharmacy", "pharmacy"),
    ("prompt_diagnoses", "diagnoses"),
]


def get_custom_prompts() -> dict:
    """Collect current prompts from session state for the pipeline."""
    prompts = {}
    for session_key, name in _PROMPT_KEYS:
        if session_key in st.session_state:
            prompts[name] = st.session_state[session_key]
    return prompts


def render():
    st.markdown("# Agent Controller")
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

import json
import os

PROMPTS_FILE = "custom_prompts.json"

def _load_saved_prompts():
    if os.path.exists(PROMPTS_FILE):
        try:
            with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_saved_prompts(key, value):
    prompts = _load_saved_prompts()
    prompts[key] = value
    with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2)

def _remove_saved_prompt(key):
    prompts = _load_saved_prompts()
    if key in prompts:
        del prompts[key]
        with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
            json.dump(prompts, f, indent=2)

def _init_defaults():
    saved = _load_saved_prompts()
    for agent in AGENT_DEFS:
        if agent["key"] not in st.session_state:
            st.session_state[agent["key"]] = saved.get(agent["key"], agent["default"])


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

            btn_cols = st.columns([1, 1, 1, 3])
            with btn_cols[0]:
                if st.button(
                    "Save Prompt",
                    key=f'_save_{agent["key"]}',
                    use_container_width=True,
                    type="primary"
                ):
                    _save_saved_prompts(agent["key"], st.session_state[agent["key"]])
                    st.success("Saved!")
                    
            with btn_cols[1]:
                if st.button(
                    "Reset to Default",
                    key=f'_reset_{agent["key"]}',
                    use_container_width=True,
                ):
                    st.session_state[agent["key"]] = agent["default"]
                    _remove_saved_prompt(agent["key"])
                    st.rerun()
            with btn_cols[2]:
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
            
            import os
            if os.path.exists(PROMPTS_FILE):
                os.remove(PROMPTS_FILE)
            st.rerun()
