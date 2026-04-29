"""Agent Controller page — tab-per-agent prompt editor.

One tab per agent in the top toolbar. Each tab shows the agent's
description, its current system prompt (editable), and Save / Reset
buttons that persist choices across sessions via custom_prompts.json.
"""

from __future__ import annotations

import json
import os

import streamlit as st

from agents.orchestrator_agent import (
    SYSTEM_PROMPT        as DEFAULT_ORCHESTRATOR,
    REPLAN_SYSTEM_PROMPT as DEFAULT_ORCHESTRATOR_REPLAN,
)
from agents.vitals_agent       import SYSTEM_PROMPT as DEFAULT_VITALS
from agents.lab_agent          import SYSTEM_PROMPT as DEFAULT_LAB
from agents.microbiology_agent import SYSTEM_PROMPT as DEFAULT_MICROBIOLOGY
from agents.pharmacy_agent     import SYSTEM_PROMPT as DEFAULT_PHARMACY
from agents.history_agent      import SYSTEM_PROMPT as DEFAULT_HISTORY
from agents.diagnoses_agent    import SYSTEM_PROMPT as DEFAULT_DIAGNOSES
from agents.evaluator_agent    import SYSTEM_PROMPT as DEFAULT_EVALUATOR
from agents._agent_utils       import TWO_PART_OUTPUT_INSTRUCTIONS as DEFAULT_TWO_PART_FORMAT


PROMPTS_FILE = "custom_prompts.json"

AGENT_DEFS = [
    {
        "key":         "prompt_orchestrator",
        "label":       "Orchestrator (Phase 1)",
        "icon":        "🧠",
        "description": (
            "Phase-1 pre-plan. Decides whether the History Agent must run "
            "first and crafts an intent-aware instruction for it. No "
            "feature agents are picked at this phase."
        ),
        "pipeline_name": "orchestrator",
        "default":     DEFAULT_ORCHESTRATOR,
    },
    {
        "key":         "prompt_orchestrator_replan",
        "label":       "Orchestrator (Phase 2)",
        "icon":        "🧭",
        "description": (
            "Phase-2 re-plan. Runs after the History Agent (or directly "
            "for single-visit runs). Picks the active feature agents and "
            "writes their per-agent, baseline-aware instructions."
        ),
        "pipeline_name": "orchestrator_replan",
        "default":     DEFAULT_ORCHESTRATOR_REPLAN,
    },
    {
        "key":         "prompt_history",
        "label":       "History Agent",
        "icon":        "📋",
        "description": (
            "Multi-visit baseline extractor. Activated by the Orchestrator "
            "only when more than one visit is selected."
        ),
        "pipeline_name": "history",
        "default":     DEFAULT_HISTORY,
    },
    {
        "key":         "prompt_vitals",
        "label":       "Vitals Agent",
        "icon":        "💓",
        "description": (
            "Strict vital-signs domain expert: HR, BP/MAP, Temperature, "
            "RR, SpO2, GCS. Outputs two-part JSON."
        ),
        "pipeline_name": "vitals",
        "default":     DEFAULT_VITALS,
    },
    {
        "key":         "prompt_lab",
        "label":       "Lab Agent",
        "icon":        "🔬",
        "description": (
            "Strict labs domain expert: WBC, Lactate, Creatinine, "
            "Bilirubin, Platelets, blood-gas. Outputs two-part JSON."
        ),
        "pipeline_name": "lab",
        "default":     DEFAULT_LAB,
    },
    {
        "key":         "prompt_microbiology",
        "label":       "Microbiology Agent",
        "icon":        "🦠",
        "description": (
            "Strict infection-evidence expert: cultures, organisms, "
            "sensitivities, MDRO flags. Outputs two-part JSON."
        ),
        "pipeline_name": "microbiology",
        "default":     DEFAULT_MICROBIOLOGY,
    },
    {
        "key":         "prompt_pharmacy",
        "label":       "Pharmacy Agent",
        "icon":        "💊",
        "description": (
            "Strict pharmacy/fluids expert: antibiotics, vasopressors, "
            "crystalloid totals, urine output. Outputs two-part JSON."
        ),
        "pipeline_name": "pharmacy",
        "default":     DEFAULT_PHARMACY,
    },
    {
        "key":         "prompt_diagnoses",
        "label":       "Diagnoses Agent",
        "icon":        "🩻",
        "description": (
            "Master reasoning agent. Consumes Part-1 payloads from all "
            "active feature agents and emits the final summary, Patient "
            "State Score (1–5), Sepsis-3 verdict, SEP-1 compliance, and "
            "the immediate / short-term / mid-term treatment plans."
        ),
        "pipeline_name": "diagnoses",
        "default":     DEFAULT_DIAGNOSES,
    },
    {
        "key":         "prompt_evaluator",
        "label":       "Evaluator Agent",
        "icon":        "✅",
        "description": (
            "Final quality gate. Reads the user intent, raw-data digest, "
            "Orchestrator plan, every Part-1 payload, and the Diagnoses "
            "verdict, then emits a green / yellow / red flag plus an "
            "audit report on overall and per-agent performance."
        ),
        "pipeline_name": "evaluator",
        "default":     DEFAULT_EVALUATOR,
    },
    {
        "key":         "prompt_two_part_format",
        "label":       "Output Format (shared)",
        "icon":        "📐",
        "description": (
            "Shared two-part output format instructions appended to every "
            "feature agent's system prompt at runtime (Vitals, Lab, "
            "Microbiology, Pharmacy, History). Defines the strict "
            "part1_payload / part2_reasoning JSON contract. Changes here "
            "affect all five feature agents simultaneously."
        ),
        "pipeline_name": "two_part_format",
        "default":     DEFAULT_TWO_PART_FORMAT,
    },
]

_PROMPT_KEYS = [(a["key"], a["pipeline_name"]) for a in AGENT_DEFS]


# ── Public API (consumed by workspace.py) ────────────────────────────────────

def get_custom_prompts() -> dict:
    """Return {pipeline_name: prompt_text} for every modified agent."""
    prompts = {}
    for session_key, pipeline_name in _PROMPT_KEYS:
        if session_key in st.session_state:
            prompts[pipeline_name] = st.session_state[session_key]
    return prompts


# ── Page entry point ─────────────────────────────────────────────────────────

def render():
    st.markdown("# Agent Controller")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    _init_defaults()

    # ── status strip ─────────────────────────────────────────────────────
    _render_status_strip()
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── one tab per agent ─────────────────────────────────────────────────
    tab_labels = [f"{a['icon']}  {a['label']}" for a in AGENT_DEFS]
    tabs = st.tabs(tab_labels)

    for tab, agent in zip(tabs, AGENT_DEFS):
        with tab:
            _render_agent_tab(agent)

    # ── global reset ──────────────────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    _render_reset_all()


# ── Per-agent tab ─────────────────────────────────────────────────────────────

def _render_agent_tab(agent: dict):
    key      = agent["key"]
    default  = agent["default"]
    current  = st.session_state.get(key, default)
    modified = current.strip() != default.strip()

    # Description + badge row
    badge_html = ""
    if modified:
        badge_html = (
            ' &ensp;<span style="background:#D97706;color:white;'
            'padding:2px 10px;border-radius:10px;font-size:0.75rem;'
            'font-weight:600;letter-spacing:0.04em;">MODIFIED</span>'
        )

    st.markdown(
        f'<div style="margin-bottom:0.65rem;">'
        f'<span style="font-size:1.05rem;font-weight:700;color:#1A2C50;">'
        f'{agent["icon"]} {agent["label"]}</span>'
        f'{badge_html}'
        f'</div>'
        f'<div style="background:#EEF3FC;border-radius:10px;'
        f'padding:0.65rem 1rem;margin-bottom:1rem;color:#3D5A8A;'
        f'font-size:0.88rem;border:1px solid #D0DEFA;">'
        f'{agent["description"]}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Prompt text area — use a widget key so Streamlit tracks changes
    new_val = st.text_area(
        "System prompt",
        value=current,
        height=460,
        key=f"_ta_{key}",
        label_visibility="collapsed",
        placeholder="Enter the system prompt for this agent…",
    )
    # Keep session state in sync with the text area
    st.session_state[key] = new_val

    # Action buttons
    btn_c1, btn_c2, btn_c3, spacer = st.columns([1.1, 1.3, 1.5, 4])

    with btn_c1:
        if st.button(
            "💾  Save",
            key=f"_save_{key}",
            type="primary",
            use_container_width=True,
            help="Persist this prompt to disk — survives app restarts.",
        ):
            _write_prompt(key, st.session_state[key])
            st.success("Prompt saved to disk.", icon="✅")

    with btn_c2:
        if st.button(
            "↩  Reset",
            key=f"_reset_{key}",
            use_container_width=True,
            help="Restore the built-in default prompt.",
        ):
            st.session_state[key] = default
            _delete_prompt(key)
            st.rerun()

    with btn_c3:
        if not modified:
            status_color, status_text = "#059669", "Default (unchanged)"
        elif _is_saved(key, new_val):
            status_color, status_text = "#4F8EF7", "Modified — saved"
        else:
            status_color, status_text = "#D97706", "Modified — unsaved"
        st.markdown(
            f'<div style="padding-top:0.45rem;font-size:0.82rem;">'
            f'<span style="color:{status_color};font-weight:600;">'
            f'● {status_text}</span>'
            f'&ensp;<span style="color:#3D5A8A;opacity:0.7;">{len(new_val):,} chars</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Diff expander: show what changed vs default
    if modified:
        with st.expander("Show changes vs default prompt", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Default**")
                st.code(default, language="text")
            with col_b:
                st.markdown("**Current**")
                st.code(new_val, language="text")


# ── Status strip ─────────────────────────────────────────────────────────────

def _render_status_strip():
    st.markdown(
        "<small style='color:#888;'>Agent prompt status — click a tab above "
        "to edit any prompt.</small>",
        unsafe_allow_html=True,
    )
    st.markdown("")
    cols = st.columns(len(AGENT_DEFS))
    for col, agent in zip(cols, AGENT_DEFS):
        cur     = st.session_state.get(agent["key"], agent["default"])
        modified = cur.strip() != agent["default"].strip()
        saved    = _is_saved(agent["key"], cur)

        if not modified:
            bar_color, status = "#059669", "Default"
        elif saved:
            bar_color, status = "#4F8EF7", "Saved"
        else:
            bar_color, status = "#D97706", "Modified"

        with col:
            st.markdown(
                f'<div style="background:linear-gradient(160deg,#1C2E45,#243555);'
                f'border-radius:12px;padding:0.8rem 0.4rem;text-align:center;'
                f'border-top:3px solid {bar_color};'
                f'border:1px solid rgba(79,142,247,0.15);'
                f'border-top:3px solid {bar_color};">'
                f'<div style="font-size:1.4rem;">{agent["icon"]}</div>'
                f'<div style="font-weight:600;font-size:0.75rem;'
                f'color:#C8D8F0;margin:0.2rem 0;white-space:nowrap;'
                f'overflow:hidden;text-overflow:ellipsis;">'
                f'{agent["label"]}</div>'
                f'<div style="color:{bar_color};font-weight:700;'
                f'font-size:0.72rem;">{status}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── Global reset ──────────────────────────────────────────────────────────────

def _render_reset_all():
    any_modified = any(
        st.session_state.get(a["key"], a["default"]).strip() != a["default"].strip()
        for a in AGENT_DEFS
    )
    if any_modified:
        if st.button(
            "↩  Reset ALL prompts to built-in defaults",
            type="secondary",
            use_container_width=True,
        ):
            for agent in AGENT_DEFS:
                st.session_state[agent["key"]] = agent["default"]
            if os.path.exists(PROMPTS_FILE):
                os.remove(PROMPTS_FILE)
            st.rerun()


# ── Persistence helpers ───────────────────────────────────────────────────────

def _load_saved_prompts() -> dict:
    if os.path.exists(PROMPTS_FILE):
        try:
            with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _write_prompt(key: str, value: str) -> None:
    saved = _load_saved_prompts()
    saved[key] = value
    with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
        json.dump(saved, f, indent=2, ensure_ascii=False)


def _delete_prompt(key: str) -> None:
    saved = _load_saved_prompts()
    if key in saved:
        del saved[key]
        with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
            json.dump(saved, f, indent=2, ensure_ascii=False)


def _is_saved(key: str, current_value: str) -> bool:
    """True when the current prompt matches what is on disk."""
    return _load_saved_prompts().get(key, "").strip() == current_value.strip()


def _init_defaults():
    """Populate session state once from disk (or built-in defaults)."""
    saved = _load_saved_prompts()
    for agent in AGENT_DEFS:
        if agent["key"] not in st.session_state:
            st.session_state[agent["key"]] = saved.get(
                agent["key"], agent["default"]
            )
