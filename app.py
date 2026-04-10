"""
Sepsis Diagnostic Agent — Streamlit Application
Entry point: configures the page, injects CSS, renders the active section.
"""

import streamlit as st

from app.secrets import init_into_session
from app.css import CUSTOM_CSS
from app import settings, dashboard, controller, workspace

# ── Bootstrap ────────────────────────────────────────────────────────────────

init_into_session()

st.set_page_config(
    page_title="Sepsis Diagnostic Agent",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Sidebar navigation ──────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🩺 Sepsis Agent")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "⚙️ Settings",
            "📊 Dashboard",
            "🎛️ Agent Controller",
            "🤖 Agent Workspace",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#888'>Powered by LangGraph + MIMIC-IV<br>"
        "DESIGNED BY PARSA</small>",
        unsafe_allow_html=True,
    )

# ── Page dispatch ────────────────────────────────────────────────────────────

if page == "⚙️ Settings":
    settings.render()
elif page == "📊 Dashboard":
    dashboard.render()
elif page == "🎛️ Agent Controller":
    controller.render()
elif page == "🤖 Agent Workspace":
    workspace.render()
