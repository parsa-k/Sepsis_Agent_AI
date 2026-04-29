"""
Sepsis Diagnostic Agent — Streamlit Application
Entry point: configures the page, injects CSS, renders the active section.
"""

import streamlit as st

from app.secrets import init_into_session
from app.css import CUSTOM_CSS
from app import settings, dashboard, controller, workspace, history

# ── Bootstrap ────────────────────────────────────────────────────────────────

init_into_session()

st.set_page_config(
    page_title="Sepsis Diagnostic Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if "current_page" not in st.session_state:
    st.session_state.current_page = "Settings"

def set_page(page_name):
    st.session_state.current_page = page_name

# ── Sidebar navigation ──────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Sepsis Agent")
    st.markdown("---")
    
    st.markdown("### User")
    st.button("Settings", on_click=set_page, args=("Settings",), use_container_width=True, type="primary" if st.session_state.current_page == "Settings" else "secondary")
    st.button("Dashboard", on_click=set_page, args=("Dashboard",), use_container_width=True, type="primary" if st.session_state.current_page == "Dashboard" else "secondary")
    st.button("Agent Workspace", on_click=set_page, args=("Agent Workspace",), use_container_width=True, type="primary" if st.session_state.current_page == "Agent Workspace" else "secondary")
    st.button("Patient History", on_click=set_page, args=("Patient History",), use_container_width=True, type="primary" if st.session_state.current_page == "Patient History" else "secondary")
        
    st.markdown("### Admin")
    st.button("Agent Controller", on_click=set_page, args=("Agent Controller",), use_container_width=True, type="primary" if st.session_state.current_page == "Agent Controller" else "secondary")

    st.markdown("---")
    st.markdown(
        "<small style='color:#888'>PARSA</small>",
        unsafe_allow_html=True,
    )

# ── Page dispatch ────────────────────────────────────────────────────────────

page = st.session_state.current_page

if page == "Settings":
    settings.render()
elif page == "Dashboard":
    dashboard.render()
elif page == "Agent Controller":
    controller.render()
elif page == "Agent Workspace":
    workspace.render()
elif page == "Patient History":
    history.render()
