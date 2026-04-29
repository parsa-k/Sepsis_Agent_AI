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
    st.markdown(
        '<div style="padding:0.5rem 0 0.25rem 0;">'
        '<span style="font-size:1.35rem;font-weight:800;'
        'background:linear-gradient(135deg,#4F8EF7,#7C3AED);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;">'
        'Sepsis Agent</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<hr style="border-color:rgba(79,142,247,0.25);margin:0.5rem 0 1rem 0;">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p style="font-size:0.7rem;font-weight:600;letter-spacing:0.1em;'
        'color:#4F8EF7;opacity:0.8;margin:0 0 0.4rem 0;">USER</p>',
        unsafe_allow_html=True,
    )
    st.button("Settings",        on_click=set_page, args=("Settings",),        use_container_width=True, type="primary"   if st.session_state.current_page == "Settings"        else "secondary")
    st.button("Dataset Explorer",        on_click=set_page, args=("Dashboard",),       use_container_width=True, type="primary"   if st.session_state.current_page == "Dashboard"       else "secondary")
    st.button("Agent Workspace",  on_click=set_page, args=("Agent Workspace",), use_container_width=True, type="primary"   if st.session_state.current_page == "Agent Workspace" else "secondary")
    st.button("Patient History",  on_click=set_page, args=("Patient History",), use_container_width=True, type="primary"   if st.session_state.current_page == "Patient History" else "secondary")

    st.markdown(
        '<p style="font-size:0.7rem;font-weight:600;letter-spacing:0.1em;'
        'color:#4F8EF7;opacity:0.8;margin:1rem 0 0.4rem 0;">ADMIN</p>',
        unsafe_allow_html=True,
    )
    st.button("Agent Controller", on_click=set_page, args=("Agent Controller",), use_container_width=True, type="primary" if st.session_state.current_page == "Agent Controller" else "secondary")

    st.markdown(
        '<hr style="border-color:rgba(79,142,247,0.18);margin:1.5rem 0 0.75rem 0;">',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:0.72rem;color:rgba(138,170,206,0.6);'
        'letter-spacing:0.08em;text-align:center;">PARSA</p>',
        unsafe_allow_html=True,
    )

# ── Page dispatch ────────────────────────────────────────────────────────────

page = st.session_state.current_page

# When the active page changes, force a clean rerun before rendering anything.
# This clears Streamlit's widget delta buffer so no stale widgets from the
# previous page bleed into the new one (a known Streamlit session-state
# routing bug — most visible when tab-heavy pages like Agent Controller
# are followed by other tabbed pages like Patient History).
if st.session_state.get("_active_page") != page:
    st.session_state["_active_page"] = page
    st.rerun()

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
