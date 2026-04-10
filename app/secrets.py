"""API key persistence — load, save, and initialise into Streamlit session."""

import json
import os

import streamlit as st

SECRETS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".secrets.json",
)

_PERSISTED_KEYS = (
    "openai_key",
    "anthropic_key",
    "google_key",
    "llm_provider",
    "model_name",
)


def load_saved() -> dict:
    """Read the local secrets JSON file (returns {} on any error)."""
    if os.path.exists(SECRETS_FILE):
        try:
            with open(SECRETS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save(data: dict):
    """Write *data* to the local secrets JSON file."""
    with open(SECRETS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def init_into_session():
    """On first Streamlit run, load any saved keys into session state."""
    if "_secrets_loaded" not in st.session_state:
        saved = load_saved()
        for key in _PERSISTED_KEYS:
            if key in saved and key not in st.session_state:
                st.session_state[key] = saved[key]
        st.session_state["_secrets_loaded"] = True


def secrets_file_exists() -> bool:
    return os.path.exists(SECRETS_FILE)
