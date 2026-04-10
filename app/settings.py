"""Settings & Configuration page."""

import streamlit as st

from app.llm import get_llm, test_gemini_with_fallback, MODEL_DEFAULTS
from app.secrets import load_saved, save, secrets_file_exists

# Widget keys use a `_w_` prefix so Streamlit's widget lifecycle doesn't
# erase the persistent values when the user navigates to another page.
_WIDGET_TO_PERSISTENT = {
    "_w_llm_provider": "llm_provider",
    "_w_model_name": "model_name",
    "_w_openai_key": "openai_key",
    "_w_anthropic_key": "anthropic_key",
    "_w_google_key": "google_key",
}


def _restore_widget_keys():
    """Copy persistent values → widget keys before widgets render."""
    for wk, sk in _WIDGET_TO_PERSISTENT.items():
        if sk in st.session_state and wk not in st.session_state:
            st.session_state[wk] = st.session_state[sk]


def _sync_from_widget(widget_key: str):
    """Copy a single widget value → its persistent key."""
    sk = _WIDGET_TO_PERSISTENT.get(widget_key)
    if sk and widget_key in st.session_state:
        st.session_state[sk] = st.session_state[widget_key]


def render():
    _restore_widget_keys()

    st.markdown("# ⚙️ Settings & Configuration")
    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### LLM Provider")
        provider = st.selectbox(
            "Select provider",
            ["Google Gemini", "OpenAI", "Anthropic Claude"],
            key="_w_llm_provider",
        )
        st.session_state["llm_provider"] = provider

        st.markdown("### Model Name *(optional)*")
        st.text_input(
            "Model identifier (leave blank for default)",
            key="_w_model_name",
            placeholder=MODEL_DEFAULTS.get(provider, ""),
            help=(
                "Optional. Leave blank to use the default: "
                f"**{MODEL_DEFAULTS.get(provider, '')}**"
            ),
        )
        _sync_from_widget("_w_model_name")

    with col2:
        st.markdown("### API Keys")
        _render_api_key_input(provider)
        _render_save_load_buttons()

        if secrets_file_exists():
            st.markdown(
                "<small style='color:#888;'>Keys are saved locally in "
                "<code>.secrets.json</code> (not committed to git).</small>",
                unsafe_allow_html=True,
            )

        st.markdown("### Connection Test")
        if st.button("Test LLM Connection", use_container_width=True):
            _run_connection_test(provider)


# ── Private helpers ──────────────────────────────────────────────────────────

def _render_api_key_input(provider: str):
    if provider == "OpenAI":
        st.text_input("OpenAI API Key", type="password",
                       key="_w_openai_key", help="sk-...")
        _sync_from_widget("_w_openai_key")
    elif provider == "Anthropic Claude":
        st.text_input("Anthropic API Key", type="password",
                       key="_w_anthropic_key", help="sk-ant-...")
        _sync_from_widget("_w_anthropic_key")
    elif provider == "Google Gemini":
        st.text_input("Google API Key", type="password",
                       key="_w_google_key", help="Your Gemini API key")
        _sync_from_widget("_w_google_key")


def _render_save_load_buttons():
    save_col, load_col = st.columns(2)
    with save_col:
        if st.button(
            "💾 Save Keys",
            use_container_width=True,
            help="Save API keys locally so they persist across sessions",
        ):
            secrets_to_save = {}
            for k in ("openai_key", "anthropic_key", "google_key",
                       "llm_provider", "model_name"):
                val = st.session_state.get(k, "")
                if val:
                    secrets_to_save[k] = val
            save(secrets_to_save)
            st.success("Keys saved to `.secrets.json`")

    with load_col:
        if st.button(
            "📂 Load Keys",
            use_container_width=True,
            help="Load previously saved API keys",
        ):
            saved = load_saved()
            if saved:
                for k, v in saved.items():
                    st.session_state[k] = v
                    wk = next(
                        (w for w, s in _WIDGET_TO_PERSISTENT.items() if s == k),
                        None,
                    )
                    if wk:
                        st.session_state[wk] = v
                st.success(f"Loaded {len(saved)} saved setting(s).")
                st.rerun()
            else:
                st.warning("No saved keys found.")


def _run_connection_test(provider: str):
    if provider == "Google Gemini":
        _test_gemini(provider)
    else:
        _test_generic(provider)


def _test_gemini(provider: str):
    api_key = st.session_state.get("google_key", "")
    if not api_key:
        st.error("Please enter a valid API key first.")
        return

    preferred = (
        st.session_state.get("model_name", "").strip()
        or MODEL_DEFAULTS["Google Gemini"]
    )
    with st.spinner(
        f"Testing Gemini models (starting with {preferred})..."
    ):
        ok, used_model, msg = test_gemini_with_fallback(
            api_key, preferred
        )

    if ok:
        if used_model != preferred:
            st.warning(
                f"Model `{preferred}` quota exhausted (429). "
                f"Fell back to **`{used_model}`** successfully."
            )
            st.session_state["model_name"] = used_model
        st.success(
            f"Connected via **{used_model}**! Response: {msg}"
        )
    else:
        st.error(f"Connection failed on `{used_model}`: {msg}")
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            st.info(
                "All Gemini free-tier models are rate-limited. "
                "Options:\n"
                "- Wait a few minutes and try again\n"
                "- Use a paid API key with billing enabled\n"
                "- Try a different provider (OpenAI / Anthropic)"
            )


def _test_generic(provider: str):
    llm = get_llm()
    if llm is None:
        st.error("Please enter a valid API key first.")
        return
    with st.spinner("Testing connection..."):
        try:
            from langchain_core.messages import HumanMessage
            resp = llm.invoke(
                [HumanMessage(content="Respond with: Connection successful.")]
            )
            st.success(f"Connected! Response: {resp.content[:100]}")
        except Exception as e:
            st.error(f"Connection failed: {e}")
