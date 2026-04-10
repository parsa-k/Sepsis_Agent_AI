"""LLM provider instantiation and Gemini model-fallback logic."""

from __future__ import annotations

import streamlit as st

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

MODEL_DEFAULTS = {
    "OpenAI": "gpt-4o",
    "Anthropic Claude": "claude-sonnet-4-20250514",
    "Google Gemini": "gemini-2.5-flash",
}


def get_llm(model_override: str | None = None):
    """Return a LangChain chat model based on current session settings."""
    provider = st.session_state.get("llm_provider", "Google Gemini")
    model_name = (
        model_override
        or st.session_state.get("model_name", "").strip()
    )

    if provider == "OpenAI":
        api_key = st.session_state.get("openai_key", "")
        if not api_key:
            return None
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name or MODEL_DEFAULTS["OpenAI"],
            api_key=api_key,
            temperature=0.1,
        )

    if provider == "Anthropic Claude":
        api_key = st.session_state.get("anthropic_key", "")
        if not api_key:
            return None
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name or MODEL_DEFAULTS["Anthropic Claude"],
            api_key=api_key,
            temperature=0.1,
        )

    if provider == "Google Gemini":
        api_key = st.session_state.get("google_key", "")
        if not api_key:
            return None
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name or MODEL_DEFAULTS["Google Gemini"],
            google_api_key=api_key,
            temperature=0.1,
            max_retries=2,
        )

    return None


def test_gemini_with_fallback(
    api_key: str,
    preferred_model: str,
) -> tuple[bool, str, str]:
    """Try *preferred_model*, then fall back through alternatives on 429.

    Returns (success, model_used, message_or_error).
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage

    models_to_try = [preferred_model] + [
        m for m in GEMINI_MODELS if m != preferred_model
    ]
    last_error: Exception | None = None

    for model in models_to_try:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=0.1,
                max_retries=1,
            )
            resp = llm.invoke(
                [HumanMessage(content="Respond with exactly: Connection successful.")]
            )
            return True, model, resp.content[:100]
        except Exception as e:
            last_error = e
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                continue
            return False, model, str(e)

    return False, preferred_model, str(last_error)
