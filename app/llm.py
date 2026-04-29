"""LLM provider instantiation and Gemini model-fallback logic."""

from __future__ import annotations

import time
import logging
import streamlit as st

logger = logging.getLogger(__name__)

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

_TRANSIENT_MARKERS = ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE")


def _is_transient(exc: Exception) -> bool:
    """Return True if this exception is a retryable API overload error."""
    s = str(exc)
    return any(m in s for m in _TRANSIENT_MARKERS)


# ── Wrapper LLM ──────────────────────────────────────────────────────────────

class GeminiFallbackLLM:
    """
    Drop-in replacement for ChatGoogleGenerativeAI that automatically
    falls back through GEMINI_MODELS when a 503/429 is encountered on
    *any* call to .invoke() — including calls made deep inside the
    LangGraph pipeline.

    All attribute accesses other than .invoke() are forwarded to the
    currently active underlying LLM instance, so LangChain internals
    (e.g. bind_tools, with_config, etc.) continue to work transparently.
    """

    def __init__(
        self,
        api_key: str,
        models: list[str],
        temperature: float = 0.1,
        max_retries_per_model: int = 1,
    ):
        self._api_key = api_key
        self._models = models
        self._temperature = temperature
        self._max_retries_per_model = max_retries_per_model
        self._model_idx = 0
        self._llm = self._make_llm(models[0])

    # ── internal helpers ─────────────────────────────────────────────────────

    def _make_llm(self, model_name: str):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self._api_key,
            temperature=self._temperature,
            max_retries=self._max_retries_per_model,
        )

    @property
    def current_model(self) -> str:
        return self._models[self._model_idx]

    # ── primary interface ────────────────────────────────────────────────────

    def invoke(self, input, config=None, **kwargs):
        """
        Call the active model's .invoke(). On 503/429, promote to the
        next model and retry, until all models are exhausted.
        """
        start_idx = self._model_idx

        while self._model_idx < len(self._models):
            try:
                return self._llm.invoke(input, config, **kwargs) if config is not None \
                    else self._llm.invoke(input, **kwargs)
            except Exception as exc:
                if not _is_transient(exc):
                    raise  # auth errors, bad requests, etc. — surface immediately

                logger.warning(
                    "GeminiFallbackLLM: %s on model '%s'; trying next.",
                    type(exc).__name__, self.current_model,
                )

                self._model_idx += 1
                if self._model_idx >= len(self._models):
                    raise RuntimeError(
                        f"All Gemini models exhausted after 503/429 errors. "
                        f"Last error: {exc}"
                    ) from exc

                # Brief back-off before switching models
                time.sleep(2)
                self._llm = self._make_llm(self._models[self._model_idx])
                logger.info(
                    "GeminiFallbackLLM: switched to '%s'.", self.current_model
                )

    # ── transparent attribute forwarding ────────────────────────────────────

    def __getattr__(self, name: str):
        """Forward any unknown attribute access to the active LLM."""
        return getattr(self._llm, name)


# ── Public factory functions ─────────────────────────────────────────────────

def get_llm(model_override: str | None = None):
    """Return a plain LangChain chat model based on current session settings.

    For Gemini, prefer get_llm_with_fallback() which handles 503/429
    automatically during pipeline execution.
    """
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


def get_llm_with_fallback(model_override: str | None = None):
    """Return a GeminiFallbackLLM (or plain LLM for non-Gemini providers).

    The returned object's .invoke() method automatically retries on
    503/429 by cycling through GEMINI_MODELS — even mid-pipeline.

    Returns:
        (llm, preferred_model_name, None)  — llm is None if no key configured.
    """
    provider = st.session_state.get("llm_provider", "Google Gemini")

    if provider != "Google Gemini":
        return get_llm(model_override=model_override), None, None

    api_key = st.session_state.get("google_key", "")
    if not api_key:
        return None, None, None

    preferred = (
        model_override
        or st.session_state.get("model_name", "").strip()
        or MODEL_DEFAULTS["Google Gemini"]
    )

    # Build the ordered fallback list: preferred first, rest after
    models_to_try = [preferred] + [m for m in GEMINI_MODELS if m != preferred]

    llm = GeminiFallbackLLM(
        api_key=api_key,
        models=models_to_try,
        temperature=0.1,
        max_retries_per_model=1,
    )
    return llm, preferred, None


def test_gemini_with_fallback(
    api_key: str,
    preferred_model: str,
) -> tuple[bool, str, str]:
    """Try *preferred_model*, then fall back through alternatives on 429/503.

    Returns (success, model_used, message_or_error).
    Used by the Settings page connection test.
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
            if (
                "429" in err_str
                or "RESOURCE_EXHAUSTED" in err_str
                or "503" in err_str
                or "UNAVAILABLE" in err_str
            ):
                continue
            return False, model, str(e)

    return False, preferred_model, str(last_error)
