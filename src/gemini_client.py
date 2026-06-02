"""
Backward-compatible re-export from unified llm_client.

All LLM call logic now lives in src.llm_client.
This file is kept so existing imports continue to work.
"""

from .llm_client import (
    call_llm,
    call_llm_gemini,
    set_log_path,
    LAST_THOUGHT,
    DEFAULT_MODEL as GEMINI_DEFAULT_MODEL,
    DEFAULT_BASE_URL as GEMINI_COMPAT_BASE_URL,
    DEFAULT_BACKEND as GEMINI_DEFAULT_BACKEND,
    DEFAULT_THINKING_LEVEL as THINKING_LEVEL,
)

__all__ = [
    "call_llm",
    "call_llm_gemini",
    "set_log_path",
    "LAST_THOUGHT",
    "GEMINI_DEFAULT_MODEL",
    "GEMINI_COMPAT_BASE_URL",
    "GEMINI_DEFAULT_BACKEND",
    "THINKING_LEVEL",
]
