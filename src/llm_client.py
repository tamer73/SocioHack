"""
Unified LLM Client for SocioHack.

Consolidates all LLM API calls into a single module with two backends:
  - "openai_compat": OpenAI SDK with configurable base_url.
    Works with Gemini-compatible proxies and local vLLM servers.
  - "google_sdk": Google GenAI SDK with optional custom base_url.

Default model: gemini-3-flash-preview
Default backend: google_sdk

Usage:
    from src.llm_client import call_llm

    # Gemini via OpenAI-compat endpoint (default)
    resp = call_llm("What is 1+1?")

    # Gemini via Google SDK
    resp = call_llm("What is 1+1?", backend="google_sdk")

    # Local vLLM server
    resp = call_llm("What is 1+1?", api_key="EMPTY",
                     model="my-model",
                     base_url="http://localhost:8421/v1",
                     thinking_level=None)
"""

import os
import time
import threading
from typing import Optional

try:
    from google import genai
    from google.genai import types as genai_types
    _GOOGLE_SDK_AVAILABLE = True
except ImportError:
    _GOOGLE_SDK_AVAILABLE = False

from openai import OpenAI

# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
DEFAULT_GOOGLE_SDK_BASE_URL = os.getenv("GEMINI_SDK_BASE_URL") or None
DEFAULT_BACKEND = os.getenv("GEMINI_BACKEND", "google_sdk")
DEFAULT_THINKING_LEVEL = "minimal"

# ── Module state ────────────────────────────────────────────────────────────

_openai_client_cache: dict = {}
_google_client_cache: dict = {}
_log_file_path: Optional[str] = None
_log_lock = threading.Lock()

# Last thinking/reasoning content from the most recent call.
# Thread-unsafe global; kept for backward compat with code that reads
# gemini_client.LAST_THOUGHT.
LAST_THOUGHT: str = ""


def set_log_path(path: str):
    """Set file path for detailed LLM call logging."""
    global _log_file_path
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _log_file_path = path


def _log(message: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {message}"
    if _log_file_path:
        with _log_lock:
            with open(_log_file_path, "a", encoding="utf-8") as f:
                f.write(formatted + "\n")
    else:
        print(formatted)


# ── Client caches ───────────────────────────────────────────────────────────

def _get_openai_client(api_key: str, base_url: str) -> OpenAI:
    cache_key = (api_key, base_url)
    if cache_key not in _openai_client_cache:
        _openai_client_cache[cache_key] = OpenAI(api_key=api_key, base_url=base_url)
    return _openai_client_cache[cache_key]


def _get_google_client(api_key: str, base_url: Optional[str] = None):
    cache_key = (api_key, base_url)
    if cache_key not in _google_client_cache:
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["http_options"] = {"base_url": base_url}
        _google_client_cache[cache_key] = genai.Client(**kwargs)
    return _google_client_cache[cache_key]


# ── OpenAI-compat backend ──────────────────────────────────────────────────

def _call_openai_compat(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    max_retries: int,
    retry_delay: float,
    thinking_level: Optional[str],
    timeout: Optional[int],
) -> str:
    global LAST_THOUGHT
    client = _get_openai_client(api_key, base_url)
    last_error = None
    start = time.time()
    _log(f"[openai_compat] call started (model={model}, base_url={base_url})")

    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            if thinking_level:
                kwargs["reasoning_effort"] = thinking_level
            if timeout:
                kwargs["timeout"] = timeout
            resp = client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            LAST_THOUGHT = getattr(msg, "reasoning_content", "") or ""
            # Print token usage to terminal
            usage = getattr(resp, "usage", None)
            if usage:
                input_tok = getattr(usage, "prompt_tokens", 0) or 0
                output_tok = getattr(usage, "completion_tokens", 0) or 0
                # Extract thinking tokens from completion_tokens_details if available
                details = getattr(usage, "completion_tokens_details", None)
                thinking_tok = getattr(details, "reasoning_tokens", 0) if details else 0
                response_tok = output_tok - thinking_tok
                print(f"[LLM tokens] input={input_tok}  thinking={thinking_tok}  response={response_tok}  total_output={output_tok}")
            duration = time.time() - start
            _log(f"[openai_compat] SUCCESS in {duration:.2f}s")
            return msg.content or ""
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            is_rate_limit = "429" in err_str or "rate" in err_str or "quota" in err_str
            wait = retry_delay * (2 ** attempt)
            if is_rate_limit:
                wait = max(wait, 60.0)
            _log(f"[openai_compat] attempt {attempt + 1} FAILED: {e}. retry in {wait:.0f}s")
            time.sleep(wait)

    _log(f"[openai_compat] FATAL: all {max_retries} attempts failed: {last_error}")
    return ""


# ── Google SDK backend ──────────────────────────────────────────────────────

def _call_google_sdk(
    prompt: str,
    api_key: str,
    model: str,
    base_url: Optional[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    max_retries: int,
    retry_delay: float,
    thinking_level: Optional[str],
) -> str:
    if not _GOOGLE_SDK_AVAILABLE:
        raise ImportError("google-genai package not installed.")

    global LAST_THOUGHT
    client = _get_google_client(api_key, base_url)
    last_error = None
    start = time.time()
    _log(f"[google_sdk] call started (model={model}, base_url={base_url})")

    config_kwargs = dict(temperature=temperature, top_p=top_p, max_output_tokens=max_tokens)
    if thinking_level:
        config_kwargs["thinking_config"] = genai_types.ThinkingConfig(thinking_level=thinking_level)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(**config_kwargs),
            )
            text_parts, thought_parts = [], []
            if hasattr(response, "candidates") and response.candidates:
                for part in response.candidates[0].content.parts:
                    if getattr(part, "thought", False):
                        thought_parts.append(getattr(part, "text", ""))
                    elif getattr(part, "text", ""):
                        text_parts.append(part.text)
            LAST_THOUGHT = "".join(thought_parts).strip()
            # Print token usage to terminal
            meta = getattr(response, "usage_metadata", None)
            if meta:
                input_tok = getattr(meta, "prompt_token_count", 0) or 0
                thinking_tok = getattr(meta, "thoughts_token_count", 0) or 0
                output_tok = getattr(meta, "candidates_token_count", 0) or 0
                response_tok = output_tok - thinking_tok
                print(f"[LLM tokens] input={input_tok}  thinking={thinking_tok}  response={response_tok}  total_output={output_tok}")
            duration = time.time() - start
            _log(f"[google_sdk] SUCCESS in {duration:.2f}s")
            return "".join(text_parts).strip()
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            is_rate_limit = "429" in err_str or "quota" in err_str
            wait = retry_delay * (2 ** attempt)
            if is_rate_limit:
                wait = max(wait, 60.0)
            _log(f"[google_sdk] attempt {attempt + 1} FAILED: {e}. retry in {wait:.0f}s")
            time.sleep(wait)

    _log(f"[google_sdk] FATAL: all {max_retries} attempts failed: {last_error}")
    return ""


# ── Public API ──────────────────────────────────────────────────────────────

def call_llm(
    prompt: str,
    *,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    base_url: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_retries: int = 8,
    retry_delay: float = 5.0,
    backend: Optional[str] = None,
    thinking_level: Optional[str] = DEFAULT_THINKING_LEVEL,
    timeout: Optional[int] = None,
) -> str:
    """Unified LLM call.

    Args:
        prompt: The prompt text.
        api_key: API key. Falls back to GEMINI_API_KEY env var.
        model: Model identifier (default: gemini-3-flash-preview).
        base_url: Override the default endpoint URL.
            - openai_compat: defaults to DEFAULT_BASE_URL
            - google_sdk: defaults to DEFAULT_GOOGLE_SDK_BASE_URL (None = official)
        max_tokens: Maximum response tokens.
        temperature: Sampling temperature.
        top_p: Top-p sampling.
        max_retries: Number of retry attempts.
        retry_delay: Base delay between retries (exponential backoff).
        backend: "openai_compat" or "google_sdk". Defaults to DEFAULT_BACKEND.
        thinking_level: Thinking budget ("minimal"/"low"/"medium"/"high").
            Set to None to skip (e.g. for local vLLM models).
        timeout: Request timeout in seconds (openai_compat only).

    Returns:
        Response text, or empty string on failure.
    """
    resolved_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_key:
        raise ValueError("Missing API key. Set GEMINI_API_KEY env var or pass api_key=.")

    # # Append "Skip the think." to reduce thinking token usage for Gemini calls
    # if thinking_level and not prompt.rstrip().endswith("Do not think, just answer."):
    #     prompt = prompt.rstrip() + "\nDo not think, just answer."

    resolved_backend = backend or DEFAULT_BACKEND

    if resolved_backend == "google_sdk":
        return _call_google_sdk(
            prompt, resolved_key, model,
            base_url if base_url is not None else DEFAULT_GOOGLE_SDK_BASE_URL,
            max_tokens, temperature, top_p,
            max_retries, retry_delay, thinking_level,
        )
    return _call_openai_compat(
        prompt, resolved_key, model,
        base_url or DEFAULT_BASE_URL,
        max_tokens, temperature, top_p,
        max_retries, retry_delay, thinking_level, timeout,
    )


# ── Backward-compatible alias ───────────────────────────────────────────────

def call_llm_gemini(
    prompt: str,
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_retries: int = 8,
    retry_delay: float = 5.0,
    backend: Optional[str] = None,
    thinking_level: Optional[str] = DEFAULT_THINKING_LEVEL,
) -> str:
    """Backward-compatible wrapper. Prefer call_llm() for new code."""
    return call_llm(
        prompt,
        api_key=api_key,
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        max_retries=max_retries,
        retry_delay=retry_delay,
        backend=backend,
        thinking_level=thinking_level,
    )


# ── Backward-compatible constants ───────────────────────────────────────────

GEMINI_DEFAULT_MODEL = DEFAULT_MODEL
GEMINI_COMPAT_BASE_URL = DEFAULT_BASE_URL
GEMINI_DEFAULT_BACKEND = DEFAULT_BACKEND
THINKING_LEVEL = DEFAULT_THINKING_LEVEL
