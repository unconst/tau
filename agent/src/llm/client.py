"""LLM Client using httpx for Chutes API (OpenAI-compatible).

Supports both blocking and streaming modes.  The streaming path
(``chat_stream``) yields ``StreamChunk`` objects as SSE events arrive,
enabling real-time text output and early tool-call dispatch.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Set

import httpx

from src.core.budget import AgentBudget


# ---------------------------------------------------------------------------
# Rate-limit state — parsed from response headers
# ---------------------------------------------------------------------------

@dataclass
class RateLimitInfo:
    """Snapshot of rate-limit state parsed from API response headers."""

    remaining_requests: Optional[int] = None
    remaining_tokens: Optional[int] = None
    reset_requests_at: Optional[float] = None  # epoch seconds
    reset_tokens_at: Optional[float] = None     # epoch seconds

    def seconds_until_reset(self) -> Optional[float]:
        """Seconds until the earliest limit resets, or None if unknown."""
        now = time.time()
        candidates = []
        if self.reset_requests_at is not None:
            candidates.append(max(0.0, self.reset_requests_at - now))
        if self.reset_tokens_at is not None:
            candidates.append(max(0.0, self.reset_tokens_at - now))
        return min(candidates) if candidates else None


def _parse_rate_limit_headers(headers: httpx.Headers) -> RateLimitInfo:
    """Extract rate-limit info from response headers (OpenAI convention)."""
    info = RateLimitInfo()

    def _int_or_none(name: str) -> Optional[int]:
        val = headers.get(name)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass
        return None

    def _parse_reset(name: str) -> Optional[float]:
        """Parse reset header — could be epoch seconds, ISO timestamp, or relative duration."""
        val = headers.get(name)
        if val is None:
            return None
        try:
            # Try as epoch float first
            ts = float(val)
            if ts > 1_000_000_000:  # looks like epoch
                return ts
            else:
                # Relative seconds
                return time.time() + ts
        except (ValueError, TypeError):
            pass
        # Try parsing duration strings like "2s", "500ms"
        val_lower = val.lower().strip()
        if val_lower.endswith("ms"):
            try:
                return time.time() + float(val_lower[:-2]) / 1000.0
            except ValueError:
                pass
        elif val_lower.endswith("s"):
            try:
                return time.time() + float(val_lower[:-1])
            except ValueError:
                pass
        return None

    info.remaining_requests = _int_or_none("x-ratelimit-remaining-requests")
    info.remaining_tokens = _int_or_none("x-ratelimit-remaining-tokens")
    info.reset_requests_at = _parse_reset("x-ratelimit-reset-requests")
    info.reset_tokens_at = _parse_reset("x-ratelimit-reset-tokens")
    return info


# ---------------------------------------------------------------------------
# Model validation — fetch & cache available models from the API
# ---------------------------------------------------------------------------

_valid_models_lock = threading.Lock()
_valid_models: Set[str] = set()
_valid_models_fetched_at: float = 0.0
_VALID_MODELS_TTL = 3600.0  # re-fetch every 1 hour (was 5 min — hot-path overhead)

# ---------------------------------------------------------------------------
# Shared httpx.Client pool — avoids TCP/TLS handshake per LLMClient instance.
# Keyed by (base_url, api_key_hash, timeout).  Thread-safe via lock.
# ---------------------------------------------------------------------------
_http_pool_lock = threading.Lock()
_http_pool: Dict[tuple, httpx.Client] = {}


def _get_shared_http_client(
    base_url: str,
    api_key: str,
    timeout: Optional[float],
) -> httpx.Client:
    """Return a shared httpx.Client for the given (base_url, key, timeout) tuple.

    Connection pooling means subsequent requests reuse existing TCP/TLS
    connections, eliminating handshake latency for subagents and retries.
    """
    import hashlib
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
    pool_key = (base_url, key_hash, timeout)

    with _http_pool_lock:
        client = _http_pool.get(pool_key)
        if client is not None:
            return client
        client = httpx.Client(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout=timeout, connect=30.0),
        )
        _http_pool[pool_key] = client
        return client

# Known-good fallback model if the requested one isn't available
_FALLBACK_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def _fetch_available_models(base_url: str, api_key: str) -> Set[str]:
    """Fetch the list of available model IDs from the API."""
    try:
        resp = httpx.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            return {m["id"] for m in data.get("data", []) if "id" in m}
    except Exception:
        pass
    return set()


def validate_model(model: str, base_url: str, api_key: str) -> str:
    """Validate that *model* is available on the API.

    Returns the model name if valid, otherwise returns a known-good
    fallback and prints a warning.  The available-models list is cached
    for ``_VALID_MODELS_TTL`` seconds to avoid hammering the API.
    """
    global _valid_models, _valid_models_fetched_at

    with _valid_models_lock:
        now = time.time()
        if not _valid_models or (now - _valid_models_fetched_at) > _VALID_MODELS_TTL:
            fetched = _fetch_available_models(base_url, api_key)
            if fetched:
                _valid_models = fetched
                _valid_models_fetched_at = now

    # If we have a model list and the requested model isn't in it, fallback
    if _valid_models and model not in _valid_models:
        import sys
        print(
            f"[LLMClient] WARNING: model '{model}' not found in available models. "
            f"Falling back to '{_FALLBACK_MODEL}'.",
            file=sys.stderr,
        )
        return _FALLBACK_MODEL

    return model


class CostLimitExceeded(Exception):
    """Raised when cost limit is exceeded."""

    def __init__(self, message: str, used: float = 0, limit: float = 0):
        super().__init__(message)
        self.used = used
        self.limit = limit


class LLMError(Exception):
    """LLM API error."""

    def __init__(
        self,
        message: str,
        code: str = "unknown",
        rate_limit_info: Optional[RateLimitInfo] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.rate_limit_info = rate_limit_info


class ContextWindowExceeded(LLMError):
    """Raised when the request exceeds the model's context window."""

    def __init__(self, message: str):
        super().__init__(message, code="context_window_exceeded")


@dataclass
class FunctionCall:
    """Represents a function/tool call from the LLM."""

    id: str
    name: str
    arguments: Dict[str, Any]

    @classmethod
    def from_openai(cls, call: Dict[str, Any]) -> "FunctionCall":
        """Parse from OpenAI tool_calls format."""
        func = call.get("function", {})
        args_str = func.get("arguments", "{}")

        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {"raw": args_str}

        return cls(
            id=call.get("id", ""),
            name=func.get("name", ""),
            arguments=args,
        )


@dataclass
class LLMResponse:
    """Response from the LLM."""

    text: str = ""
    function_calls: List[FunctionCall] = field(default_factory=list)
    tokens: Optional[Dict[str, int]] = None
    model: str = ""
    finish_reason: str = ""
    raw: Optional[Dict[str, Any]] = None

    def has_function_calls(self) -> bool:
        """Check if response contains function calls."""
        return bool(self.function_calls) and len(self.function_calls) > 0

    @property
    def is_empty(self) -> bool:
        """True when the model produced no usable output (no text, no tool calls).

        An empty response typically means the model failed silently —
        returning a well-formed HTTP 200 but with zero content.  This
        should be treated as a retryable error, not a valid result.
        """
        has_text = bool(self.text and self.text.strip())
        has_calls = bool(self.function_calls)
        return not has_text and not has_calls


@dataclass
class StreamChunk:
    """A single chunk from a streaming LLM response.

    Exactly one of the content fields will be non-None per chunk.
    """

    # Text delta (partial assistant message)
    text_delta: Optional[str] = None
    # A fully-parsed tool call (emitted once arguments JSON is complete)
    tool_call: Optional[FunctionCall] = None
    # Token usage (emitted with the final chunk)
    tokens: Optional[Dict[str, int]] = None
    # Finish reason (emitted with the final chunk)
    finish_reason: Optional[str] = None


def _log_client(msg: str) -> None:
    """Log to stderr from client module."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [llm] {msg}", file=sys.stderr, flush=True)


class LLMClient:
    """LLM Client using httpx for Chutes API (OpenAI-compatible format).

    Supports both blocking (``chat``) and streaming (``chat_stream``) modes.
    """

    # Default Chutes API configuration
    DEFAULT_BASE_URL = "https://llm.chutes.ai/v1"
    # Accepted env var names for the API key (checked in order)
    API_KEY_ENV_VARS = ("CHUTES_API_TOKEN", "CHUTES_API_KEY")

    def __init__(
        self,
        model: str,
        temperature: Optional[float] = None,
        max_tokens: int = 16384,
        cost_limit: Optional[float] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        budget: Optional[AgentBudget] = None,
        budget_reservation_key: Optional[str] = None,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cost_limit = cost_limit or float(os.environ.get("LLM_COST_LIMIT", "10.0"))
        self.base_url = base_url or os.environ.get("CHUTES_BASE_URL", self.DEFAULT_BASE_URL)
        self.timeout = timeout
        self._budget = budget
        self._budget_reservation_key = budget_reservation_key

        # Get API key — accept CHUTES_API_TOKEN (canonical) or CHUTES_API_KEY (legacy)
        self._api_key = api_key
        if not self._api_key:
            for env_var in self.API_KEY_ENV_VARS:
                self._api_key = os.environ.get(env_var)
                if self._api_key:
                    break
        if not self._api_key:
            raise ValueError(
                f"API key required. Set CHUTES_API_TOKEN environment variable or pass api_key parameter."
            )

        # Validate model against available models (auto-fallback on 404-prone names)
        self.model = validate_model(model, self.base_url, self._api_key)

        self._total_cost = 0.0
        self._total_tokens = 0
        self._request_count = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._cached_tokens = 0
        self._reasoning_tokens = 0

        # Rate-limit tracking
        self._last_rate_limit: Optional[RateLimitInfo] = None

        # Stream retry configuration
        self.stream_max_retries = int(os.environ.get("STREAM_MAX_RETRIES", "5"))
        self.stream_idle_timeout = float(os.environ.get("STREAM_IDLE_TIMEOUT", "300.0"))

        # Create httpx client with timeout
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout=self.timeout, connect=30.0),
        )

    @staticmethod
    def _is_context_window_error(status_code: int, error_msg: str) -> bool:
        """Detect context-window-exceeded errors from status code or message."""
        lowered = error_msg.lower()
        keywords = (
            "context_length_exceeded",
            "context window",
            "maximum context length",
            "token limit",
            "context length",
            "too many tokens",
            "input is too long",
        )
        if status_code == 400 and any(kw in lowered for kw in keywords):
            return True
        return False

    def _raise_http_error(
        self,
        status_code: int,
        error_msg: str,
        headers: Optional[httpx.Headers] = None,
    ) -> None:
        """Map HTTP status to the appropriate LLMError subclass and raise."""
        rl_info = _parse_rate_limit_headers(headers) if headers else None
        if rl_info and rl_info.remaining_requests is not None:
            self._last_rate_limit = rl_info

        if self._is_context_window_error(status_code, error_msg):
            raise ContextWindowExceeded(error_msg)
        if status_code == 401:
            raise LLMError(error_msg, code="authentication_error", rate_limit_info=rl_info)
        elif status_code == 429:
            raise LLMError(error_msg, code="rate_limit", rate_limit_info=rl_info)
        elif status_code >= 500:
            raise LLMError(error_msg, code="server_error", rate_limit_info=rl_info)
        else:
            raise LLMError(
                f"HTTP {status_code}: {error_msg}", code="api_error", rate_limit_info=rl_info
            )

    def _supports_temperature(self, model: str) -> bool:
        """Check if model supports temperature parameter."""
        model_lower = model.lower()
        # Reasoning models don't support temperature
        if any(x in model_lower for x in ["o1", "o3", "deepseek-r1"]):
            return False
        return True

    def _build_tools(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Build tools in OpenAI format."""
        if not tools:
            return None

        result = []
        for tool in tools:
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                    },
                }
            )
        return result

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> LLMResponse:
        """Send a chat request to Chutes API."""
        # Check cost limit
        if self._total_cost >= self.cost_limit or (
            self._budget is not None and self._budget.is_exhausted()
        ):
            raise CostLimitExceeded(
                f"Cost limit exceeded: ${self._total_cost:.4f} >= ${self.cost_limit:.4f}",
                used=self._total_cost,
                limit=self.cost_limit,
            )

        # Per-call model override — skip re-validation to avoid lock +
        # possible HTTP fetch on every LLM call in the hot path.
        effective_model = model or self.model

        # Build request payload
        payload: Dict[str, Any] = {
            "model": effective_model,
            "messages": self._prepare_messages(messages),
            "max_tokens": max_tokens or self.max_tokens,
        }

        if self._supports_temperature(payload["model"]) and self.temperature is not None:
            payload["temperature"] = self.temperature

        if tools:
            payload["tools"] = self._build_tools(tools)
            payload["tool_choice"] = "auto"

        # Add extra body params (like reasoning effort) - some may be ignored by API
        if extra_body:
            payload.update(extra_body)

        try:
            response = self._client.post("/chat/completions", json=payload)
            self._request_count += 1

            # Parse rate-limit headers from every response
            rl_info = _parse_rate_limit_headers(response.headers)
            if rl_info.remaining_requests is not None:
                self._last_rate_limit = rl_info

            # Handle HTTP errors
            if response.status_code != 200:
                error_body = response.text
                try:
                    error_json = response.json()
                    error_msg = error_json.get("error", {}).get("message", error_body)
                except (json.JSONDecodeError, KeyError):
                    error_msg = error_body

                self._raise_http_error(response.status_code, error_msg, response.headers)

            data = response.json()

        except (CostLimitExceeded, LLMError, ContextWindowExceeded):
            raise
        except httpx.TimeoutException as e:
            raise LLMError(f"Request timed out: {e}", code="timeout")
        except httpx.ConnectError as e:
            raise LLMError(f"Connection error: {e}", code="connection_error")
        except httpx.HTTPError as e:
            raise LLMError(f"HTTP error: {e}", code="api_error")

        # Parse response
        result = LLMResponse(raw=data)

        # Extract usage
        usage = data.get("usage", {})
        if usage:
            input_tokens = usage.get("prompt_tokens", 0) or 0
            output_tokens = usage.get("completion_tokens", 0) or 0
            cached_tokens = 0
            reasoning_tokens = 0

            # Check for cached tokens (OpenAI format)
            prompt_details = usage.get("prompt_tokens_details", {})
            if prompt_details:
                cached_tokens = prompt_details.get("cached_tokens", 0) or 0

            # Check for reasoning tokens (OpenAI format)
            completion_details = usage.get("completion_tokens_details", {})
            if completion_details:
                reasoning_tokens = completion_details.get("reasoning_tokens", 0) or 0

            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
            self._cached_tokens += cached_tokens
            self._reasoning_tokens += reasoning_tokens
            self._total_tokens += input_tokens + output_tokens

            result.tokens = {
                "input": input_tokens,
                "output": output_tokens,
                "cached": cached_tokens,
                "reasoning": reasoning_tokens,
            }

            # Estimate cost (generic pricing, adjust per model if needed)
            # Using conservative estimates: $3/1M input, $15/1M output
            cost = (input_tokens * 3.0 / 1_000_000) + (output_tokens * 15.0 / 1_000_000)
            if self._budget is not None and not self._budget.consume(
                cost=cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                reasoning_tokens=reasoning_tokens,
                reservation_key=self._budget_reservation_key,
            ):
                raise CostLimitExceeded(
                    "Shared runtime budget exceeded",
                    used=self._budget.snapshot().consumed_cost,
                    limit=self._budget.max_cost,
                )
            self._total_cost += cost

        # Extract model
        result.model = data.get("model", self.model)

        # Extract choices
        choices = data.get("choices") or []
        if choices:
            choice = choices[0]
            message = choice.get("message") or {}

            result.finish_reason = choice.get("finish_reason", "") or ""
            result.text = message.get("content", "") or ""

            # Extract function calls
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                for call in tool_calls:
                    func = call.get("function") or {}
                    args_str = func.get("arguments", "{}")

                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except json.JSONDecodeError:
                        args = {"raw": args_str}

                    result.function_calls.append(
                        FunctionCall(
                            id=call.get("id", "") or "",
                            name=func.get("name", "") or "",
                            arguments=args if isinstance(args, dict) else {},
                        )
                    )

        # Detect empty responses — model returned HTTP 200 but produced
        # nothing.  Raise as a retryable error so the loop can retry or
        # fall back to another model.
        if result.is_empty:
            _log_client(
                f"Empty response from model {effective_model} "
                f"(finish_reason={result.finish_reason!r})"
            )
            raise LLMError(
                f"Empty response: model '{effective_model}' produced no text and no tool calls "
                f"(finish_reason={result.finish_reason!r})",
                code="empty_response",
            )

        return result

    # -----------------------------------------------------------------
    # Streaming API
    # -----------------------------------------------------------------

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        on_text: Optional[Callable[[str], None]] = None,
        on_tool_call: Optional[Callable[[FunctionCall], None]] = None,
    ) -> LLMResponse:
        """Stream a chat response via SSE, returning the assembled result.

        Includes **stream-level retry**: if the SSE connection drops mid-stream
        (idle timeout, network reset, etc.) the entire request is retried up to
        ``stream_max_retries`` times with exponential back-off.  This is
        separate from the loop-level retry in ``loop.py`` which restarts the
        whole turn.

        Args:
            messages: Conversation messages.
            tools: Tool specifications (OpenAI format).
            max_tokens: Max output tokens.
            extra_body: Extra payload keys (e.g. reasoning effort).
            model: Model override.
            on_text: Callback fired for each text delta.
            on_tool_call: Callback fired when a tool call is complete.

        Returns:
            The fully-assembled LLMResponse (same shape as ``chat``).
        """
        if self._total_cost >= self.cost_limit or (
            self._budget is not None and self._budget.is_exhausted()
        ):
            raise CostLimitExceeded(
                f"Cost limit exceeded: ${self._total_cost:.4f} >= ${self.cost_limit:.4f}",
                used=self._total_cost,
                limit=self.cost_limit,
            )

        # Per-call model override — skip re-validation (same rationale as chat()).
        effective_model = model or self.model

        payload: Dict[str, Any] = {
            "model": effective_model,
            "messages": self._prepare_messages(messages),
            "max_tokens": max_tokens or self.max_tokens,
            "stream": True,
        }

        if self._supports_temperature(payload["model"]) and self.temperature is not None:
            payload["temperature"] = self.temperature

        if tools:
            payload["tools"] = self._build_tools(tools)
            payload["tool_choice"] = "auto"

        if extra_body:
            payload.update(extra_body)

        # Stream-level retry loop
        last_stream_error: Optional[Exception] = None
        for stream_attempt in range(1, self.stream_max_retries + 1):
            # Accumulators for incremental parsing (reset each attempt)
            text_parts: List[str] = []
            tc_accum: Dict[int, Dict[str, Any]] = {}
            finish_reason = ""
            usage: Dict[str, Any] = {}
            stream_completed = False

            try:
                # Use a per-stream timeout for idle detection
                stream_timeout = httpx.Timeout(
                    timeout=None,
                    connect=30.0,
                    read=self.stream_idle_timeout,
                )
                stream_client = httpx.Client(
                    base_url=self.base_url,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=stream_timeout,
                )

                try:
                    with stream_client.stream("POST", "/chat/completions", json=payload) as resp:
                        self._request_count += 1

                        # Parse rate-limit headers
                        rl_info = _parse_rate_limit_headers(resp.headers)
                        if rl_info.remaining_requests is not None:
                            self._last_rate_limit = rl_info

                        if resp.status_code != 200:
                            body_bytes = resp.read()
                            error_body = body_bytes.decode("utf-8", errors="replace")
                            try:
                                error_json = json.loads(error_body)
                                error_msg = error_json.get("error", {}).get("message", error_body)
                            except (json.JSONDecodeError, KeyError):
                                error_msg = error_body
                            self._raise_http_error(resp.status_code, error_msg, resp.headers)

                        for raw_line in resp.iter_lines():
                            line = raw_line.strip()
                            if not line or not line.startswith("data: "):
                                continue
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                stream_completed = True
                                break
                            try:
                                chunk = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            if "usage" in chunk and chunk["usage"]:
                                usage = chunk["usage"]

                            choices = chunk.get("choices") or []
                            if not choices:
                                continue
                            delta = choices[0].get("delta") or {}
                            fr = choices[0].get("finish_reason")
                            if fr:
                                finish_reason = fr

                            content = delta.get("content")
                            if content:
                                text_parts.append(content)
                                if on_text:
                                    on_text(content)

                            tc_deltas = delta.get("tool_calls") or []
                            for tcd in tc_deltas:
                                idx = tcd.get("index", 0)
                                if idx not in tc_accum:
                                    tc_accum[idx] = {
                                        "id": tcd.get("id", ""),
                                        "name": tcd.get("function", {}).get("name", ""),
                                        "arguments_parts": [],
                                    }
                                acc = tc_accum[idx]
                                if tcd.get("id"):
                                    acc["id"] = tcd["id"]
                                fn = tcd.get("function", {})
                                if fn.get("name"):
                                    acc["name"] = fn["name"]
                                if fn.get("arguments"):
                                    acc["arguments_parts"].append(fn["arguments"])

                        # If we read all lines without [DONE], treat as completed
                        # (some providers close the stream without [DONE])
                        stream_completed = True
                finally:
                    stream_client.close()

            except (ContextWindowExceeded, CostLimitExceeded):
                raise  # Never retry these

            except (LLMError, httpx.TimeoutException, httpx.ConnectError, httpx.HTTPError) as e:
                last_stream_error = e
                # Don't retry auth or context window errors
                if isinstance(e, LLMError) and e.code in ("authentication_error", "context_window_exceeded"):
                    raise

                if stream_attempt < self.stream_max_retries:
                    # Compute backoff — respect server-suggested delay if available
                    base_wait = min(20.0, 0.2 * (2 ** (stream_attempt - 1)))
                    suggested_wait: Optional[float] = None
                    if isinstance(e, LLMError) and e.rate_limit_info:
                        suggested_wait = e.rate_limit_info.seconds_until_reset()
                    wait_time = max(base_wait, suggested_wait or 0.0)
                    _log_client(
                        f"Stream error (attempt {stream_attempt}/{self.stream_max_retries}): "
                        f"{type(e).__name__}: {e} — retrying in {wait_time:.1f}s"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    # Exhausted retries — raise as LLMError
                    if isinstance(e, LLMError):
                        raise
                    raise LLMError(
                        f"Stream failed after {self.stream_max_retries} attempts: {e}",
                        code="stream_exhausted",
                    )

            if stream_completed:
                break

        # --- Assemble final LLMResponse ---
        result = LLMResponse()
        result.text = "".join(text_parts)
        result.finish_reason = finish_reason

        # Parse accumulated tool calls
        for idx in sorted(tc_accum.keys()):
            acc = tc_accum[idx]
            args_str = "".join(acc["arguments_parts"])
            try:
                args = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                args = {"raw": args_str}
            fc = FunctionCall(id=acc["id"], name=acc["name"], arguments=args)
            result.function_calls.append(fc)
            if on_tool_call:
                on_tool_call(fc)

        # Track tokens
        if usage:
            input_tokens = usage.get("prompt_tokens", 0) or 0
            output_tokens = usage.get("completion_tokens", 0) or 0
            cached_tokens = 0
            reasoning_tokens = 0
            prompt_details = usage.get("prompt_tokens_details", {})
            if prompt_details:
                cached_tokens = prompt_details.get("cached_tokens", 0) or 0
            completion_details = usage.get("completion_tokens_details", {})
            if completion_details:
                reasoning_tokens = completion_details.get("reasoning_tokens", 0) or 0
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
            self._cached_tokens += cached_tokens
            self._reasoning_tokens += reasoning_tokens
            self._total_tokens += input_tokens + output_tokens
            result.tokens = {
                "input": input_tokens,
                "output": output_tokens,
                "cached": cached_tokens,
                "reasoning": reasoning_tokens,
            }
            cost = (input_tokens * 3.0 / 1_000_000) + (output_tokens * 15.0 / 1_000_000)
            if self._budget is not None and not self._budget.consume(
                cost=cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                reasoning_tokens=reasoning_tokens,
                reservation_key=self._budget_reservation_key,
            ):
                raise CostLimitExceeded(
                    "Shared runtime budget exceeded",
                    used=self._budget.snapshot().consumed_cost,
                    limit=self._budget.max_cost,
                )
            self._total_cost += cost

        result.model = model or self.model

        # Detect empty responses — model returned a valid stream but
        # produced no text and no tool calls.  Treat as retryable.
        if result.is_empty:
            _log_client(
                f"Empty streaming response from model {effective_model} "
                f"(finish_reason={result.finish_reason!r})"
            )
            raise LLMError(
                f"Empty response: model '{effective_model}' produced no text and no tool calls "
                f"(finish_reason={result.finish_reason!r})",
                code="empty_response",
            )

        return result

    @property
    def last_rate_limit(self) -> Optional[RateLimitInfo]:
        """Return the most recent rate-limit snapshot, if any."""
        return self._last_rate_limit

    def _prepare_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare messages for the API, cleaning up any incompatible fields."""
        prepared = []
        for msg in messages:
            new_msg = dict(msg)

            # Handle content with cache_control (Anthropic-specific, strip for OpenAI compat)
            content = new_msg.get("content")
            if isinstance(content, list):
                # Convert multipart format, removing cache_control
                cleaned_parts = []
                for part in content:
                    if isinstance(part, dict):
                        cleaned_part = {k: v for k, v in part.items() if k != "cache_control"}
                        cleaned_parts.append(cleaned_part)
                    else:
                        cleaned_parts.append(part)
                new_msg["content"] = cleaned_parts

            prepared.append(new_msg)

        return prepared

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        stats = {
            "total_tokens": self._total_tokens,
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "cached_tokens": self._cached_tokens,
            "reasoning_tokens": self._reasoning_tokens,
            "total_cost": self._total_cost,
            "request_count": self._request_count,
        }
        if self._budget is not None:
            snap = self._budget.snapshot()
            stats["shared_budget"] = {
                "max_cost": snap.max_cost,
                "consumed_cost": snap.consumed_cost,
                "reserved_cost": snap.reserved_cost,
                "remaining_cost": snap.remaining_cost,
            }
        return stats

    @property
    def budget(self) -> Optional[AgentBudget]:
        """Return the shared runtime budget, if configured."""
        return self._budget

    def attach_budget(self, budget: Optional[AgentBudget]) -> None:
        """Attach/replace shared runtime budget."""
        self._budget = budget

    def set_budget_reservation_key(self, reservation_key: Optional[str]) -> None:
        """Set reservation key used for budget consumption."""
        self._budget_reservation_key = reservation_key

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# Alias for backward compatibility
LiteLLMClient = LLMClient
