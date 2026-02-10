"""LLM Client using httpx for Chutes API (OpenAI-compatible).

Supports both blocking and streaming modes.  The streaming path
(``chat_stream``) yields ``StreamChunk`` objects as SSE events arrive,
enabling real-time text output and early tool-call dispatch.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional

import httpx


class CostLimitExceeded(Exception):
    """Raised when cost limit is exceeded."""

    def __init__(self, message: str, used: float = 0, limit: float = 0):
        super().__init__(message)
        self.used = used
        self.limit = limit


class LLMError(Exception):
    """LLM API error."""

    def __init__(self, message: str, code: str = "unknown"):
        super().__init__(message)
        self.message = message
        self.code = code


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
        return len(self.function_calls) > 0


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
        timeout: float = 120.0,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cost_limit = cost_limit or float(os.environ.get("LLM_COST_LIMIT", "10.0"))
        self.base_url = base_url or os.environ.get("CHUTES_BASE_URL", self.DEFAULT_BASE_URL)
        self.timeout = timeout

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

        self._total_cost = 0.0
        self._total_tokens = 0
        self._request_count = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._cached_tokens = 0

        # Create httpx client with timeout
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout, connect=30.0),
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
        if self._total_cost >= self.cost_limit:
            raise CostLimitExceeded(
                f"Cost limit exceeded: ${self._total_cost:.4f} >= ${self.cost_limit:.4f}",
                used=self._total_cost,
                limit=self.cost_limit,
            )

        # Build request payload
        payload: Dict[str, Any] = {
            "model": model or self.model,
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

            # Handle HTTP errors
            if response.status_code != 200:
                error_body = response.text
                try:
                    error_json = response.json()
                    error_msg = error_json.get("error", {}).get("message", error_body)
                except (json.JSONDecodeError, KeyError):
                    error_msg = error_body

                # Map status codes to error codes
                if response.status_code == 401:
                    raise LLMError(error_msg, code="authentication_error")
                elif response.status_code == 429:
                    raise LLMError(error_msg, code="rate_limit")
                elif response.status_code >= 500:
                    raise LLMError(error_msg, code="server_error")
                else:
                    raise LLMError(f"HTTP {response.status_code}: {error_msg}", code="api_error")

            data = response.json()

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

            # Check for cached tokens (OpenAI format)
            prompt_details = usage.get("prompt_tokens_details", {})
            if prompt_details:
                cached_tokens = prompt_details.get("cached_tokens", 0) or 0

            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
            self._cached_tokens += cached_tokens
            self._total_tokens += input_tokens + output_tokens

            result.tokens = {
                "input": input_tokens,
                "output": output_tokens,
                "cached": cached_tokens,
            }

            # Estimate cost (generic pricing, adjust per model if needed)
            # Using conservative estimates: $3/1M input, $15/1M output
            cost = (input_tokens * 3.0 / 1_000_000) + (output_tokens * 15.0 / 1_000_000)
            self._total_cost += cost

        # Extract model
        result.model = data.get("model", self.model)

        # Extract choices
        choices = data.get("choices", [])
        if choices:
            choice = choices[0]
            message = choice.get("message", {})

            result.finish_reason = choice.get("finish_reason", "") or ""
            result.text = message.get("content", "") or ""

            # Extract function calls
            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                for call in tool_calls:
                    func = call.get("function", {})
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

        This is the preferred method for interactive use.  Text deltas are
        emitted via callbacks as they arrive and complete tool calls are
        dispatched as soon as their argument JSON is fully received.

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
        if self._total_cost >= self.cost_limit:
            raise CostLimitExceeded(
                f"Cost limit exceeded: ${self._total_cost:.4f} >= ${self.cost_limit:.4f}",
                used=self._total_cost,
                limit=self.cost_limit,
            )

        payload: Dict[str, Any] = {
            "model": model or self.model,
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

        # Accumulators for incremental parsing
        text_parts: List[str] = []
        # tool_call index -> {id, name, arguments_parts}
        tc_accum: Dict[int, Dict[str, Any]] = {}
        finish_reason = ""
        usage: Dict[str, Any] = {}

        try:
            with self._client.stream("POST", "/chat/completions", json=payload) as resp:
                self._request_count += 1
                if resp.status_code != 200:
                    # Read full body for error
                    body_bytes = resp.read()
                    error_body = body_bytes.decode("utf-8", errors="replace")
                    try:
                        error_json = json.loads(error_body)
                        error_msg = error_json.get("error", {}).get("message", error_body)
                    except (json.JSONDecodeError, KeyError):
                        error_msg = error_body
                    if resp.status_code == 401:
                        raise LLMError(error_msg, code="authentication_error")
                    elif resp.status_code == 429:
                        raise LLMError(error_msg, code="rate_limit")
                    elif resp.status_code >= 500:
                        raise LLMError(error_msg, code="server_error")
                    else:
                        raise LLMError(f"HTTP {resp.status_code}: {error_msg}", code="api_error")

                for raw_line in resp.iter_lines():
                    line = raw_line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Extract usage if present (some providers send it on the last chunk)
                    if "usage" in chunk and chunk["usage"]:
                        usage = chunk["usage"]

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    fr = choices[0].get("finish_reason")
                    if fr:
                        finish_reason = fr

                    # --- Text delta ---
                    content = delta.get("content")
                    if content:
                        text_parts.append(content)
                        if on_text:
                            on_text(content)

                    # --- Tool call deltas ---
                    tc_deltas = delta.get("tool_calls", [])
                    for tcd in tc_deltas:
                        idx = tcd.get("index", 0)
                        if idx not in tc_accum:
                            tc_accum[idx] = {
                                "id": tcd.get("id", ""),
                                "name": tcd.get("function", {}).get("name", ""),
                                "arguments_parts": [],
                            }
                        acc = tc_accum[idx]
                        # Update id/name if provided (first chunk)
                        if tcd.get("id"):
                            acc["id"] = tcd["id"]
                        fn = tcd.get("function", {})
                        if fn.get("name"):
                            acc["name"] = fn["name"]
                        if fn.get("arguments"):
                            acc["arguments_parts"].append(fn["arguments"])

        except httpx.TimeoutException as e:
            raise LLMError(f"Request timed out: {e}", code="timeout")
        except httpx.ConnectError as e:
            raise LLMError(f"Connection error: {e}", code="connection_error")
        except httpx.HTTPError as e:
            raise LLMError(f"HTTP error: {e}", code="api_error")

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
            prompt_details = usage.get("prompt_tokens_details", {})
            if prompt_details:
                cached_tokens = prompt_details.get("cached_tokens", 0) or 0
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
            self._cached_tokens += cached_tokens
            self._total_tokens += input_tokens + output_tokens
            result.tokens = {"input": input_tokens, "output": output_tokens, "cached": cached_tokens}
            cost = (input_tokens * 3.0 / 1_000_000) + (output_tokens * 15.0 / 1_000_000)
            self._total_cost += cost

        result.model = model or self.model
        return result

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
        return {
            "total_tokens": self._total_tokens,
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "cached_tokens": self._cached_tokens,
            "total_cost": self._total_cost,
            "request_count": self._request_count,
        }

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
