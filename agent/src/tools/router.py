"""Tool call parsing and routing."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable, List

from src.tools.base import ToolResult
from src.tools.specs import validate_tool_arguments


@dataclass
class ToolInvocation:
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    parse_error: str | None = None
    validation_errors: list[str] | None = None


class ToolRouter:
    """Parses and dispatches LLM tool calls."""

    @staticmethod
    def parse_calls(function_calls: Iterable[Any]) -> List[ToolInvocation]:
        if function_calls is None:
            return []
        parsed: list[ToolInvocation] = []
        for idx, call in enumerate(function_calls):
            call_id = getattr(call, "id", None) or f"call_{idx}"
            tool_name = getattr(call, "name", "")
            raw_args = getattr(call, "arguments", {}) or {}
            parse_error: str | None = None
            args: dict[str, Any]
            if isinstance(raw_args, dict):
                args = raw_args
            elif isinstance(raw_args, str):
                try:
                    loaded = json.loads(raw_args)
                    if isinstance(loaded, dict):
                        args = loaded
                    else:
                        args = {}
                        parse_error = "arguments must decode to a JSON object"
                except json.JSONDecodeError:
                    args = {}
                    parse_error = "arguments must be valid JSON"
            else:
                args = {}
                parse_error = "arguments must be an object"
            validation_errors = validate_tool_arguments(tool_name, args)
            parsed.append(
                ToolInvocation(
                    call_id=call_id,
                    tool_name=tool_name,
                    arguments=args,
                    parse_error=parse_error,
                    validation_errors=validation_errors,
                )
            )
        return parsed

    @staticmethod
    def normalize_error(tool_name: str, err: Exception) -> ToolResult:
        return ToolResult.fail(f"Tool `{tool_name}` failed: {err}")

