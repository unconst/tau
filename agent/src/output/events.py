"""Event types for agent JSONL output."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class EventType(str, Enum):
    """Types of events that can be emitted."""

    TURN_STARTED = "turn.started"
    TURN_COMPLETED = "turn.completed"
    TURN_FAILED = "turn.failed"

    ITEM_STARTED = "item.started"
    ITEM_UPDATED = "item.updated"
    ITEM_COMPLETED = "item.completed"

    MESSAGE = "message"
    THINKING = "thinking"

    TOOL_CALL_START = "tool.call.start"
    TOOL_CALL_END = "tool.call.end"
    STREAM_TEXT_DELTA = "stream.text.delta"
    STREAM_RETRY = "stream.retry"
    STREAM_TOOL_STARTED = "stream.tool.started"
    STREAM_TOOL_COMPLETED = "stream.tool.completed"
    TOOL_DECISION = "tool.decision"
    TOOL_ESCALATION = "tool.escalation"
    POLICY_EVALUATION = "policy.evaluation"

    ERROR = "error"


@dataclass
class Event:
    """An event from the agent."""

    type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            **self.data,
        }

    @classmethod
    def turn_started(cls, session_id: str) -> "Event":
        """Create a turn started event."""
        return cls(
            type=EventType.TURN_STARTED,
            data={"session_id": session_id},
        )

    @classmethod
    def turn_completed(
        cls,
        session_id: str,
        final_message: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> "Event":
        """Create a turn completed event."""
        return cls(
            type=EventType.TURN_COMPLETED,
            data={
                "session_id": session_id,
                "final_message": final_message,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cached_tokens": cached_tokens,
                },
            },
        )

    @classmethod
    def message(cls, content: str, role: str = "assistant") -> "Event":
        """Create a message event."""
        return cls(
            type=EventType.MESSAGE,
            data={"content": content, "role": role},
        )

    @classmethod
    def thinking(cls) -> "Event":
        """Create a thinking event."""
        return cls(type=EventType.THINKING)

    @classmethod
    def tool_call_start(cls, name: str, arguments: dict[str, Any]) -> "Event":
        """Create a tool call start event."""
        return cls(
            type=EventType.TOOL_CALL_START,
            data={"name": name, "arguments": arguments},
        )

    @classmethod
    def tool_call_end(
        cls,
        name: str,
        success: bool,
        output: str,
        error: Optional[str] = None,
    ) -> "Event":
        """Create a tool call end event."""
        return cls(
            type=EventType.TOOL_CALL_END,
            data={
                "name": name,
                "success": success,
                "output": output,
                "error": error,
            },
        )

    @classmethod
    def error(cls, message: str, details: Optional[dict[str, Any]] = None) -> "Event":
        """Create an error event."""
        return cls(
            type=EventType.ERROR,
            data={"message": message, "details": details or {}},
        )

    @classmethod
    def stream_text_delta(cls, delta: str) -> "Event":
        """Create a streaming text-delta event."""
        return cls(type=EventType.STREAM_TEXT_DELTA, data={"delta": delta})

    @classmethod
    def stream_retry(
        cls,
        attempt: int,
        max_attempts: int,
        wait_seconds: int,
        error_code: str,
    ) -> "Event":
        """Create a stream retry event."""
        return cls(
            type=EventType.STREAM_RETRY,
            data={
                "attempt": attempt,
                "max_attempts": max_attempts,
                "wait_seconds": wait_seconds,
                "error_code": error_code,
            },
        )

    @classmethod
    def tool_decision(
        cls,
        tool_name: str,
        decision: str,
        reason: Optional[str],
        source: str,
    ) -> "Event":
        """Create a tool decision event."""
        return cls(
            type=EventType.TOOL_DECISION,
            data={
                "tool_name": tool_name,
                "decision": decision,
                "reason": reason,
                "source": source,
            },
        )

    @classmethod
    def tool_escalation(
        cls,
        tool_name: str,
        attempt: int,
        retried_without_guards: bool,
    ) -> "Event":
        """Create a tool escalation event."""
        return cls(
            type=EventType.TOOL_ESCALATION,
            data={
                "tool_name": tool_name,
                "attempt": attempt,
                "retried_without_guards": retried_without_guards,
            },
        )

    @classmethod
    def policy_evaluation(
        cls,
        tool_name: str,
        evaluation: Optional[dict[str, Any]],
        fallback: bool,
    ) -> "Event":
        """Create a policy evaluation event."""
        return cls(
            type=EventType.POLICY_EVALUATION,
            data={
                "tool_name": tool_name,
                "evaluation": evaluation,
                "fallback": fallback,
            },
        )
