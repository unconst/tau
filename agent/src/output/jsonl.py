"""
JSONL output format for structured agent events.

Emits events to stdout in JSONL format for machine consumption.
All events follow a consistent structure for machine-readable output.

Event Types:
- thread.started: New thread/session started
- turn.started: Turn (request/response cycle) started
- turn.completed: Turn completed with usage stats
- turn.failed: Turn failed with error
- item.started: Item (tool call, message) started
- item.updated: Item updated (e.g., todo list)
- item.completed: Item completed
- stream.text.delta: Streaming assistant text chunk
- stream.tool.started: Tool lifecycle start marker
- stream.tool.completed: Tool lifecycle completion marker
- stream.retry: LLM stream retry/backoff notification
- error: Fatal error
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

# Thread-local storage for event callbacks.
# When set, emit() routes events to the callback instead of stdout.
_tls = threading.local()


def set_event_callback(callback: Callable[[Dict[str, Any]], None] | None) -> None:
    """Set an event callback for the current thread.

    When set, :func:`emit` will call *callback(event_dict)* instead of
    printing to stdout.  Pass ``None`` to clear.
    """
    _tls.event_callback = callback


def get_event_callback() -> Callable[[Dict[str, Any]], None] | None:
    """Return the event callback for the current thread, or ``None``."""
    return getattr(_tls, "event_callback", None)

# =============================================================================
# Thread Events
# =============================================================================


@dataclass
class ThreadStartedEvent:
    """Emitted when a new thread/session is started."""

    thread_id: str
    trace_id: Optional[str] = None
    parent_trace_id: Optional[str] = None
    subagent_id: Optional[str] = None
    depth: Optional[int] = None
    type: str = field(default="thread.started", init=False)


# =============================================================================
# Turn Events
# =============================================================================


@dataclass
class TurnStartedEvent:
    """Emitted when a turn is started (user sends message)."""

    trace_id: Optional[str] = None
    parent_trace_id: Optional[str] = None
    subagent_id: Optional[str] = None
    depth: Optional[int] = None
    type: str = field(default="turn.started", init=False)


@dataclass
class Usage:
    """Token usage statistics."""

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class TurnCompletedEvent:
    """Emitted when a turn is completed successfully."""

    usage: Dict[str, int]
    trace_id: Optional[str] = None
    parent_trace_id: Optional[str] = None
    subagent_id: Optional[str] = None
    depth: Optional[int] = None
    type: str = field(default="turn.completed", init=False)


@dataclass
class TurnFailedEvent:
    """Emitted when a turn fails."""

    error: Dict[str, str]
    trace_id: Optional[str] = None
    parent_trace_id: Optional[str] = None
    subagent_id: Optional[str] = None
    depth: Optional[int] = None
    type: str = field(default="turn.failed", init=False)


# =============================================================================
# Item Events
# =============================================================================


@dataclass
class ItemStartedEvent:
    """Emitted when an item starts processing."""

    item: Dict[str, Any]
    type: str = field(default="item.started", init=False)


@dataclass
class ItemUpdatedEvent:
    """Emitted when an item is updated (e.g., todo list)."""

    item: Dict[str, Any]
    type: str = field(default="item.updated", init=False)


@dataclass
class ItemCompletedEvent:
    """Emitted when an item completes processing."""

    item: Dict[str, Any]
    type: str = field(default="item.completed", init=False)


@dataclass
class StreamTextDeltaEvent:
    """Emitted for incremental assistant text streaming."""

    delta: str
    type: str = field(default="stream.text.delta", init=False)


@dataclass
class StreamToolStartedEvent:
    """Emitted when the runtime starts a tool call."""

    tool_name: str
    call_id: str
    type: str = field(default="stream.tool.started", init=False)


@dataclass
class StreamToolCompletedEvent:
    """Emitted when the runtime finishes a tool call."""

    tool_name: str
    call_id: str
    success: bool
    retried_without_guards: bool = False
    type: str = field(default="stream.tool.completed", init=False)


@dataclass
class StreamRetryEvent:
    """Emitted when the runtime retries a failed stream request."""

    attempt: int
    max_attempts: int
    wait_seconds: int
    error_code: str
    type: str = field(default="stream.retry", init=False)


@dataclass
class StreamErrorEvent:
    """Emitted for recoverable stream/runtime errors."""

    stage: str
    error_code: str
    message: str
    consecutive_failures: int
    type: str = field(default="stream.error", init=False)


@dataclass
class ToolDecisionEvent:
    """Emitted after policy decision for a tool call."""

    tool_name: str
    call_id: str
    decision: str
    reason: Optional[str]
    source: str
    approval_outcome: str
    type: str = field(default="tool.decision", init=False)


@dataclass
class ToolEscalationEvent:
    """Emitted when a tool call escalates retries."""

    tool_name: str
    call_id: str
    attempt: int
    retried_without_guards: bool
    type: str = field(default="tool.escalation", init=False)


@dataclass
class PolicyEvaluationEvent:
    """Emitted with exec-policy or heuristic evaluation details."""

    tool_name: str
    call_id: str
    evaluation: Optional[Dict[str, Any]]
    fallback: bool
    type: str = field(default="policy.evaluation", init=False)


@dataclass
class PlanProposedEvent:
    """Emitted when the agent proposes a plan for user approval."""

    plan: str
    type: str = field(default="plan.proposed", init=False)


@dataclass
class PlanApprovedEvent:
    """Emitted when the user approves a proposed plan."""

    plan: str
    type: str = field(default="plan.approved", init=False)


@dataclass
class UserInputRequestedEvent:
    """Emitted when the agent requests user input via ask_user tool."""

    question: str
    options: Optional[List[str]] = None
    request_id: str = ""
    type: str = field(default="user_input.requested", init=False)


@dataclass
class UserInputReceivedEvent:
    """Emitted when user input is received in response to ask_user."""

    answer: str
    request_id: str = ""
    type: str = field(default="user_input.received", init=False)


@dataclass
class TurnMetricsEvent:
    """Emitted with machine-readable turn metrics."""

    session_id: str
    iterations: int
    llm_retries: int
    compactions: int
    parallel_batches: int
    approval_denials: int
    guard_escalations: int
    completion_reason: str
    type: str = field(default="turn.metrics", init=False)


# =============================================================================
# Error Events
# =============================================================================


@dataclass
class ErrorEvent:
    """Emitted for fatal errors."""

    message: str
    type: str = field(default="error", init=False)


# =============================================================================
# Item Types (for item.started/completed payloads)
# =============================================================================


def make_agent_message_item(item_id: str, text: str) -> Dict[str, Any]:
    """Create an agent_message item."""
    return {
        "id": item_id,
        "type": "agent_message",
        "text": text,
    }


def make_reasoning_item(item_id: str, text: str) -> Dict[str, Any]:
    """Create a reasoning item."""
    return {
        "id": item_id,
        "type": "reasoning",
        "text": text,
    }


def make_command_execution_item(
    item_id: str,
    command: str,
    status: str = "in_progress",
    aggregated_output: str = "",
    exit_code: Optional[int] = None,
) -> Dict[str, Any]:
    """Create a command_execution item."""
    item = {
        "id": item_id,
        "type": "command_execution",
        "command": command,
        "status": status,
        "aggregated_output": aggregated_output,
    }
    if exit_code is not None:
        item["exit_code"] = exit_code
    return item


def make_file_change_item(
    item_id: str,
    changes: List[Dict[str, str]],
    status: str = "completed",
) -> Dict[str, Any]:
    """Create a file_change item."""
    return {
        "id": item_id,
        "type": "file_change",
        "changes": changes,
        "status": status,
    }


def make_todo_list_item(
    item_id: str,
    items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Create a todo_list item."""
    return {
        "id": item_id,
        "type": "todo_list",
        "items": items,
    }


def make_error_item(item_id: str, message: str) -> Dict[str, Any]:
    """Create an error item."""
    return {
        "id": item_id,
        "type": "error",
        "message": message,
    }


# =============================================================================
# Emitter
# =============================================================================

_item_counter = 0


def next_item_id() -> str:
    """Generate the next item ID."""
    global _item_counter
    _item_counter += 1
    return f"item_{_item_counter}"


def reset_item_counter() -> None:
    """Reset the item counter (for testing)."""
    global _item_counter
    _item_counter = 0


def emit(event) -> None:
    """
    Emit a single JSONL event.

    If a thread-local event callback has been registered via
    :func:`set_event_callback`, the event dict is passed to it.
    Otherwise the event is printed to stdout as a JSON line.

    Args:
        event: Dataclass event to emit
    """
    try:
        data = asdict(event)
        cb = get_event_callback()
        if cb is not None:
            cb(data)
        else:
            line = json.dumps(data, ensure_ascii=False)
            print(line, flush=True)
    except Exception as e:
        # Fallback: emit error event
        error_data = {"type": "error", "message": f"Failed to emit event: {e}"}
        cb = get_event_callback()
        if cb is not None:
            cb(error_data)
        else:
            print(json.dumps(error_data), flush=True)


def emit_raw(data: Dict[str, Any]) -> None:
    """
    Emit a raw dictionary as JSONL.

    If a thread-local event callback has been registered via
    :func:`set_event_callback`, the dict is passed to it.
    Otherwise it is printed to stdout as a JSON line.

    Args:
        data: Dictionary to emit
    """
    try:
        cb = get_event_callback()
        if cb is not None:
            cb(data)
        else:
            line = json.dumps(data, ensure_ascii=False)
            print(line, flush=True)
    except Exception as e:
        error_data = {"type": "error", "message": f"Failed to emit: {e}"}
        cb = get_event_callback()
        if cb is not None:
            cb(error_data)
        else:
            print(json.dumps(error_data), flush=True)
