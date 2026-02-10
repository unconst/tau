"""Session management for SuperAgent.

Defines the ``AgentContext`` protocol that callers must implement,
and the ``Session`` class that tracks conversation state.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

from src.config.models import AgentConfig


# =============================================================================
# AgentContext Protocol
# =============================================================================


class ShellResult:
    """Result from a shell command execution."""

    __slots__ = ("output", "exit_code", "stdout", "stderr")

    def __init__(
        self,
        output: str,
        exit_code: int,
        stdout: str = "",
        stderr: str = "",
    ):
        self.output = output
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


@runtime_checkable
class AgentContext(Protocol):
    """Protocol that callers must implement to drive the agent loop.

    Any object that has these attributes/methods can be passed to
    ``run_agent_loop()`` without duck-typing hacks.
    """

    instruction: str
    cwd: str
    is_done: bool

    def shell(self, cmd: str, timeout: int = 120) -> ShellResult: ...

    def done(self) -> None: ...


class SimpleAgentContext:
    """Concrete implementation of AgentContext for standalone use.

    Replaces the ad-hoc ``_Ctx`` classes that were previously duplicated
    in multiple places.
    """

    def __init__(
        self,
        instruction: str,
        cwd: str | None = None,
        *,
        depth: int = 0,
        max_subagent_depth: int = 1,
        trace_id: str | None = None,
        parent_trace_id: str | None = None,
        subagent_id: str | None = None,
    ):
        self.instruction = instruction
        self.cwd = cwd or str(Path.cwd())
        self.is_done = False
        self.depth = depth
        self.max_subagent_depth = max_subagent_depth
        self.trace_id = trace_id
        self.parent_trace_id = parent_trace_id
        self.subagent_id = subagent_id
        self.runtime_constraints: dict[str, Any] = {}
        self.agent_budget: Any = None

    def shell(self, cmd: str, timeout: int = 120) -> ShellResult:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=self.cwd,
        )
        return ShellResult(
            output=r.stdout + r.stderr,
            exit_code=r.returncode,
            stdout=r.stdout,
            stderr=r.stderr,
        )

    def done(self) -> None:
        self.is_done = True


@dataclass
class TokenUsage:
    """Token usage tracking."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, other: "TokenUsage") -> None:
        """Add usage from another TokenUsage instance."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_tokens += other.cached_tokens


@dataclass
class Message:
    """A message in the conversation history."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    name: Optional[str] = None  # For tool messages

    def to_dict(self) -> dict[str, Any]:
        """Convert to API format."""
        msg: dict[str, Any] = {"role": self.role, "content": self.content}

        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id

        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls

        if self.name:
            msg["name"] = self.name

        return msg


@dataclass
class Session:
    """Manages the state of an agent session."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    config: AgentConfig = field(default_factory=AgentConfig)
    cwd: Path = field(default_factory=Path.cwd)

    # Conversation history
    messages: list[Message] = field(default_factory=list)

    # Token usage
    usage: TokenUsage = field(default_factory=TokenUsage)

    # Iteration tracking
    iteration: int = 0

    # Timestamps
    started_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

    # Status
    is_done: bool = False
    final_message: Optional[str] = None
    approval_cache: dict[str, bool] = field(default_factory=dict)
    checkpoint_dir_name: str = ".agent/sessions"

    def add_system_message(self, content: str) -> None:
        """Add a system message."""
        self.messages.append(Message(role="system", content=content))
        self._update_activity()

    def add_user_message(self, content: str) -> None:
        """Add a user message."""
        self.messages.append(Message(role="user", content=content))
        self._update_activity()

    def add_assistant_message(
        self,
        content: str,
        tool_calls: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Add an assistant message."""
        self.messages.append(
            Message(
                role="assistant",
                content=content,
                tool_calls=tool_calls,
            )
        )
        self._update_activity()

    def add_tool_result(self, tool_call_id: str, name: str, content: str) -> None:
        """Add a tool result message."""
        self.messages.append(
            Message(
                role="tool",
                content=content,
                tool_call_id=tool_call_id,
                name=name,
            )
        )
        self._update_activity()

    def get_messages_for_api(self) -> list[dict[str, Any]]:
        """Get messages formatted for the API."""
        return [msg.to_dict() for msg in self.messages]

    def update_usage(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> None:
        """Update token usage."""
        self.usage.input_tokens += input_tokens
        self.usage.output_tokens += output_tokens
        self.usage.cached_tokens += cached_tokens

    def increment_iteration(self) -> bool:
        """Increment iteration and check if we should continue.

        Returns:
            True if we can continue, False if max iterations reached
        """
        self.iteration += 1
        return self.iteration < self.config.max_iterations

    def mark_done(self, final_message: Optional[str] = None) -> None:
        """Mark the session as done."""
        self.is_done = True
        self.final_message = final_message
        self._update_activity()

    def _update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now()

    def is_approved(self, key: str) -> bool:
        """Check whether an operation is approved for this session."""
        return self.approval_cache.get(key, False)

    def approve_for_session(self, key: str) -> None:
        """Cache an approval for the current session."""
        self.approval_cache[key] = True

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return (datetime.now() - self.started_at).total_seconds()

    def checkpoint_path(self) -> Path:
        """Return the checkpoint path for this session."""
        base = self.cwd / self.checkpoint_dir_name
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{self.id}.json"

    def save_rollout(
        self,
        *,
        messages: list[dict[str, Any]],
        iteration: int,
        pending_completion: bool,
        tool_call_count: int,
        usage: dict[str, int],
    ) -> Path:
        """Persist session state for crash-safe resume."""
        payload = {
            "session_id": self.id,
            "saved_at": datetime.now().isoformat(),
            "cwd": str(self.cwd),
            "iteration": iteration,
            "pending_completion": pending_completion,
            "tool_call_count": tool_call_count,
            "usage": usage,
            "approval_cache": self.approval_cache,
            "messages": messages,
        }
        path = self.checkpoint_path()
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load_rollout(
        cls,
        cwd: Path,
        session_id: str | None = None,
        resume_latest: bool = False,
    ) -> dict[str, Any] | None:
        """Load rollout state from disk."""
        base = cwd / ".agent/sessions"
        if not base.exists():
            return None

        target: Path | None = None
        if session_id:
            candidate = base / f"{session_id}.json"
            if candidate.exists():
                target = candidate
        elif resume_latest:
            candidates = sorted(base.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                target = candidates[0]

        if target is None:
            return None

        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            return None
