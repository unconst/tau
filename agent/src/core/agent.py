"""High-level Agent class — thin wrapper around ``run_agent_loop``.

Provides an OOP interface for callers who prefer class-based usage
over the procedural ``run_agent_loop`` function.  All actual logic
lives in ``loop.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.config.models import AgentConfig
from src.core.loop import run_agent_loop
from src.core.session import SimpleAgentContext
from src.llm.client import LLMClient
from src.output.jsonl import set_event_callback
from src.output.processor import OutputProcessor
from src.prompts.system import get_system_prompt
from src.tools.registry import ToolRegistry


class Agent:
    """Main agent that wraps ``run_agent_loop`` with a clean OOP interface.

    Usage::

        agent = Agent(cwd=Path("/my/project"))
        result = agent.run("Fix the failing tests")
        print(result)
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        cwd: Optional[Path] = None,
        output_processor: Optional[OutputProcessor] = None,
    ):
        self.config = config or AgentConfig()
        self.cwd = cwd or Path(self.config.paths.cwd or ".").resolve()
        self.output = output_processor

    def run(
        self,
        prompt: str,
        on_message: Optional[Callable[[str], None]] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        readonly: bool = False,
        resume_session_id: Optional[str] = None,
        resume_latest: bool = False,
    ) -> str:
        """Run the agent with a user prompt.

        Args:
            prompt: User's instruction/prompt.
            on_message: Optional callback for each assistant message.
            system_prompt: Optional system prompt override.
            model: Optional model override.
            readonly: If True, restrict to read-only tools.

        Returns:
            Final assistant message.
        """
        effective_model = model or self.config.model
        llm = LLMClient(
            model=effective_model,
            temperature=0.0,
            max_tokens=self.config.max_tokens,
            timeout=300.0,
        )

        tools = ToolRegistry(cwd=self.cwd)
        ctx = SimpleAgentContext(instruction=prompt, cwd=str(self.cwd))

        config: Dict[str, Any] = {
            "max_iterations": self.config.max_iterations,
            "max_tokens": self.config.max_tokens,
            "max_output_tokens": 2500,
            "reasoning_effort": self.config.reasoning.effort.value,
            "cache_enabled": self.config.cache.enabled,
            "streaming": True,
            "approval_policy": "on-failure",
            "bypass_approvals": False,
            "bypass_sandbox": False,
            "readonly": readonly,
            "readable_roots": self.config.paths.readable_roots,
            "writable_roots": self.config.paths.writable_roots,
            "resume_session_id": resume_session_id,
            "resume_latest": resume_latest,
        }

        captured_messages: list[str] = []

        def _capture(event: Dict[str, Any]) -> None:
            if event.get("type") == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message" and item.get("text"):
                    text = item["text"]
                    captured_messages.append(text)
                    if on_message:
                        on_message(text)

        set_event_callback(_capture)

        try:
            run_agent_loop(
                llm=llm,
                tools=tools,
                ctx=ctx,
                config=config,
                system_prompt=system_prompt,
            )
        finally:
            set_event_callback(None)
            llm.close()

        return captured_messages[-1] if captured_messages else ""
