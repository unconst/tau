"""Subagent tool — spawn lightweight child agent loops for task delegation.

Supports two subagent types:
  - **explore**: Fast, read-only. Uses a cheaper model and only read-only tools
    (read_file, list_dir, grep_files, web_search). Good for codebase exploration,
    searching, and answering questions.
  - **execute**: Full-capability. Uses the main model and all tools.
    Good for independent subtasks that need to write files or run commands.

The parent agent receives the subagent's final message as the tool result.
Up to ``MAX_CONCURRENT`` (4) subagents can run in parallel.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from src.tools.base import ToolResult

MAX_CONCURRENT = 4

# Read-only tools for explore subagents
EXPLORE_TOOLS = {"read_file", "list_dir", "grep_files", "web_search", "view_image"}

# Tool specs for the subagent
SUBAGENT_SPEC: Dict[str, Any] = {
    "name": "spawn_subagent",
    "description": (
        "Spawn a lightweight subagent to handle a subtask autonomously. "
        "Use 'explore' type for fast read-only codebase exploration (cheaper model, read-only tools). "
        "Use 'execute' type for subtasks that need to write files or run commands (full model, all tools). "
        "The subagent runs independently and returns its final message. "
        "Up to 4 subagents can run in parallel."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": (
                    "A detailed description of what the subagent should do. "
                    "Be specific — the subagent has no context from the parent conversation."
                ),
            },
            "type": {
                "type": "string",
                "enum": ["explore", "execute"],
                "description": (
                    "'explore' = fast model, read-only tools (for searching/understanding). "
                    "'execute' = full model, all tools (for making changes)."
                ),
            },
            "cwd": {
                "type": "string",
                "description": "Working directory for the subagent (defaults to parent cwd).",
            },
        },
        "required": ["task", "type"],
    },
}


def run_subagent(
    task: str,
    subagent_type: str = "explore",
    cwd: Optional[str] = None,
    max_iterations: int = 50,
) -> ToolResult:
    """Run a subagent loop and return its final message.

    Args:
        task: The task prompt for the subagent.
        subagent_type: 'explore' or 'execute'.
        cwd: Working directory.
        max_iterations: Max iterations for the subagent loop.

    Returns:
        ToolResult with the subagent's final message.
    """
    from src.core.loop import run_agent_loop
    from src.core.session import SimpleAgentContext
    from src.llm.client import LLMClient
    from src.llm.router import ModelRouter
    from src.output.jsonl import set_event_callback
    from src.tools.registry import ToolRegistry

    effective_cwd = cwd or os.getcwd()
    is_explore = subagent_type == "explore"

    # Use ModelRouter to select the right model tier for this subagent type
    router = ModelRouter()
    tier = router.for_subagent(subagent_type)

    # Create a fresh LLM client with the routed model
    llm = LLMClient(
        model=tier.model,
        temperature=0.0,
        max_tokens=tier.max_tokens,
        timeout=120.0 if is_explore else 300.0,
    )

    # Create tool registry — restrict for explore
    tools = ToolRegistry(cwd=Path(effective_cwd))

    if is_explore:
        # Remove write-capable tools from the registry by overriding execute
        original_execute = tools.execute

        def restricted_execute(ctx, name, arguments):
            if name not in EXPLORE_TOOLS:
                return ToolResult.fail(
                    f"Tool '{name}' not available in explore mode. "
                    f"Available: {', '.join(sorted(EXPLORE_TOOLS))}"
                )
            return original_execute(ctx, name, arguments)

        tools.execute = restricted_execute  # type: ignore[assignment]
    else:
        # Block nested subagent spawning to prevent infinite recursion
        original_execute = tools.execute

        def no_nesting_execute(ctx, name, arguments):
            if name == "spawn_subagent":
                return ToolResult.fail("Nested subagents are not allowed.")
            return original_execute(ctx, name, arguments)

        tools.execute = no_nesting_execute  # type: ignore[assignment]

    # Build subagent config
    config = {
        "max_iterations": 30 if is_explore else max_iterations,
        "max_tokens": tier.max_tokens,
        "max_output_tokens": 2000,
        "reasoning_effort": "none",
        "cache_enabled": True,
    }

    ctx = SimpleAgentContext(instruction=task, cwd=effective_cwd)

    # Capture final agent message
    captured: list[str] = []

    def _capture(event: Dict[str, Any]) -> None:
        if event.get("type") == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message" and item.get("text"):
                captured.append(item["text"])

    set_event_callback(_capture)

    try:
        run_agent_loop(llm=llm, tools=tools, ctx=ctx, config=config)
    except Exception as e:
        return ToolResult.fail(f"Subagent error: {e}")
    finally:
        set_event_callback(None)
        llm.close()

    final_msg = captured[-1] if captured else "(subagent produced no output)"
    return ToolResult.ok(final_msg)
