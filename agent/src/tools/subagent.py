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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.budget import AgentBudget
from src.tools.base import ToolResult

MAX_CONCURRENT = 4

# Read-only tools for explore subagents
EXPLORE_TOOLS = {"read_file", "list_dir", "grep_files", "web_search", "view_image"}

# Tool spec for comparison subagents
COMPARISON_SPEC: Dict[str, Any] = {
    "name": "spawn_comparison",
    "description": (
        "Spawn multiple subagents in parallel to explore different approaches to a task. "
        "Each approach runs independently and their results are returned for comparison. "
        "Use this when there are multiple valid strategies and you want to evaluate "
        "which produces the best result before committing to one. "
        "Maximum 3 approaches. Each approach uses an explore subagent (read-only)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The base task description (shared context for all approaches).",
            },
            "approaches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": "Short label for this approach (e.g. 'Approach A: Redis cache')",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Specific instructions for this approach variant.",
                        },
                    },
                    "required": ["label", "prompt"],
                },
                "description": "List of approaches to compare (max 3).",
            },
            "cwd": {
                "type": "string",
                "description": "Working directory for the subagents (defaults to parent cwd).",
            },
        },
        "required": ["task", "approaches"],
    },
}

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
            "timeout_seconds": {
                "type": "number",
                "description": "Optional overall wall-clock timeout for the subagent run.",
            },
            "llm_timeout_seconds": {
                "type": "number",
                "description": "Optional timeout for each LLM API call inside the subagent.",
            },
            "tool_timeout_seconds": {
                "type": "number",
                "description": "Optional timeout for each tool execution inside the subagent.",
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
    parent_constraints: Optional[Dict[str, Any]] = None,
    budget: Optional[AgentBudget] = None,
    timeout_seconds: Optional[int] = None,
    llm_timeout_seconds: Optional[int] = None,
    tool_timeout_seconds: Optional[int] = None,
    budget_reservation_key: Optional[str] = None,
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
    from src.tools.registry import ExecutorConfig

    effective_cwd = cwd or os.getcwd()
    is_explore = subagent_type == "explore"
    parent_constraints = parent_constraints or {}
    parent_trace_id = str(parent_constraints.get("trace_id", ""))
    parent_depth = int(parent_constraints.get("depth", 0) or 0)
    max_subagent_depth = int(parent_constraints.get("max_subagent_depth", 1) or 1)
    readonly = bool(parent_constraints.get("readonly", False))
    if readonly and not is_explore:
        return ToolResult.fail("readonly parent cannot spawn execute subagent")
    if parent_depth >= max_subagent_depth:
        return ToolResult.fail("Subagent depth limit reached")

    # Use ModelRouter to select the right model tier for this subagent type
    router = ModelRouter()
    tier = router.for_subagent(subagent_type)
    effective_llm_timeout = float(llm_timeout_seconds or (120.0 if is_explore else 300.0))
    effective_tool_timeout = float(tool_timeout_seconds or (60 if is_explore else 180))
    effective_overall_timeout = int(timeout_seconds or (120 if is_explore else 300))

    # Create a fresh LLM client with the routed model
    llm = LLMClient(
        model=tier.model,
        temperature=0.0,
        max_tokens=tier.max_tokens,
        timeout=effective_llm_timeout,
        budget=budget,
        budget_reservation_key=budget_reservation_key,
    )

    # Create tool registry with explicit capability limits.
    tools = ToolRegistry(
        cwd=Path(effective_cwd),
        config=ExecutorConfig(default_timeout=effective_tool_timeout),
        allowed_tools=EXPLORE_TOOLS if is_explore else None,
        allow_subagent_spawn=False,
    )
    tools.configure_guards(
        readable_roots=parent_constraints.get("readable_roots", []),
        writable_roots=parent_constraints.get("writable_roots", []),
        readonly=readonly or is_explore,
        enabled=not bool(parent_constraints.get("bypass_sandbox", False)),
    )

    # Build subagent config
    config = {
        "max_iterations": 30 if is_explore else max_iterations,
        "max_tokens": tier.max_tokens,
        "max_output_tokens": 2000,
        "reasoning_effort": "none",
        "cache_enabled": True,
        "readonly": readonly or is_explore,
        "approval_policy": parent_constraints.get("approval_policy", "on-failure"),
        "bypass_sandbox": bool(parent_constraints.get("bypass_sandbox", False)),
        "readable_roots": parent_constraints.get("readable_roots", []),
        "writable_roots": parent_constraints.get("writable_roots", []),
        "depth": parent_depth + 1,
        "max_subagent_depth": max_subagent_depth,
        "parent_trace_id": parent_trace_id or None,
        "trace_id": f"trace_sub_{os.getpid()}_{parent_depth + 1}",
        "subagent_id": f"subagent_{os.getpid()}_{parent_depth + 1}",
        "budget": budget,
    }

    ctx = SimpleAgentContext(
        instruction=task,
        cwd=effective_cwd,
        depth=parent_depth + 1,
        max_subagent_depth=max_subagent_depth,
        trace_id=config["trace_id"],
        parent_trace_id=parent_trace_id or None,
        subagent_id=config["subagent_id"],
    )

    # Capture final agent message from the worker thread where events are emitted.
    captured: list[str] = []

    def _run_subagent_loop() -> None:
        def _capture(event: Dict[str, Any]) -> None:
            if event.get("type") == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message" and item.get("text"):
                    captured.append(item["text"])

        set_event_callback(_capture)
        try:
            run_agent_loop(llm=llm, tools=tools, ctx=ctx, config=config)
        finally:
            set_event_callback(None)

    executor = ThreadPoolExecutor(max_workers=1)
    fut = executor.submit(_run_subagent_loop)
    try:
        fut.result(timeout=effective_overall_timeout)
    except FutureTimeoutError:
        fut.cancel()
        return ToolResult.fail(
            "Subagent overall timeout reached",
            output=f"error_code=overall_timeout timeout_seconds={effective_overall_timeout}",
        )
    except Exception as e:
        text = str(e)
        if "timed out" in text.lower() or "timeout" in text.lower():
            return ToolResult.fail(f"Subagent error: {text}", output="error_code=llm_timeout")
        return ToolResult.fail(f"Subagent error: {text}", output="error_code=subagent_error")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        llm.close()

    final_msg = captured[-1] if captured else "(subagent produced no output)"
    return ToolResult.ok(final_msg)


def run_comparison(
    task: str,
    approaches: list[Dict[str, str]],
    cwd: Optional[str] = None,
    parent_constraints: Optional[Dict[str, Any]] = None,
    budget: Optional[AgentBudget] = None,
) -> ToolResult:
    """Run multiple explore subagents in parallel and compare their results.

    Args:
        task: Base task description shared by all approaches.
        approaches: List of dicts with 'label' and 'prompt' keys.
        cwd: Working directory.
        parent_constraints: Constraints from the parent agent.
        budget: Shared budget.

    Returns:
        ToolResult with all approach results formatted for comparison.
    """
    if not approaches:
        return ToolResult.fail("No approaches provided")
    if len(approaches) > 3:
        return ToolResult.fail("Maximum 3 approaches allowed")

    parent_constraints = parent_constraints or {}

    # Reserve budget for all approaches
    reservation_keys = []
    reservation_per_approach = 0.05  # explore budget
    if budget is not None:
        for i, approach in enumerate(approaches):
            key = f"compare:{i}:{os.getpid()}"
            if not budget.reserve(key, reservation_per_approach):
                # Release already reserved
                for prev_key in reservation_keys:
                    budget.release(prev_key)
                snap = budget.snapshot()
                return ToolResult.fail(
                    f"Insufficient budget for comparison ({len(approaches)} approaches)",
                    output=f"remaining_cost={snap.remaining_cost:.4f}",
                )
            reservation_keys.append(key)

    results: list[tuple[str, str]] = []  # (label, result_text)

    def _run_one(idx: int, approach: Dict[str, str]) -> tuple[str, str]:
        label = approach.get("label", f"Approach {idx + 1}")
        prompt = f"{task}\n\nApproach: {approach.get('prompt', '')}"
        r_key = reservation_keys[idx] if idx < len(reservation_keys) else None

        try:
            result = run_subagent(
                task=prompt,
                subagent_type="explore",
                cwd=cwd,
                max_iterations=30,
                parent_constraints=parent_constraints,
                budget=budget,
                timeout_seconds=120,
                budget_reservation_key=r_key,
            )
            return (label, result.output if result.success else f"[Failed: {result.error or result.output}]")
        except Exception as e:
            return (label, f"[Error: {e}]")
        finally:
            if budget is not None and r_key:
                budget.release(r_key)

    # Run all approaches in parallel
    with ThreadPoolExecutor(max_workers=min(len(approaches), MAX_CONCURRENT)) as executor:
        futures = {
            executor.submit(_run_one, i, a): i
            for i, a in enumerate(approaches)
        }
        for future in futures:
            try:
                label, text = future.result(timeout=150)
                results.append((label, text))
            except Exception as e:
                idx = futures[future]
                label = approaches[idx].get("label", f"Approach {idx + 1}")
                results.append((label, f"[Timeout/Error: {e}]"))

    # Format comparison output
    lines = [f"## Comparison Results ({len(results)} approaches)\n"]
    for label, text in results:
        lines.append(f"### {label}\n")
        # Truncate very long results
        if len(text) > 2000:
            text = text[:1997] + "..."
        lines.append(text)
        lines.append("")

    lines.append(
        "---\nReview the approaches above and decide which one to proceed with, "
        "or combine the best elements from multiple approaches."
    )

    return ToolResult.ok("\n".join(lines))
