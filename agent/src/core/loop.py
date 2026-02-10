"""
Main agent loop - the heart of the SuperAgent system.

Implements the agentic loop that:
1. Receives instruction via --instruction argument
2. Calls LLM with tools (using Chutes API)
3. Executes tool calls
4. Loops until task is complete
5. Emits JSONL events throughout

Context management strategy:
- Token-based overflow detection (not message count)
- Tool output pruning (clear old outputs first)
- AI compaction when needed (summarize conversation)
- Stable system prompt for cache hits
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from src.core.compaction import (
    manage_context,
    _TokenBudget,
)
from src.core.session import Session
from src.core.turn_runtime import TurnRuntime
from src.llm.client import CostLimitExceeded, LLMError
from src.llm.router import ModelRouter
from src.output.jsonl import (
    ItemCompletedEvent,
    ItemStartedEvent,
    ThreadStartedEvent,
    TurnCompletedEvent,
    TurnFailedEvent,
    TurnStartedEvent,
    emit,
    make_agent_message_item,
    make_command_execution_item,
    next_item_id,
    reset_item_counter,
    emit_raw,
)
from src.prompts.system import get_system_prompt
from src.tools.policy import ApprovalPolicy
from src.utils.truncate import middle_out_truncate

if TYPE_CHECKING:
    from src.llm.client import LLMClient
    from src.tools.registry import ToolRegistry


def _log(msg: str) -> None:
    """Log to stderr."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [loop] {msg}", file=sys.stderr, flush=True)


def _add_cache_control_to_message(
    msg: Dict[str, Any],
    cache_control: Dict[str, str],
) -> Dict[str, Any]:
    """Add cache_control to a message, converting to multipart if needed."""
    content = msg.get("content")

    if isinstance(content, list):
        has_cache = any(isinstance(p, dict) and "cache_control" in p for p in content)
        if has_cache:
            return msg

        new_content = list(content)
        for i in range(len(new_content) - 1, -1, -1):
            part = new_content[i]
            if isinstance(part, dict) and part.get("type") == "text":
                new_content[i] = {**part, "cache_control": cache_control}
                break
        return {**msg, "content": new_content}

    if isinstance(content, str):
        return {
            **msg,
            "content": [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": cache_control,
                }
            ],
        }

    return msg


def _apply_caching(
    messages: List[Dict[str, Any]],
    enabled: bool = True,
) -> List[Dict[str, Any]]:
    """
    Apply prompt caching like OpenCode does:
    - Cache first 2 system messages (stable prefix)
    - Cache last 2 non-system messages (extends cache to cover conversation history)

    How Anthropic caching works:
    - Cache is based on IDENTICAL PREFIX
    - A cache_control breakpoint tells Anthropic to cache everything BEFORE it
    - By marking the last messages, we cache the entire conversation history
    - Each new request only adds new messages after the cached prefix

    Anthropic limits:
    - Maximum 4 cache_control breakpoints
    - Minimum tokens per breakpoint: 1024 (Sonnet), 4096 (Opus 4.5 on Bedrock)

    Reference: OpenCode transform.ts applyCaching()
    """
    if not enabled or not messages:
        return messages

    cache_control = {"type": "ephemeral"}

    # Separate system and non-system message indices
    system_indices = []
    non_system_indices = []

    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            system_indices.append(i)
        else:
            non_system_indices.append(i)

    # Determine which messages to cache:
    # 1. First 2 system messages (stable system prompt)
    # 2. Last 2 non-system messages (extends cache to conversation history)
    # Total: up to 4 breakpoints (Anthropic limit)
    indices_to_cache = set()

    # Add first 2 system messages
    for idx in system_indices[:2]:
        indices_to_cache.add(idx)

    # Add last 2 non-system messages
    for idx in non_system_indices[-2:]:
        indices_to_cache.add(idx)

    # Build result with cache_control added to selected messages
    result = []
    for i, msg in enumerate(messages):
        if i in indices_to_cache:
            result.append(_add_cache_control_to_message(msg, cache_control))
        else:
            result.append(msg)

    cached_system = len([i for i in indices_to_cache if i in system_indices])
    cached_final = len([i for i in indices_to_cache if i in non_system_indices])

    if indices_to_cache:
        _log(
            f"Prompt caching: {cached_system} system + {cached_final} final messages marked ({len(indices_to_cache)} breakpoints)"
        )

    return result


def _is_retryable_llm_error(code: str, message: str) -> bool:
    retryable_codes = {
        "timeout",
        "rate_limit",
        "server_error",
        "connection_error",
        "api_error",
    }
    if code in retryable_codes:
        return True
    lowered = message.lower()
    return any(
        needle in lowered
        for needle in (
            "timeout",
            "timed out",
            "overloaded",
            "503",
            "502",
            "500",
            "504",
            "rate_limit",
            "temporarily unavailable",
            "connection reset",
            "empty response",
        )
    )


def _compute_retry_delay(attempt: int, base_delay: float = 1.5, max_delay: float = 20.0) -> float:
    delay = min(max_delay, base_delay * (2 ** max(0, attempt - 1)))
    jitter = random.uniform(0.0, min(1.0, delay * 0.25))
    return delay + jitter


def run_agent_loop(
    llm: "LLMClient",
    tools: "ToolRegistry",
    ctx: Any,
    config: Dict[str, Any],
    system_prompt: str | None = None,
) -> None:
    """
    Run the main agent loop.

    Args:
        llm: LLM client
        tools: Tool registry with available tools
        ctx: Agent context with instruction, shell(), done()
        config: Configuration dictionary
        system_prompt: Optional system prompt override. When provided, this
            replaces the default system prompt entirely, eliminating the
            need for monkey-patching ``get_system_prompt``.
    """
    # Reset item counter for fresh session
    reset_item_counter()

    cwd = Path(ctx.cwd)
    session = Session(cwd=cwd)
    resume_session_id = config.get("resume_session_id")
    resume_latest = bool(config.get("resume_latest", False))
    restored = Session.load_rollout(
        cwd,
        session_id=resume_session_id,
        resume_latest=resume_latest,
    )
    session_id = f"sess_{int(time.time() * 1000)}"
    if restored and restored.get("session_id"):
        session_id = str(restored["session_id"])
    session.id = session_id

    # 1. Emit thread.started
    emit(ThreadStartedEvent(thread_id=session_id))

    # 2. Emit turn.started
    emit(TurnStartedEvent())

    bypass_approvals = config.get("bypass_approvals", False)
    try:
        approval_policy = ApprovalPolicy(config.get("approval_policy", "on-failure"))
    except ValueError:
        approval_policy = ApprovalPolicy.ON_FAILURE
    readonly_mode = bool(config.get("readonly", False))
    bypass_sandbox = bool(config.get("bypass_sandbox", False))
    readable_roots = config.get("readable_roots", [])
    writable_roots = config.get("writable_roots", [])

    # 3. Build initial messages
    if system_prompt is None:
        system_prompt = get_system_prompt(
            cwd=cwd,
            shell=config.get("shell"),
            environment_context={
                "approval_policy": approval_policy.value,
                "sandbox_mode": "bypass" if bypass_sandbox else "guarded",
                "network_access": config.get("network_access", "unknown"),
                "readonly": readonly_mode,
                "readable_roots": readable_roots,
                "writable_roots": writable_roots,
            },
        )

    # 4. Build fresh or restored state
    max_output_tokens = config.get("max_output_tokens", 2500)
    messages: List[Dict[str, Any]]
    total_input_tokens = 0
    total_output_tokens = 0
    total_cached_tokens = 0
    pending_completion = False
    last_agent_message = ""
    tool_call_count = 0
    if restored and isinstance(restored.get("messages"), list):
        messages = restored.get("messages", [])
        usage = restored.get("usage", {})
        total_input_tokens = int(usage.get("input_tokens", 0) or 0)
        total_output_tokens = int(usage.get("output_tokens", 0) or 0)
        total_cached_tokens = int(usage.get("cached_input_tokens", 0) or 0)
        pending_completion = bool(restored.get("pending_completion", False))
        tool_call_count = int(restored.get("tool_call_count", 0) or 0)
        approval_cache = restored.get("approval_cache", {})
        if isinstance(approval_cache, dict):
            session.approval_cache.update({str(k): bool(v) for k, v in approval_cache.items()})
        _log(f"Resumed session {session_id} with {len(messages)} messages")
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ctx.instruction},
        ]
        if not config.get("skip_initial_state", False):
            _log("Getting initial state...")
            initial_result = ctx.shell("pwd && ls -la")
            initial_state = middle_out_truncate(initial_result.output, max_tokens=max_output_tokens)
            messages.append(
                {
                    "role": "user",
                    "content": f"Current directory and files:\n```\n{initial_state}\n```",
                }
            )

    max_iterations = config.get("max_iterations", 200)
    cache_enabled = config.get("cache_enabled", True)
    use_streaming = config.get("streaming", True)
    skip_verification = config.get("skip_verification", False)

    # Configure root/path guards for first sandboxed attempts.
    tools.configure_guards(
        readable_roots=readable_roots,
        writable_roots=writable_roots,
        readonly=readonly_mode,
        enabled=not bypass_sandbox,
    )

    # Model router for dynamic model selection
    router = ModelRouter()

    # Incremental token budget — avoids O(n) recount every iteration
    token_budget = _TokenBudget()

    verification_prompt = f"""<system-reminder>
# Self-Verification Required - CRITICAL

You indicated the task might be complete. Before finishing, you MUST perform a thorough self-verification.

## Original Task (re-read carefully):
{ctx.instruction}

## Self-Verification Checklist:

### 1. Requirements Analysis
- Re-read the ENTIRE original task above word by word
- List EVERY requirement, constraint, and expected outcome mentioned
- Check if there are any implicit requirements you might have missed

### 2. Work Verification
- For EACH requirement identified, verify it was completed:
  - Run commands to check file contents, test outputs, or verify state
  - Do NOT assume something works - actually verify it
  - If you created code, run it to confirm it works
  - If you modified files, read them back to confirm changes are correct

### 3. Edge Cases & Quality
- Are there any edge cases the task mentioned that you haven't handled?
- Did you follow any specific format/style requirements mentioned?
- Are there any errors, warnings, or issues in your implementation?

### 4. Final Decision
After completing the above verification:
- If EVERYTHING is verified and correct: Summarize what was done and confirm completion
- If ANYTHING is missing or broken: Fix it now using the appropriate tools

## CRITICAL REMINDERS:
- You are running in HEADLESS mode - DO NOT ask questions to the user
- DO NOT ask for confirmation or clarification - make reasonable decisions
- If something is ambiguous, make the most reasonable choice and proceed
- If you find issues during verification, FIX THEM before completing
- Only complete if you have VERIFIED (not assumed) that everything works

Proceed with verification now.
</system-reminder>"""

    # 6. Main loop
    iteration = int(restored.get("iteration", 0) or 0) if restored else 0
    checkpoint_every = max(1, int(config.get("checkpoint_every", 1) or 1))
    consecutive_failures = 0
    llm_retry_count = 0
    compaction_count = 0
    parallel_batch_count = 0
    approval_denials = 0
    guard_escalations = 0
    completion_reason = "max_iterations_reached"
    while iteration < max_iterations:
        iteration += 1
        _log(f"Iteration {iteration}/{max_iterations}")

        try:
            # ================================================================
            # Context Management (replaces sliding window)
            # ================================================================
            # Check token usage and apply pruning/compaction if needed
            context_messages = manage_context(
                messages=messages,
                system_prompt=system_prompt,
                llm=llm,
                _token_budget=token_budget,
            )

            request_messages = list(context_messages)
            if hasattr(tools, "format_plan_for_context"):
                plan_context = tools.format_plan_for_context()
                if plan_context:
                    request_messages.append(
                        {
                            "role": "system",
                            "content": plan_context,
                        }
                    )

            # If compaction happened, update our messages reference
            if len(context_messages) < len(messages):
                _log(f"Context compacted: {len(messages)} -> {len(context_messages)} messages")
                messages = context_messages
                compaction_count += 1

            # ================================================================
            # Apply caching (system prompt only for stability)
            # ================================================================
            cached_messages = _apply_caching(request_messages, enabled=cache_enabled)

            # Get tool specs
            tool_specs = tools.get_tools_for_llm()

            # ================================================================
            # Call LLM with retry logic + model routing
            # ================================================================
            max_retries = 5
            response = None
            last_error = None

            # Select model tier via router
            is_verification = pending_completion
            tier = router.select(
                messages=cached_messages,
                iteration=iteration,
                tool_count=tool_call_count,
                is_verification=is_verification,
            )

            for attempt in range(1, max_retries + 1):
                try:
                    call_kwargs = dict(
                        messages=cached_messages,
                        tools=tool_specs,
                        max_tokens=config.get("max_tokens", tier.max_tokens),
                        model=tier.model,
                        extra_body={
                            "reasoning": {"effort": config.get("reasoning_effort", tier.reasoning_effort)},
                        },
                    )

                    # Use streaming if available and enabled
                    if use_streaming and hasattr(llm, "chat_stream"):
                        def _on_stream_text(delta: str) -> None:
                            emit_raw({"type": "stream.text.delta", "delta": delta})

                        response = llm.chat_stream(**call_kwargs, on_text=_on_stream_text)
                    else:
                        response = llm.chat(**call_kwargs)

                    # Track token usage from response
                    if hasattr(response, "tokens") and response.tokens:
                        tokens = response.tokens
                        if isinstance(tokens, dict):
                            total_input_tokens += tokens.get("input", 0)
                            total_output_tokens += tokens.get("output", 0)
                            total_cached_tokens += tokens.get("cached", 0)

                    consecutive_failures = 0
                    break  # Success, exit retry loop

                except CostLimitExceeded:
                    raise  # Don't retry cost limit errors

                except LLMError as e:
                    last_error = e
                    error_msg = str(e.message) if hasattr(e, "message") else str(e)
                    _log(f"LLM error (attempt {attempt}/{max_retries}): {e.code} - {error_msg}")

                    # Don't retry authentication errors
                    if e.code in ("authentication_error", "invalid_api_key"):
                        raise

                    # Check if it's a retryable error
                    is_retryable = _is_retryable_llm_error(e.code, error_msg)

                    if attempt < max_retries and is_retryable:
                        wait_time = _compute_retry_delay(attempt)
                        _log(f"Retrying in {wait_time:.1f} seconds...")
                        emit_raw(
                            {
                                "type": "stream.retry",
                                "attempt": attempt,
                                "max_attempts": max_retries,
                                "wait_seconds": round(wait_time, 2),
                                "error_code": e.code,
                            }
                        )
                        llm_retry_count += 1
                        time.sleep(wait_time)
                    else:
                        raise

                except Exception as e:
                    last_error = e
                    error_msg = str(e)
                    _log(
                        f"Unexpected error (attempt {attempt}/{max_retries}): {type(e).__name__}: {error_msg}"
                    )

                    is_retryable = _is_retryable_llm_error("unexpected", error_msg)

                    if attempt < max_retries and is_retryable:
                        wait_time = _compute_retry_delay(attempt)
                        _log(f"Retrying in {wait_time:.1f} seconds...")
                        emit_raw(
                            {
                                "type": "stream.retry",
                                "attempt": attempt,
                                "max_attempts": max_retries,
                                "wait_seconds": round(wait_time, 2),
                                "error_code": "unexpected",
                            }
                        )
                        llm_retry_count += 1
                        time.sleep(wait_time)
                    else:
                        raise

        except CostLimitExceeded as e:
            _log(f"Cost limit exceeded: {e}")
            emit(TurnFailedEvent(error={"message": f"Cost limit exceeded: {e}"}))
            completion_reason = "cost_limit_exceeded"
            session.save_rollout(
                messages=messages,
                iteration=iteration,
                pending_completion=pending_completion,
                tool_call_count=tool_call_count,
                usage={
                    "input_tokens": total_input_tokens,
                    "cached_input_tokens": total_cached_tokens,
                    "output_tokens": total_output_tokens,
                },
            )
            ctx.done()
            return

        except LLMError as e:
            consecutive_failures += 1
            _log(f"LLM error: {e.code} - {e.message}")
            emit_raw(
                {
                    "type": "stream.error",
                    "stage": "llm",
                    "error_code": e.code,
                    "message": str(e),
                    "consecutive_failures": consecutive_failures,
                }
            )
            if e.code in ("authentication_error", "invalid_api_key") or consecutive_failures >= 3:
                emit(TurnFailedEvent(error={"message": str(e)}))
                completion_reason = "llm_fatal_error"
                session.save_rollout(
                    messages=messages,
                    iteration=iteration,
                    pending_completion=pending_completion,
                    tool_call_count=tool_call_count,
                    usage={
                        "input_tokens": total_input_tokens,
                        "cached_input_tokens": total_cached_tokens,
                        "output_tokens": total_output_tokens,
                    },
                )
                ctx.done()
                return
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"System note: previous LLM request failed with {e.code}. "
                        "Recover and continue the task from the latest available context."
                    ),
                }
            )
            continue

        except Exception as e:
            consecutive_failures += 1
            _log(f"Unexpected error: {type(e).__name__}: {e}")
            emit_raw(
                {
                    "type": "stream.error",
                    "stage": "loop",
                    "error_code": type(e).__name__,
                    "message": str(e),
                    "consecutive_failures": consecutive_failures,
                }
            )
            if consecutive_failures >= 3:
                emit(TurnFailedEvent(error={"message": str(e)}))
                completion_reason = "runtime_fatal_error"
                session.save_rollout(
                    messages=messages,
                    iteration=iteration,
                    pending_completion=pending_completion,
                    tool_call_count=tool_call_count,
                    usage={
                        "input_tokens": total_input_tokens,
                        "cached_input_tokens": total_cached_tokens,
                        "output_tokens": total_output_tokens,
                    },
                )
                ctx.done()
                return
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "System note: a transient runtime error occurred. "
                        "Continue with the task using the latest valid state."
                    ),
                }
            )
            continue

        runtime = TurnRuntime(
            cwd=Path(ctx.cwd),
            tools=tools,
            approval_policy=approval_policy,
            bypass_approvals=bypass_approvals,
            approval_cache=session.approval_cache,
            max_output_tokens=max_output_tokens,
            readonly=readonly_mode,
        )
        runtime_result = runtime.process_response(
            response=response,
            messages=messages,
            ctx=ctx,
            pending_completion=pending_completion,
            skip_verification=skip_verification,
            verification_prompt=verification_prompt,
        )
        messages = runtime_result.messages
        pending_completion = runtime_result.pending_completion
        last_agent_message = runtime_result.last_agent_message
        tool_call_count += runtime_result.tool_call_count_delta
        parallel_batch_count += runtime_result.parallel_batch_count_delta
        approval_denials += runtime_result.approval_denials_delta
        guard_escalations += runtime_result.guard_escalations_delta
        if iteration % checkpoint_every == 0:
            session.save_rollout(
                messages=messages,
                iteration=iteration,
                pending_completion=pending_completion,
                tool_call_count=tool_call_count,
                usage={
                    "input_tokens": total_input_tokens,
                    "cached_input_tokens": total_cached_tokens,
                    "output_tokens": total_output_tokens,
                },
            )
        if runtime_result.aborted:
            emit(TurnFailedEvent(error={"message": runtime_result.abort_reason}))
            completion_reason = "tool_abort"
            session.save_rollout(
                messages=messages,
                iteration=iteration,
                pending_completion=pending_completion,
                tool_call_count=tool_call_count,
                usage={
                    "input_tokens": total_input_tokens,
                    "cached_input_tokens": total_cached_tokens,
                    "output_tokens": total_output_tokens,
                },
            )
            ctx.done()
            return
        if runtime_result.completed:
            _log("Task completion confirmed")
            completion_reason = "completed"
            break

    # 7. Emit turn.completed
    emit(
        TurnCompletedEvent(
            usage={
                "input_tokens": total_input_tokens,
                "cached_input_tokens": total_cached_tokens,
                "output_tokens": total_output_tokens,
            }
        )
    )

    _log(f"Loop complete after {iteration} iterations")
    _log(
        f"Tokens: {total_input_tokens} input, {total_cached_tokens} cached, {total_output_tokens} output"
    )
    emit_raw(
        {
            "type": "turn.metrics",
            "session_id": session_id,
            "iterations": iteration,
            "llm_retries": llm_retry_count,
            "compactions": compaction_count,
            "parallel_batches": parallel_batch_count,
            "approval_denials": approval_denials,
            "guard_escalations": guard_escalations,
            "completion_reason": completion_reason,
        }
    )
    session.save_rollout(
        messages=messages,
        iteration=iteration,
        pending_completion=pending_completion,
        tool_call_count=tool_call_count,
        usage={
            "input_tokens": total_input_tokens,
            "cached_input_tokens": total_cached_tokens,
            "output_tokens": total_output_tokens,
        },
    )
    ctx.done()
