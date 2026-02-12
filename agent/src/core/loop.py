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
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from src.core.budget import AgentBudget
from src.core.compaction import (
    manage_context,
    _TokenBudget,
)
from src.core.session import Session
from src.core.turn_runtime import TurnRuntime
from src.llm.client import ContextWindowExceeded, CostLimitExceeded, LLMError
from src.llm.router import ModelRouter
from src.output.jsonl import (
    ItemCompletedEvent,
    ItemStartedEvent,
    PlanProposedEvent,
    PlanApprovedEvent,
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
        "empty_response",
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


def _is_resource_exhausted_error(e: "LLMError") -> bool:
    """Return True if the error indicates the model's resources are exhausted.

    These errors warrant falling back to a different model rather than
    retrying the same one endlessly:
    - 429 rate-limit errors
    - Quota / capacity / resource exhaustion messages
    - Stream exhaustion after all stream-level retries
    """
    code = getattr(e, "code", "")
    msg = (getattr(e, "message", "") or str(e)).lower()

    if code in ("rate_limit", "stream_exhausted"):
        return True

    resource_keywords = (
        "rate limit",
        "rate_limit",
        "quota exceeded",
        "resource exhausted",
        "capacity",
        "too many requests",
        "throttled",
        "exhausted",
        "429",
    )
    return any(kw in msg for kw in resource_keywords)


def _is_connection_error(e: "LLMError") -> bool:
    """Return True if the error indicates a DNS or connection-level failure.

    When all models share the same API provider, a connection error for one
    model means every fallback will fail identically.  Callers should raise
    immediately instead of cycling through fallback models.

    ``chat_stream`` wraps ``ConnectError`` as ``LLMError(code="stream_exhausted")``
    but preserves the original error text, so we inspect the message too.
    """
    code = getattr(e, "code", "")
    if code == "connection_error":
        return True

    msg = (getattr(e, "message", "") or str(e)).lower()
    connection_keywords = (
        "connecterror",
        "nodename nor servname",
        "name or service not known",
        "connection refused",
        "network unreachable",
        "no route to host",
        "dns",
    )
    return any(kw in msg for kw in connection_keywords)


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
    trace_id = str(config.get("trace_id") or f"trace_{uuid.uuid4().hex[:12]}")
    parent_trace_id = config.get("parent_trace_id")
    subagent_id = config.get("subagent_id")
    depth = int(config.get("depth", 0) or 0)
    max_subagent_depth = int(config.get("max_subagent_depth", 1) or 1)
    trace_context = {
        "trace_id": trace_id,
        "parent_trace_id": parent_trace_id,
        "subagent_id": subagent_id,
        "depth": depth,
    }
    budget = config.get("budget")
    if not isinstance(budget, AgentBudget):
        budget = AgentBudget(max_cost=float(getattr(llm, "cost_limit", 10.0)))
    if hasattr(llm, "attach_budget"):
        llm.attach_budget(budget)

    # 1. Emit thread.started
    emit(
        ThreadStartedEvent(
            thread_id=session_id,
            trace_id=trace_id,
            parent_trace_id=parent_trace_id,
            subagent_id=subagent_id,
            depth=depth,
        )
    )
    emit_raw({"type": "turn.context", "session_id": session_id, **trace_context, "max_subagent_depth": max_subagent_depth})

    # 2. Emit turn.started
    emit(
        TurnStartedEvent(
            trace_id=trace_id,
            parent_trace_id=parent_trace_id,
            subagent_id=subagent_id,
            depth=depth,
        )
    )

    bypass_approvals = config.get("bypass_approvals", False)
    try:
        approval_policy = ApprovalPolicy(config.get("approval_policy", "on-failure"))
    except ValueError:
        approval_policy = ApprovalPolicy.ON_FAILURE
    readonly_mode = bool(config.get("readonly", False))
    bypass_sandbox = bool(config.get("bypass_sandbox", False))
    readable_roots = config.get("readable_roots", [])
    writable_roots = config.get("writable_roots", [])
    # Propagate runtime constraints/budget to tool handlers via context.
    setattr(
        ctx,
        "runtime_constraints",
        {
            "readonly": readonly_mode,
            "bypass_sandbox": bypass_sandbox,
            "approval_policy": approval_policy.value,
            "readable_roots": readable_roots,
            "writable_roots": writable_roots,
            "max_subagent_depth": max_subagent_depth,
            "depth": depth,
            "trace_id": trace_id,
        },
    )
    setattr(ctx, "agent_budget", budget)

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
            model=config.get("model"),
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
        if config.get("include_initial_state", False):
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
    plan_first = config.get("plan_first", False)

    # ================================================================
    # Plan-Then-Execute Phase
    # ================================================================
    if plan_first and not restored:
        _log("Plan-first mode: generating plan before execution...")
        plan_system_prompt = (
            system_prompt
            + "\n\n# PLANNING PHASE\n\n"
            "You are in PLANNING mode. Before executing anything, create a structured plan.\n"
            "Do NOT make any changes, do NOT run commands, do NOT write files.\n"
            "Only use read-only tools (read_file, list_dir, grep_files, glob_files, web_search) "
            "to understand the codebase.\n\n"
            "Output a clear, numbered plan with:\n"
            "1. Steps to accomplish the task\n"
            "2. Files that need to be modified/created\n"
            "3. Key decisions or trade-offs\n\n"
            "Format the plan clearly so the user can review it."
        )
        plan_messages: List[Dict[str, Any]] = [
            {"role": "system", "content": plan_system_prompt},
            {"role": "user", "content": ctx.instruction},
        ]
        try:
            plan_cached = _apply_caching(plan_messages, enabled=cache_enabled)
            plan_kwargs = dict(
                messages=plan_cached,
                tools=tools.get_tools_for_llm(),
                max_tokens=config.get("max_tokens", 8192),
                model=config.get("model"),
            )
            if use_streaming and hasattr(llm, "chat_stream"):
                def _on_plan_text(delta: str) -> None:
                    emit_raw({"type": "stream.text.delta", "delta": delta})
                plan_response = llm.chat_stream(**plan_kwargs, on_text=_on_plan_text)
            else:
                plan_response = llm.chat(**plan_kwargs)

            plan_text = getattr(plan_response, "text", "") or ""
            if plan_text.strip():
                _log(f"Plan generated ({len(plan_text)} chars), emitting for approval")
                emit(PlanProposedEvent(plan=plan_text))

                # Check if a plan approval callback is provided (Telegram integration)
                plan_callback = config.get("plan_approval_callback")
                if plan_callback and callable(plan_callback):
                    approved = plan_callback(plan_text)
                    if not approved:
                        _log("Plan rejected by user, aborting")
                        emit_raw({"type": "plan.rejected"})
                        ctx.done()
                        return
                    _log("Plan approved by user, proceeding to execution")
                    emit(PlanApprovedEvent(plan=plan_text))

                # Inject the plan as context for the execution phase
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"Here is my plan for this task:\n\n{plan_text}",
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Good plan. Now execute it step by step. "
                            "Use update_plan to track your progress."
                        ),
                    }
                )
        except Exception as e:
            _log(f"Plan generation failed: {e}, proceeding without plan")
            emit_raw({"type": "plan.error", "message": str(e)})

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

    verification_prompt = """<system-reminder>
Self-verification required. Re-read the original task (first user message). For each requirement:
1. Verify it was completed — run commands, check outputs, don't assume
2. If anything is missing or broken, fix it now
3. If everything checks out, confirm completion with a brief summary
Do NOT ask questions — make reasonable decisions and proceed.
</system-reminder>"""

    # 6. Main loop
    iteration = int(restored.get("iteration", 0) or 0) if restored else 0
    checkpoint_every = max(1, int(config.get("checkpoint_every", 1) or 1))
    consecutive_failures = 0
    consecutive_tool_failure_turns = 0  # Turns where ALL tool calls failed
    _last_plan_hash = ""  # Cache plan context to avoid re-injecting unchanged plans
    _recovery_msg_count = 0  # Track error recovery messages to cap accumulation
    _tool_nudge_count = 0  # Track tool-failure nudge messages to cap accumulation
    _plan_staleness_nudge_count = 0  # Track plan-staleness nudge messages
    _plan_only_nudge_count = 0  # Track plan-only turn nudge messages
    _consecutive_plan_only_turns = 0  # Track consecutive plan-only turns
    _plan_only_freebies_used = 0  # Plan-only turns forgiven (iteration not consumed)
    _max_plan_only_freebies = int(config.get("max_plan_only_freebies", 5) or 5)
    _total_plan_only_turns = 0  # Total plan-only turns observed
    _has_made_edits_ever = False  # Whether any file edit has been made in this session
    _consecutive_no_edit_turns = 0  # Consecutive turns with tool calls but no file edits
    _investigation_nudge_count = 0  # Track investigation loop nudge messages
    llm_retry_count = 0
    compaction_count = 0
    parallel_batch_count = 0
    # Track empty-response counts per model to deprioritize bad fallbacks.
    _model_empty_counts: Dict[str, int] = {}
    _MODEL_EMPTY_THRESHOLD = 3  # Skip model after this many empty responses
    approval_denials = 0
    guard_escalations = 0
    subagent_failures = 0
    subagent_rate_limit_failures = 0
    completion_reason = "max_iterations_reached"
    while iteration < max_iterations:
        iteration += 1
        _log(f"Iteration {iteration}/{max_iterations}")

        try:
            # ================================================================
            # Context Management (replaces sliding window)
            # ================================================================
            # Use per-model metadata for context limits when available
            _tier = router.select(
                messages=messages,
                iteration=iteration,
                tool_count=tool_call_count,
                is_verification=pending_completion,
            )
            context_messages = manage_context(
                messages=messages,
                system_prompt=system_prompt,
                llm=llm,
                _token_budget=token_budget,
                context_window=_tier.context_window,
                output_reserve=_tier.output_reserve,
                auto_compact_threshold=_tier.auto_compact_threshold,
            )

            request_messages = list(context_messages)
            if hasattr(tools, "format_plan_for_context"):
                plan_context = tools.format_plan_for_context()
                if plan_context:
                    import hashlib as _hl
                    _plan_hash = _hl.md5(plan_context.encode()).hexdigest()
                    if _plan_hash != _last_plan_hash:
                        _last_plan_hash = _plan_hash
                        _log(f"Plan context updated ({len(plan_context)} chars)")
                    # Always inject plan — request_messages is rebuilt each
                    # iteration so the plan would vanish if we skip this.
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
            # Call LLM with retry logic + model routing + fallback models
            # ================================================================
            max_retries = 5
            response = None
            last_error = None

            # Reuse tier from context management (avoid double selection)
            tier = _tier

            # Build the ordered list of models to try: primary + fallbacks.
            # Exclude the primary model from fallbacks to avoid duplicates.
            # Also skip models that have produced too many empty responses.
            _fallbacks = [
                m for m in (tier.fallback_models or [])
                if m != tier.model and _model_empty_counts.get(m, 0) < _MODEL_EMPTY_THRESHOLD
            ]
            models_to_try = [tier.model] + _fallbacks

            model_succeeded = False
            for _model_idx, _current_model in enumerate(models_to_try):
                if model_succeeded:
                    break

                model_exhausted = False  # set True when resource-exhausted → try next

                for attempt in range(1, max_retries + 1):
                    try:
                        call_kwargs = dict(
                            messages=cached_messages,
                            tools=tool_specs,
                            max_tokens=config.get("max_tokens", tier.max_tokens),
                            model=_current_model,
                        )

                        # Only send reasoning effort when the model supports it
                        reasoning_effort = config.get("reasoning_effort", tier.reasoning_effort)
                        if reasoning_effort and reasoning_effort != "none" and getattr(tier, "supports_reasoning", False):
                            call_kwargs["extra_body"] = {
                                "reasoning": {"effort": reasoning_effort},
                            }

                        # Encourage parallel tool calls when the model supports it.
                        # Some models (e.g. DeepSeek-V3) only emit parallel calls
                        # when this flag is explicitly set in the request.
                        if tool_specs and getattr(tier, "supports_parallel_tools", True):
                            extra = call_kwargs.get("extra_body") or {}
                            extra["parallel_tool_calls"] = True
                            call_kwargs["extra_body"] = extra

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

                        # Emit live usage so the Telegram UI can show context size
                        emit_raw({
                            "type": "stream.usage",
                            "input_tokens": total_input_tokens,
                            "output_tokens": total_output_tokens,
                            "cached_tokens": total_cached_tokens,
                        })

                        consecutive_failures = 0
                        model_succeeded = True
                        break  # Success, exit retry loop

                    except CostLimitExceeded:
                        raise  # Don't retry cost limit errors

                    except ContextWindowExceeded as e:
                        # Auto-recovery: shrink context and retry instead of
                        # wasting a generic retry attempt.
                        _log(f"Context window exceeded — shrinking history and retrying (attempt {attempt})")
                        emit_raw({
                            "type": "stream.context_recovery",
                            "attempt": attempt,
                            "message_count": len(messages),
                        })

                        # Force compaction and rebuild cached_messages
                        from src.core.compaction import manage_context as _manage_ctx
                        messages = _manage_ctx(
                            messages=messages,
                            system_prompt=system_prompt,
                            llm=llm,
                            force_compaction=True,
                            _token_budget=token_budget,
                            context_window=tier.context_window,
                            output_reserve=tier.output_reserve,
                            auto_compact_threshold=tier.auto_compact_threshold,
                        )
                        compaction_count += 1
                        # Rebuild request messages with new compacted history
                        request_messages = list(messages)
                        if hasattr(tools, "format_plan_for_context"):
                            plan_context = tools.format_plan_for_context()
                            if plan_context:
                                request_messages.append({"role": "system", "content": plan_context})
                        cached_messages = _apply_caching(request_messages, enabled=cache_enabled)
                        # Update call_kwargs for next attempt
                        if attempt < max_retries:
                            llm_retry_count += 1
                            continue
                        else:
                            raise  # Exhausted retries

                    except LLMError as e:
                        last_error = e
                        error_msg = str(e.message) if hasattr(e, "message") else str(e)
                        _log(f"LLM error (attempt {attempt}/{max_retries}, model={_current_model}): {e.code} - {error_msg}")

                        # Track empty responses per model so we can skip bad fallbacks.
                        if e.code == "empty_response":
                            _model_empty_counts[_current_model] = _model_empty_counts.get(_current_model, 0) + 1
                            if _model_empty_counts[_current_model] >= _MODEL_EMPTY_THRESHOLD:
                                _log(f"Model '{_current_model}' has {_model_empty_counts[_current_model]} empty responses — will be skipped in future fallbacks")

                        # Don't retry authentication errors
                        if e.code in ("authentication_error", "invalid_api_key"):
                            raise

                        # Connection/DNS errors affect all models (same provider) —
                        # skip fallbacks entirely so we reach the 3-failure abort fast.
                        if _is_connection_error(e):
                            raise

                        # Check if this is a resource-exhaustion error and we have
                        # more fallback models available.
                        if _is_resource_exhausted_error(e) and _model_idx + 1 < len(models_to_try):
                            next_model = models_to_try[_model_idx + 1]
                            _log(
                                f"Model '{_current_model}' resource-exhausted, "
                                f"falling back to '{next_model}' "
                                f"(fallback {_model_idx + 1}/{len(models_to_try) - 1})"
                            )
                            emit_raw({
                                "type": "stream.model_fallback",
                                "exhausted_model": _current_model,
                                "fallback_model": next_model,
                                "fallback_index": _model_idx + 1,
                                "total_fallbacks": len(models_to_try) - 1,
                                "error_code": e.code,
                                "error_message": error_msg[:200],
                            })
                            model_exhausted = True
                            break  # Break retry loop → try next model

                        # Check if it's a retryable error
                        is_retryable = _is_retryable_llm_error(e.code, error_msg)

                        if attempt < max_retries and is_retryable:
                            # Use rate-limit header info for smarter backoff
                            wait_time = _compute_retry_delay(attempt)
                            if e.rate_limit_info and e.code == "rate_limit":
                                header_wait = e.rate_limit_info.seconds_until_reset()
                                if header_wait is not None and header_wait > 0:
                                    wait_time = min(header_wait + 0.5, 60.0)  # cap at 60s
                                    _log(f"Rate-limit header suggests reset in {header_wait:.1f}s")
                            _log(f"Retrying in {wait_time:.1f} seconds...")
                            emit_raw(
                                {
                                    "type": "stream.retry",
                                    "attempt": attempt,
                                    "max_attempts": max_retries,
                                    "wait_seconds": round(wait_time, 2),
                                    "error_code": e.code,
                                    "model": _current_model,
                                }
                            )
                            llm_retry_count += 1
                            time.sleep(wait_time)
                        else:
                            # Connection/DNS errors — don't bother with fallbacks
                            if _is_connection_error(e):
                                raise

                            # Last model and retries exhausted — raise
                            if _model_idx + 1 < len(models_to_try):
                                # Still have fallbacks — try next model
                                next_model = models_to_try[_model_idx + 1]
                                _log(
                                    f"Model '{_current_model}' failed after {max_retries} retries, "
                                    f"falling back to '{next_model}'"
                                )
                                emit_raw({
                                    "type": "stream.model_fallback",
                                    "exhausted_model": _current_model,
                                    "fallback_model": next_model,
                                    "fallback_index": _model_idx + 1,
                                    "total_fallbacks": len(models_to_try) - 1,
                                    "error_code": e.code,
                                    "error_message": error_msg[:200],
                                })
                                model_exhausted = True
                                break
                            raise

                    except Exception as e:
                        last_error = e
                        error_msg = str(e)
                        _log(
                            f"Unexpected error (attempt {attempt}/{max_retries}, model={_current_model}): "
                            f"{type(e).__name__}: {error_msg}"
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
                                    "model": _current_model,
                                }
                            )
                            llm_retry_count += 1
                            time.sleep(wait_time)
                        else:
                            raise

                # If the model was resource-exhausted, continue to the next model
                if model_exhausted:
                    continue

            # All models tried without success and without raising — response
            # is still None (e.g. every model was resource-exhausted).
            if not model_succeeded or response is None:
                raise LLMError(
                    "All models exhausted without producing a response",
                    code="all_models_exhausted",
                )

        except CostLimitExceeded as e:
            _log(f"Cost limit exceeded: {e}")
            emit(
                TurnFailedEvent(
                    error={"message": f"Cost limit exceeded: {e}"},
                    trace_id=trace_id,
                    parent_trace_id=parent_trace_id,
                    subagent_id=subagent_id,
                    depth=depth,
                )
            )
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

        except ContextWindowExceeded as e:
            # Context still too large even after compaction retries — fatal
            consecutive_failures += 1
            _log(f"Context window exceeded after recovery attempts: {e}")
            emit_raw({
                "type": "stream.error",
                "stage": "llm",
                "error_code": "context_window_exceeded",
                "message": str(e),
                "consecutive_failures": consecutive_failures,
            })
            if consecutive_failures >= 3:
                emit(
                    TurnFailedEvent(
                        error={"message": str(e)},
                        trace_id=trace_id,
                        parent_trace_id=parent_trace_id,
                        subagent_id=subagent_id,
                        depth=depth,
                    )
                )
                completion_reason = "context_window_fatal"
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
            # Try again on next iteration (compaction already happened)
            continue

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
                emit(
                    TurnFailedEvent(
                        error={"message": str(e)},
                        trace_id=trace_id,
                        parent_trace_id=parent_trace_id,
                        subagent_id=subagent_id,
                        depth=depth,
                    )
                )
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
            # Cap recovery messages to prevent context bloat (max 2)
            if _recovery_msg_count >= 2:
                # Remove the oldest recovery message
                for ri, rm in enumerate(messages):
                    if rm.get("role") == "user" and "System note:" in rm.get("content", ""):
                        messages.pop(ri)
                        _recovery_msg_count -= 1
                        break
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"System note: previous LLM request failed with {e.code}. "
                        "Recover and continue the task from the latest available context."
                    ),
                }
            )
            _recovery_msg_count += 1
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
                emit(
                    TurnFailedEvent(
                        error={"message": str(e)},
                        trace_id=trace_id,
                        parent_trace_id=parent_trace_id,
                        subagent_id=subagent_id,
                        depth=depth,
                    )
                )
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
            # Cap recovery messages to prevent context bloat (max 2)
            if _recovery_msg_count >= 2:
                for ri, rm in enumerate(messages):
                    if rm.get("role") == "user" and "System note:" in rm.get("content", ""):
                        messages.pop(ri)
                        _recovery_msg_count -= 1
                        break
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "System note: a transient runtime error occurred. "
                        "Continue with the task using the latest valid state."
                    ),
                }
            )
            _recovery_msg_count += 1
            continue

        runtime = TurnRuntime(
            cwd=Path(ctx.cwd),
            tools=tools,
            approval_policy=approval_policy,
            bypass_approvals=bypass_approvals,
            approval_cache=session.approval_cache,
            max_output_tokens=max_output_tokens,
            readonly=readonly_mode,
            trace_context=trace_context,
            budget=budget,
            tool_output_max_tokens=tier.tool_output_max_tokens,
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
        subagent_failures += runtime_result.subagent_failures_delta
        subagent_rate_limit_failures += runtime_result.subagent_rate_limit_failures_delta

        # ================================================================
        # Plan iteration stamping & staleness nudging
        # ================================================================
        if hasattr(tools, "stamp_plan_iteration"):
            tools.stamp_plan_iteration(iteration)

        _plan_stale_threshold = int(config.get("plan_stale_threshold", 10) or 10)
        if (
            hasattr(tools, "plan_last_updated_iteration")
            and hasattr(tools, "format_plan_for_context")
            and tools.format_plan_for_context()  # plan exists
            and tools.plan_last_updated_iteration > 0  # was set at least once
            and (iteration - tools.plan_last_updated_iteration) >= _plan_stale_threshold
            and (iteration - tools.plan_last_updated_iteration) % _plan_stale_threshold == 0
        ):
            _log(
                f"Plan stale for {iteration - tools.plan_last_updated_iteration} iterations "
                f"(last updated at iteration {tools.plan_last_updated_iteration})"
            )
            emit_raw({
                "type": "stream.plan_stale",
                "iterations_since_update": iteration - tools.plan_last_updated_iteration,
                "last_updated_iteration": tools.plan_last_updated_iteration,
            })
            # Cap nudge messages to prevent context bloat (max 1 active)
            if _plan_staleness_nudge_count >= 1:
                for ri, rm in enumerate(messages):
                    if rm.get("role") == "user" and "plan has not been updated" in rm.get("content", ""):
                        messages.pop(ri)
                        _plan_staleness_nudge_count -= 1
                        break
            stale_nudge = (
                "<system-reminder>\n"
                f"Your execution plan has not been updated for {iteration - tools.plan_last_updated_iteration} iterations. "
                "Please review your plan against the original task:\n"
                "1. Are the remaining steps still correct?\n"
                "2. Should any steps be marked completed, modified, or added?\n"
                "3. Are you still making progress toward the goal?\n"
                "Use `update_plan` to refresh your plan, even if just to confirm it's still accurate.\n"
                "</system-reminder>"
            )
            messages.append({"role": "user", "content": stale_nudge})
            _plan_staleness_nudge_count += 1

        # ================================================================
        # Plan-only turn detection
        # ================================================================
        # The TurnRuntime now detects plan-only turns early (before full
        # tool execution) and injects a nudge directly into the tool result.
        # We only need to handle the bookkeeping: iteration forgiveness,
        # logging, and consecutive-turn tracking.
        if runtime_result.plan_only_turn:
            _consecutive_plan_only_turns += 1
            _log(
                f"Plan-only turn detected (consecutive: {_consecutive_plan_only_turns}) "
                "— no productive tool calls this iteration"
            )
            emit_raw({
                "type": "stream.plan_only_turn",
                "consecutive_plan_only_turns": _consecutive_plan_only_turns,
            })
            _total_plan_only_turns += 1
            # Don't count plan-only turns against the iteration budget (up to cap)
            if _plan_only_freebies_used < _max_plan_only_freebies:
                iteration -= 1
                _plan_only_freebies_used += 1
                _log(f"Plan-only turn forgiven ({_plan_only_freebies_used}/{_max_plan_only_freebies} freebies used)")
        else:
            _consecutive_plan_only_turns = 0

        # ================================================================
        # Investigation loop detection
        # ================================================================
        # Track consecutive turns where the agent made tool calls but
        # didn't edit any files (write_file, str_replace, hashline_edit,
        # apply_patch).  This catches debugging/investigation spirals
        # where the agent keeps running diagnostic commands without
        # making progress toward the goal.
        if runtime_result.made_edits:
            _has_made_edits_ever = True
            _consecutive_no_edit_turns = 0
        elif (
            _has_made_edits_ever
            and runtime_result.tool_call_count_delta > 0
            and not runtime_result.plan_only_turn
        ):
            _consecutive_no_edit_turns += 1

        # ================================================================
        # Auto-arm completion after self-verification
        # ================================================================
        # When the agent has made file edits AND just ran a successful
        # shell command, it has already verified its own work.  Pre-arm
        # pending_completion so the next text-only response completes
        # immediately — skipping the redundant verification prompt that
        # would otherwise cost an extra LLM round-trip.
        if (
            _has_made_edits_ever
            and runtime_result.ran_successful_shell
            and not pending_completion
        ):
            pending_completion = True
            _log("Auto-armed completion: edits made + successful shell verification")

        _investigation_loop_threshold = int(config.get("investigation_loop_threshold", 3) or 3)
        if (
            _consecutive_no_edit_turns > 0
            and _consecutive_no_edit_turns >= _investigation_loop_threshold
            and _consecutive_no_edit_turns % _investigation_loop_threshold == 0
        ):
            _log(
                f"Investigation loop: {_consecutive_no_edit_turns} consecutive turns "
                "without file edits after initial editing phase"
            )
            emit_raw({
                "type": "stream.investigation_loop",
                "consecutive_no_edit_turns": _consecutive_no_edit_turns,
            })
            # Cap nudge messages to prevent context bloat (max 1 active)
            if _investigation_nudge_count >= 1:
                for ri, rm in enumerate(messages):
                    if (
                        rm.get("role") == "user"
                        and "without editing" in rm.get("content", "")
                        and "investigation" in rm.get("content", "").lower()
                    ):
                        messages.pop(ri)
                        _investigation_nudge_count -= 1
                        break
            investigation_nudge = (
                "<system-reminder>\n"
                f"WARNING: You have spent {_consecutive_no_edit_turns} consecutive turns "
                "running commands and reading files without editing any files. "
                "You are stuck in a debugging/investigation loop.\n\n"
                "STOP investigating. Take ONE of these actions NOW:\n"
                "1. If you know what to fix — apply the fix (write_file, str_replace, "
                "or hashline_edit) and re-verify once\n"
                "2. If the output is close enough (e.g. minor floating-point precision "
                "differences) — the task is DONE, stop with no tool calls\n"
                "3. If you truly cannot fix it — report what you found and stop\n\n"
                "Your next response MUST include a file edit or be a completion "
                "statement with no tool calls.\n"
                "</system-reminder>"
            )
            messages.append({"role": "user", "content": investigation_nudge})
            _investigation_nudge_count += 1

        # ================================================================
        # Consecutive tool failure loop detection
        # ================================================================
        # If the turn had tool calls and ALL of them failed, count it as a
        # tool-failure turn.  Any success resets the counter.
        _turn_tool_total = runtime_result.tool_successes_delta + runtime_result.tool_failures_delta
        if _turn_tool_total > 0:
            if runtime_result.tool_successes_delta == 0:
                consecutive_tool_failure_turns += 1
                _log(
                    f"All {runtime_result.tool_failures_delta} tool call(s) failed this turn "
                    f"(consecutive tool-failure turns: {consecutive_tool_failure_turns})"
                )
            else:
                consecutive_tool_failure_turns = 0

        # Hard abort: too many consecutive tool-failure turns
        _max_tool_failure_turns = int(config.get("max_consecutive_tool_failure_turns", 8) or 8)
        if consecutive_tool_failure_turns >= _max_tool_failure_turns:
            _log(f"Tool failure loop detected: {consecutive_tool_failure_turns} consecutive turns with all tools failing — aborting")
            emit_raw({
                "type": "stream.tool_failure_loop",
                "consecutive_tool_failure_turns": consecutive_tool_failure_turns,
                "action": "abort",
            })
            emit(
                TurnFailedEvent(
                    error={
                        "message": (
                            f"Tool failure loop: {consecutive_tool_failure_turns} consecutive turns "
                            "where every tool call failed. Aborting to prevent wasted iterations."
                        )
                    },
                    trace_id=trace_id,
                    parent_trace_id=parent_trace_id,
                    subagent_id=subagent_id,
                    depth=depth,
                )
            )
            completion_reason = "tool_failure_loop"
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

        # Soft nudge: inject a recovery message to redirect the model
        _tool_nudge_threshold = int(config.get("tool_failure_nudge_threshold", 3) or 3)
        if (
            consecutive_tool_failure_turns > 0
            and consecutive_tool_failure_turns % _tool_nudge_threshold == 0
            and consecutive_tool_failure_turns < _max_tool_failure_turns
        ):
            _log(f"Injecting tool-failure recovery nudge (consecutive={consecutive_tool_failure_turns})")
            emit_raw({
                "type": "stream.tool_failure_loop",
                "consecutive_tool_failure_turns": consecutive_tool_failure_turns,
                "action": "nudge",
            })
            # Cap nudge messages to prevent context bloat (max 2)
            if _tool_nudge_count >= 2:
                for ri, rm in enumerate(messages):
                    if rm.get("role") == "user" and "tool calls have failed" in rm.get("content", ""):
                        messages.pop(ri)
                        _tool_nudge_count -= 1
                        break
            nudge = (
                "<system-reminder>\n"
                f"WARNING: Your last {consecutive_tool_failure_turns} turns of tool calls ALL failed. "
                "You appear to be stuck in a failure loop. STOP and reassess.\n\n"
                "1. Re-read the original task (first user message)\n"
                "2. Review the recent errors — what pattern do you see?\n"
                "3. Try a DIFFERENT approach:\n"
                "   - If file paths are wrong, use list_dir to discover the correct structure\n"
                "   - If commands fail, use a simpler alternative\n"
                "   - If edits fail, re-read the file first to get current contents\n"
                "   - If you're stuck, break the problem into a smaller first step\n"
                "4. Do NOT retry the same failing operation\n"
                "</system-reminder>"
            )
            messages.append({"role": "user", "content": nudge})
            _tool_nudge_count += 1

        # Circuit breaker for repeated subagent instability.
        max_subagent_failures = int(config.get("max_subagent_failures", 3) or 3)
        max_subagent_rate_limits = int(config.get("max_subagent_rate_limits", 2) or 2)
        if subagent_failures >= max_subagent_failures or subagent_rate_limit_failures >= max_subagent_rate_limits:
            emit(
                TurnFailedEvent(
                    error={
                        "message": (
                            "Subagent circuit breaker tripped: "
                            f"failures={subagent_failures}, rate_limits={subagent_rate_limit_failures}"
                        )
                    },
                    trace_id=trace_id,
                    parent_trace_id=parent_trace_id,
                    subagent_id=subagent_id,
                    depth=depth,
                )
            )
            completion_reason = "subagent_circuit_breaker"
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
            emit(
                TurnFailedEvent(
                    error={"message": runtime_result.abort_reason},
                    trace_id=trace_id,
                    parent_trace_id=parent_trace_id,
                    subagent_id=subagent_id,
                    depth=depth,
                )
            )
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
            },
            trace_id=trace_id,
            parent_trace_id=parent_trace_id,
            subagent_id=subagent_id,
            depth=depth,
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
            **trace_context,
            "iterations": iteration,
            "llm_retries": llm_retry_count,
            "compactions": compaction_count,
            "parallel_batches": parallel_batch_count,
            "approval_denials": approval_denials,
            "guard_escalations": guard_escalations,
            "subagent_failures": subagent_failures,
            "subagent_rate_limit_failures": subagent_rate_limit_failures,
            "consecutive_tool_failure_turns": consecutive_tool_failure_turns,
            "plan_only_turns": _total_plan_only_turns,
            "plan_only_freebies_used": _plan_only_freebies_used,
            "completion_reason": completion_reason,
            "budget": {
                "max_cost": budget.snapshot().max_cost,
                "consumed_cost": round(budget.snapshot().consumed_cost, 6),
                "reserved_cost": round(budget.snapshot().reserved_cost, 6),
                "remaining_cost": round(budget.snapshot().remaining_cost, 6),
            },
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
