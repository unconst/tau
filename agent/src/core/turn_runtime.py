"""Turn runtime for response processing and tool orchestration.

Includes execution-time tool output truncation — large outputs are
truncated *before* entering the message history to prevent token waste.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.output.jsonl import (
    ItemCompletedEvent,
    ItemStartedEvent,
    emit,
    emit_raw,
    make_agent_message_item,
    make_command_execution_item,
    next_item_id,
)
from src.tools.orchestrator import ToolOrchestrator
from src.tools.base import ToolResult
from src.tools.policy import (
    ApprovalOutcome,
    ApprovalPolicy,
    PolicyDecisionKind,
    ToolPolicyEngine,
)
from src.tools.router import ToolRouter
from src.tools.specs import tool_is_mutating, tool_supports_parallel
from src.utils.truncate import middle_out_truncate


# ---------------------------------------------------------------------------
# Per-tool output limits (lines).  These apply at execution time, *before*
# the output enters the conversation history, so the model never sees the
# full bloat.  They are intentionally generous — they trim pathological cases
# (e.g. `cat huge_log`) while keeping normal outputs intact.
# ---------------------------------------------------------------------------

_TOOL_MAX_LINES: Dict[str, int] = {
    "shell_command": 100,   # keep first 10 + last 90
    "read_file": 250,       # already limited by offset/limit, but guard
    "list_dir": 80,
    "grep_files": 80,
    "glob_files": 80,
    "web_search": 60,
    "spawn_subagent": 150,
}

# Absolute byte limit before we truncate (32 KB) — catches binary blobs,
# accidental log dumps, etc.
_TOOL_MAX_BYTES = 32_768


def _truncate_tool_output(
    output: str,
    tool_name: str,
    max_output_tokens: int,
) -> str:
    """Apply execution-time truncation to a tool output.

    Two stages:
    1. Line-based truncation per tool type (cheap, no token counting).
    2. Token-based middle-out truncation (existing ``middle_out_truncate``).
    """
    if not output:
        return output

    # Stage 0: absolute byte guard
    if len(output) > _TOOL_MAX_BYTES:
        # Keep head + tail bytes
        half = _TOOL_MAX_BYTES // 2
        output = (
            output[:half]
            + f"\n\n[... truncated {len(output) - _TOOL_MAX_BYTES} bytes ...]\n\n"
            + output[-half:]
        )

    # Stage 1: line-based truncation
    max_lines = _TOOL_MAX_LINES.get(tool_name)
    if max_lines and output.count("\n") > max_lines:
        lines = output.split("\n")
        if len(lines) > max_lines:
            keep_head = min(10, max_lines // 5)
            keep_tail = max_lines - keep_head
            trimmed_count = len(lines) - max_lines
            output = "\n".join(
                lines[:keep_head]
                + [f"\n[... {trimmed_count} lines truncated ...]\n"]
                + lines[-keep_tail:]
            )

    # Stage 2: token-based truncation (existing logic)
    output = middle_out_truncate(output, max_tokens=max_output_tokens)
    return output


@dataclass
class TurnRuntimeResult:
    messages: List[Dict[str, Any]]
    pending_completion: bool
    last_agent_message: str
    tool_call_count_delta: int
    completed: bool
    aborted: bool = False
    abort_reason: str = ""
    parallel_batch_count_delta: int = 0
    approval_denials_delta: int = 0
    guard_escalations_delta: int = 0
    subagent_failures_delta: int = 0
    subagent_rate_limit_failures_delta: int = 0
    tool_successes_delta: int = 0
    tool_failures_delta: int = 0


class TurnRuntime:
    """Processes one model response and advances conversation state."""

    def __init__(
        self,
        *,
        cwd: Path,
        tools: Any,
        approval_policy: ApprovalPolicy,
        bypass_approvals: bool,
        approval_cache: dict[str, bool],
        max_output_tokens: int,
        readonly: bool,
        trace_context: dict[str, Any] | None = None,
        budget: Any = None,
        tool_output_max_tokens: int = 0,
    ) -> None:
        self._tools = tools
        self._router = ToolRouter()
        self._policy = ToolPolicyEngine(
            cwd=cwd,
            approval_policy=approval_policy,
            readonly=readonly,
        )
        self._orchestrator = ToolOrchestrator(
            policy=self._policy,
            bypass_approvals=bypass_approvals,
        )
        self._approval_cache = approval_cache
        self._max_output_tokens = max_output_tokens
        self._bypass_approvals = bypass_approvals
        self._trace_context = trace_context or {}
        self._budget = budget
        # Per-model tool output limit (from ModelTier). 0 = use max_output_tokens only.
        self._tool_output_max_tokens = tool_output_max_tokens or max_output_tokens

    def process_response(
        self,
        *,
        response: Any,
        messages: List[Dict[str, Any]],
        ctx: Any,
        pending_completion: bool,
        skip_verification: bool,
        verification_prompt: str,
    ) -> TurnRuntimeResult:
        # Defensive None normalization — some models return null fields
        if response.text is None:
            response.text = ""
        if response.function_calls is None:
            response.function_calls = []
        response_text = response.text or ""
        last_agent_message = response_text

        if response_text:
            item_id = next_item_id()
            emit(ItemCompletedEvent(item=make_agent_message_item(item_id, response_text)))

        has_function_calls = (
            response.has_function_calls()
            if hasattr(response, "has_function_calls")
            else bool(response.function_calls)
        )

        if not has_function_calls:
            if skip_verification:
                return TurnRuntimeResult(
                    messages=messages,
                    pending_completion=pending_completion,
                    last_agent_message=last_agent_message,
                    tool_call_count_delta=0,
                    completed=True,
                    aborted=False,
                )
            if pending_completion:
                return TurnRuntimeResult(
                    messages=messages,
                    pending_completion=True,
                    last_agent_message=last_agent_message,
                    tool_call_count_delta=0,
                    completed=True,
                    aborted=False,
                )
            updated = list(messages)
            updated.append({"role": "assistant", "content": response_text})
            updated.append({"role": "user", "content": verification_prompt})
            return TurnRuntimeResult(
                messages=updated,
                pending_completion=True,
                last_agent_message=last_agent_message,
                tool_call_count_delta=0,
                completed=False,
                aborted=False,
            )

        parsed_calls = self._router.parse_calls(response.function_calls)
        assistant_msg: Dict[str, Any] = {"role": "assistant", "content": response_text}
        tool_calls_data: list[dict[str, Any]] = []
        for call in parsed_calls:
            tool_calls_data.append(
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {
                        "name": call.tool_name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
            )
        assistant_msg["tool_calls"] = tool_calls_data

        updated_messages = list(messages)
        updated_messages.append(assistant_msg)

        precomputed: list[tuple[str, Any, Any]] = []
        for call in parsed_calls:
            if call.parse_error:
                invalid_result = self._invalid_invocation_result(
                    call,
                    f"Invalid tool arguments for `{call.tool_name}`: {call.parse_error}",
                )
                precomputed.append(("invalid", call, invalid_result))
                continue
            if call.validation_errors:
                joined = "; ".join(call.validation_errors)
                invalid_result = self._invalid_invocation_result(
                    call,
                    f"Invalid tool arguments for `{call.tool_name}`: {joined}",
                )
                precomputed.append(("invalid", call, invalid_result))
                continue

            decision = self._policy.evaluate(
                call.tool_name,
                call.arguments,
                is_mutating=tool_is_mutating(call.tool_name, call.arguments),
            )

            if decision.kind == PolicyDecisionKind.FORBIDDEN:
                denied_result = self._invalid_invocation_result(
                    call,
                    decision.reason or "operation forbidden",
                )
                precomputed.append(("forbidden", call, denied_result))
                continue

            if decision.kind == PolicyDecisionKind.NEEDS_APPROVAL and not self._bypass_approvals:
                denied_result = self._invalid_invocation_result(
                    call,
                    decision.reason or "operation requires explicit approval",
                )
                precomputed.append(("needs_approval", call, denied_result))
                continue

            precomputed.append(("run", call, decision))

        run_calls = [(kind, call, decision) for kind, call, decision in precomputed if kind == "run"]
        can_batch = (
            len(run_calls) > 1
            and all(tool_supports_parallel(call.tool_name) for _, call, _ in run_calls)
        )
        batch_results_by_call_id: dict[str, Any] = {}
        if can_batch:
            batch_inputs = [(call.tool_name, call.arguments) for _, call, _ in run_calls]
            batch_results = self._tools.execute_batch(ctx, batch_inputs)
            for (_, call, _), result in zip(run_calls, batch_results):
                batch_results_by_call_id[call.call_id] = result

        approval_denials = 0
        guard_escalations = 0
        subagent_failures = 0
        subagent_rate_limit_failures = 0
        tool_successes = 0
        tool_failures = 0
        for kind, call, decision_or_result in precomputed:
            item_id = next_item_id()
            emit(
                ItemStartedEvent(
                    item=make_command_execution_item(
                        item_id=item_id,
                        command=f"{call.tool_name}({call.arguments})",
                        status="in_progress",
                    )
                )
            )
            emit_raw(
                self._with_runtime_meta(
                    {
                    "type": "stream.tool.started",
                    "tool_name": call.tool_name,
                    "call_id": call.call_id,
                    }
                )
            )

            if kind in ("invalid", "forbidden", "needs_approval"):
                result = decision_or_result
                decision_kind = PolicyDecisionKind.NEEDS_APPROVAL.value
                decision_reason = result.error or result.output
                decision_source = "heuristic"
                approval_outcome = ApprovalOutcome.DENIED.value
                policy_eval = None
                retried_without_guards = False
                abort_turn = kind == "forbidden"
                if kind == "forbidden":
                    decision_kind = PolicyDecisionKind.FORBIDDEN.value
                    approval_outcome = ApprovalOutcome.ABORTED.value
                else:
                    approval_denials += 1
            elif can_batch:
                result = batch_results_by_call_id.get(call.call_id)
                decision_kind = decision_or_result.kind.value
                decision_reason = decision_or_result.reason
                decision_source = decision_or_result.source.value
                approval_outcome = ApprovalOutcome.APPROVED.value
                policy_eval = self._orchestrator._policy_eval_to_dict(  # type: ignore[attr-defined]
                    getattr(decision_or_result, "policy_evaluation", None)
                )
                retried_without_guards = False
                abort_turn = False
            else:
                orchestration = self._orchestrator.run(
                    invocation=call,
                    tools=self._tools,
                    ctx=ctx,
                    is_mutating=tool_is_mutating(call.tool_name, call.arguments),
                    approval_cache=self._approval_cache,
                )
                result = orchestration.result
                decision_kind = orchestration.decision_kind
                decision_reason = orchestration.decision_reason
                decision_source = orchestration.decision_source
                approval_outcome = orchestration.approval_outcome.value
                policy_eval = orchestration.policy_evaluation
                retried_without_guards = orchestration.retried_without_guards
                abort_turn = orchestration.abort_turn
                if orchestration.approval_outcome == ApprovalOutcome.DENIED:
                    approval_denials += 1

            if result is None:
                result = self._invalid_invocation_result(call, "Batch execution failed")
            emit_raw(
                self._with_runtime_meta(
                    {
                    "type": "tool.decision",
                    "tool_name": call.tool_name,
                    "call_id": call.call_id,
                    "decision": decision_kind,
                    "reason": decision_reason,
                    "source": decision_source,
                    "approval_outcome": approval_outcome,
                    "recoverable": decision_kind != PolicyDecisionKind.FORBIDDEN.value,
                    }
                )
            )
            emit_raw(
                self._with_runtime_meta(
                    {
                    "type": "policy.evaluation",
                    "tool_name": call.tool_name,
                    "call_id": call.call_id,
                    "evaluation": policy_eval,
                    "fallback": policy_eval is None,
                    }
                )
            )

            output = _truncate_tool_output(
                result.output,
                tool_name=call.tool_name,
                max_output_tokens=self._tool_output_max_tokens,
            )
            emit(
                ItemCompletedEvent(
                    item=make_command_execution_item(
                        item_id=item_id,
                        command=call.tool_name,
                        status="completed" if result.success else "failed",
                        aggregated_output=output,
                        exit_code=0 if result.success else 1,
                    )
                )
            )
            if result.success:
                tool_successes += 1
            else:
                tool_failures += 1
            emit_raw(
                self._with_runtime_meta(
                    {
                    "type": "stream.tool.completed",
                    "tool_name": call.tool_name,
                    "call_id": call.call_id,
                    "success": result.success,
                    "retried_without_guards": retried_without_guards,
                    "fatal": abort_turn,
                    }
                )
            )
            emit_raw(
                self._with_runtime_meta(
                    {
                    "type": "tool.escalation",
                    "tool_name": call.tool_name,
                    "call_id": call.call_id,
                    "attempt": 2 if retried_without_guards else 1,
                    "retried_without_guards": retried_without_guards,
                    }
                )
            )
            if retried_without_guards:
                guard_escalations += 1
            if call.tool_name == "spawn_subagent" and kind == "run" and not result.success:
                error_text = f"{result.error or ''} {result.output or ''}".lower()
                is_rate_limited = "rate_limit" in error_text or "429" in error_text
                is_runtime_failure = (
                    "error_code=" in (result.output or "")
                    or "subagent error" in (result.error or "").lower()
                    or is_rate_limited
                )
                if is_runtime_failure:
                    subagent_failures += 1
                    if is_rate_limited:
                        subagent_rate_limit_failures += 1

            if result.inject_content:
                updated_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Image from {call.tool_name}:"},
                            result.inject_content,
                        ],
                    }
                )

            updated_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "content": output,
                }
            )
            if abort_turn:
                return TurnRuntimeResult(
                    messages=updated_messages,
                    pending_completion=False,
                    last_agent_message=last_agent_message,
                    tool_call_count_delta=len(parsed_calls),
                    completed=False,
                    aborted=True,
                    abort_reason=result.error or result.output or "Tool operation aborted by policy",
                    parallel_batch_count_delta=1 if can_batch else 0,
                    approval_denials_delta=approval_denials,
                    guard_escalations_delta=guard_escalations,
                    subagent_failures_delta=subagent_failures,
                    subagent_rate_limit_failures_delta=subagent_rate_limit_failures,
                    tool_successes_delta=tool_successes,
                    tool_failures_delta=tool_failures,
                )

        return TurnRuntimeResult(
            messages=updated_messages,
            pending_completion=False,
            last_agent_message=last_agent_message,
            tool_call_count_delta=len(parsed_calls),
            completed=False,
            aborted=False,
            parallel_batch_count_delta=1 if can_batch else 0,
            approval_denials_delta=approval_denials,
            guard_escalations_delta=guard_escalations,
            subagent_failures_delta=subagent_failures,
            subagent_rate_limit_failures_delta=subagent_rate_limit_failures,
            tool_successes_delta=tool_successes,
            tool_failures_delta=tool_failures,
        )

    @staticmethod
    def _invalid_invocation_result(_call: Any, message: str) -> ToolResult:
        return ToolResult.fail(error=message, output=message)

    def _with_runtime_meta(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Attach trace + budget metadata to emitted runtime events."""
        out = dict(payload)
        if self._trace_context:
            out.update(self._trace_context)
        if self._budget is not None:
            snap = self._budget.snapshot()
            out["budget"] = {
                "max_cost": snap.max_cost,
                "consumed_cost": round(snap.consumed_cost, 6),
                "reserved_cost": round(snap.reserved_cost, 6),
                "remaining_cost": round(snap.remaining_cost, 6),
            }
        return out

