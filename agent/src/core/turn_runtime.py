"""Turn runtime for response processing and tool orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

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
                is_mutating=tool_is_mutating(call.tool_name),
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
                {
                    "type": "stream.tool.started",
                    "tool_name": call.tool_name,
                    "call_id": call.call_id,
                }
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
                    is_mutating=tool_is_mutating(call.tool_name),
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
            emit_raw(
                {
                    "type": "policy.evaluation",
                    "tool_name": call.tool_name,
                    "call_id": call.call_id,
                    "evaluation": policy_eval,
                    "fallback": policy_eval is None,
                }
            )

            output = middle_out_truncate(result.output, max_tokens=self._max_output_tokens)
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
            emit_raw(
                {
                    "type": "stream.tool.completed",
                    "tool_name": call.tool_name,
                    "call_id": call.call_id,
                    "success": result.success,
                    "retried_without_guards": retried_without_guards,
                    "fatal": abort_turn,
                }
            )
            emit_raw(
                {
                    "type": "tool.escalation",
                    "tool_name": call.tool_name,
                    "call_id": call.call_id,
                    "attempt": 2 if retried_without_guards else 1,
                    "retried_without_guards": retried_without_guards,
                }
            )
            if retried_without_guards:
                guard_escalations += 1

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
        )

    @staticmethod
    def _invalid_invocation_result(_call: Any, message: str) -> ToolResult:
        return ToolResult.fail(error=message, output=message)

