"""Approval + sandbox-style orchestration for tool calls."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from src.tools.base import ToolResult
from src.tools.policy import (
    ApprovalOutcome,
    DecisionSource,
    PolicyDecisionKind,
    ToolPolicyEngine,
)
from src.tools.router import ToolInvocation


@dataclass
class OrchestrationResult:
    result: ToolResult
    approved: bool
    retried_without_guards: bool = False
    approval_outcome: ApprovalOutcome = ApprovalOutcome.APPROVED
    abort_turn: bool = False
    decision_kind: str = PolicyDecisionKind.SKIP.value
    decision_reason: str | None = None
    decision_source: str = DecisionSource.HEURISTIC.value
    policy_evaluation: dict[str, Any] | None = None


class ToolOrchestrator:
    """Runs policy checks then executes tools, with optional escalation retry."""

    def __init__(
        self,
        policy: ToolPolicyEngine,
        bypass_approvals: bool = False,
    ) -> None:
        self._policy = policy
        self._bypass_approvals = bypass_approvals

    def run(
        self,
        *,
        invocation: ToolInvocation,
        tools: Any,
        ctx: Any,
        is_mutating: bool,
        approval_cache: dict[str, bool],
        escalate_on_failure: bool = True,
    ) -> OrchestrationResult:
        decision = self._policy.evaluate(
            invocation.tool_name,
            invocation.arguments,
            is_mutating=is_mutating,
        )
        cache_key = self._cache_key(invocation)
        approved = approval_cache.get(cache_key, False)
        policy_eval_dict = self._policy_eval_to_dict(getattr(decision, "policy_evaluation", None))

        if decision.kind == PolicyDecisionKind.FORBIDDEN:
            return OrchestrationResult(
                result=ToolResult.fail(decision.reason or "operation forbidden"),
                approved=False,
                approval_outcome=ApprovalOutcome.ABORTED,
                abort_turn=True,
                decision_kind=decision.kind.value,
                decision_reason=decision.reason,
                decision_source=decision.source.value,
                policy_evaluation=policy_eval_dict,
            )

        if decision.kind == PolicyDecisionKind.NEEDS_APPROVAL and not approved:
            if not self._bypass_approvals:
                abort_on_denied = bool(invocation.arguments.get("abort_on_denied", False))
                return OrchestrationResult(
                    result=ToolResult.fail(
                        decision.reason or "operation requires explicit approval"
                    ),
                    approved=False,
                    approval_outcome=(
                        ApprovalOutcome.ABORTED if abort_on_denied else ApprovalOutcome.DENIED
                    ),
                    abort_turn=abort_on_denied,
                    decision_kind=decision.kind.value,
                    decision_reason=decision.reason,
                    decision_source=decision.source.value,
                    policy_evaluation=policy_eval_dict,
                )
            approved = True
            approval_cache[cache_key] = True
            if (
                invocation.tool_name == "shell_command"
                and invocation.arguments.get("persist_approval", False)
                and decision.amendment_prefix
            ):
                self._policy.append_exec_policy_allow_prefix(decision.amendment_prefix)

        result = tools.execute(
            ctx,
            invocation.tool_name,
            invocation.arguments,
            enforce_guards=True,
        )
        if result.success:
            return OrchestrationResult(
                result=result,
                approved=approved,
                approval_outcome=ApprovalOutcome.APPROVED,
                decision_kind=decision.kind.value,
                decision_reason=decision.reason,
                decision_source=decision.source.value,
                policy_evaluation=policy_eval_dict,
            )

        if (
            escalate_on_failure
            and decision.kind in (PolicyDecisionKind.SKIP, PolicyDecisionKind.NEEDS_APPROVAL)
            and self._bypass_approvals
            and self._is_guard_or_sandbox_denied(result)
        ):
            retry = tools.execute(
                ctx,
                invocation.tool_name,
                invocation.arguments,
                enforce_guards=False,
            )
            return OrchestrationResult(
                result=retry,
                approved=approved,
                retried_without_guards=True,
                approval_outcome=ApprovalOutcome.APPROVED,
                decision_kind=decision.kind.value,
                decision_reason=decision.reason,
                decision_source=decision.source.value,
                policy_evaluation=policy_eval_dict,
            )

        return OrchestrationResult(
            result=result,
            approved=approved,
            approval_outcome=ApprovalOutcome.APPROVED,
            decision_kind=decision.kind.value,
            decision_reason=decision.reason,
            decision_source=decision.source.value,
            policy_evaluation=policy_eval_dict,
        )

    @staticmethod
    def _cache_key(invocation: ToolInvocation) -> str:
        args = json.dumps(invocation.arguments, sort_keys=True, default=str)
        return f"{invocation.tool_name}:{args}"

    @staticmethod
    def _is_guard_or_sandbox_denied(result: ToolResult) -> bool:
        text = f"{result.error or ''}\n{result.output or ''}".lower()
        denied_markers = (
            "denied",
            "readonly mode",
            "read denied",
            "write denied",
            "operation not permitted",
            "permission denied",
            "read-only file system",
            "seccomp",
            "landlock",
            "sandbox",
        )
        return any(marker in text for marker in denied_markers)

    @staticmethod
    def _policy_eval_to_dict(policy_eval: Any) -> dict[str, Any] | None:
        if policy_eval is None:
            return None
        matched = policy_eval.matched_rule
        return {
            "decision": getattr(policy_eval.decision, "value", str(policy_eval.decision)),
            "matched_rule": None
            if not matched
            else {
                "decision": getattr(matched.decision, "value", str(matched.decision)),
                "prefix": matched.prefix,
                "source": matched.source,
            },
            "checked_commands": policy_eval.checked_commands,
        }

