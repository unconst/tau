"""Approval and risk policy for tool execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from src.core.executor import AgentExecutor, RiskLevel
from src.tools.exec_policy import ExecPolicyDecision, ExecPolicyEngine, ExecPolicyEvaluation


class ApprovalPolicy(str, Enum):
    UNTRUSTED = "untrusted"
    ON_REQUEST = "on-request"
    ON_FAILURE = "on-failure"
    NEVER = "never"


class PolicyDecisionKind(str, Enum):
    SKIP = "skip"
    NEEDS_APPROVAL = "needs_approval"
    FORBIDDEN = "forbidden"


class ApprovalOutcome(str, Enum):
    APPROVED = "approved"
    DENIED = "denied"
    ABORTED = "aborted"


class DecisionSource(str, Enum):
    HEURISTIC = "heuristic"
    EXEC_POLICY = "exec_policy"
    READONLY = "readonly"


@dataclass
class PolicyDecision:
    kind: PolicyDecisionKind
    reason: Optional[str] = None
    risk_level: Optional[RiskLevel] = None
    source: DecisionSource = DecisionSource.HEURISTIC
    policy_evaluation: Optional[ExecPolicyEvaluation] = None
    amendment_prefix: Optional[list[str]] = None


class ToolPolicyEngine:
    """Maps tool requests to policy decisions."""

    def __init__(
        self,
        cwd: Path,
        approval_policy: ApprovalPolicy = ApprovalPolicy.ON_FAILURE,
        readonly: bool = False,
    ) -> None:
        self._approval_policy = approval_policy
        self._readonly = readonly
        self._risk_executor = AgentExecutor(cwd=cwd)
        self._exec_policy = ExecPolicyEngine(cwd=cwd)
        self._safe_shell_prefixes: list[tuple[str, ...]] = [
            ("ls",),
            ("pwd",),
            ("cat",),
            ("head",),
            ("tail",),
            ("rg",),
            ("grep",),
            ("find",),
            ("git", "status"),
            ("git", "log"),
            ("git", "diff"),
        ]

    def assess_risk(self, tool_name: str, arguments: dict[str, Any]) -> RiskLevel:
        return self._risk_executor.assess_risk(tool_name, arguments)

    def append_exec_policy_allow_prefix(self, prefix: list[str]) -> None:
        """Persist an allow-prefix amendment in user exec-policy rules."""
        self._exec_policy.append_allow_prefix(prefix)

    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        is_mutating: bool,
    ) -> PolicyDecision:
        risk = self.assess_risk(tool_name, arguments)

        if self._readonly and is_mutating:
            return PolicyDecision(
                kind=PolicyDecisionKind.FORBIDDEN,
                reason="readonly mode forbids mutating tools",
                risk_level=risk,
                source=DecisionSource.READONLY,
            )

        # Shell-specific exec-policy + safe-command fast path.
        if tool_name == "shell_command":
            command = (arguments.get("command") or "").strip()
            if command:
                if self._is_known_safe_shell(command):
                    return PolicyDecision(
                        kind=PolicyDecisionKind.SKIP,
                        risk_level=risk,
                        source=DecisionSource.HEURISTIC,
                    )
                if self._is_known_risky_shell(command):
                    return PolicyDecision(
                        kind=PolicyDecisionKind.NEEDS_APPROVAL,
                        reason="known-risk shell command requires approval",
                        risk_level=risk,
                        source=DecisionSource.HEURISTIC,
                        amendment_prefix=self._derive_shell_amendment(arguments),
                    )
                policy_eval = self._exec_policy.evaluate_shell_command(command)
                if policy_eval.decision == ExecPolicyDecision.FORBID:
                    return PolicyDecision(
                        kind=PolicyDecisionKind.FORBIDDEN,
                        reason="exec-policy forbids this command",
                        risk_level=risk,
                        source=DecisionSource.EXEC_POLICY,
                        policy_evaluation=policy_eval,
                    )
                if policy_eval.decision == ExecPolicyDecision.PROMPT:
                    amendment = self._derive_amendment_prefix(policy_eval)
                    return PolicyDecision(
                        kind=PolicyDecisionKind.NEEDS_APPROVAL,
                        reason="exec-policy requires approval",
                        risk_level=risk,
                        source=DecisionSource.EXEC_POLICY,
                        policy_evaluation=policy_eval,
                        amendment_prefix=amendment,
                    )
                # ALLOW falls through to standard policy mode semantics.
                allowed = PolicyDecision(
                    kind=PolicyDecisionKind.SKIP,
                    risk_level=risk,
                    source=DecisionSource.EXEC_POLICY,
                    policy_evaluation=policy_eval,
                )
                if self._approval_policy == ApprovalPolicy.UNTRUSTED and risk != RiskLevel.SAFE:
                    return PolicyDecision(
                        kind=PolicyDecisionKind.NEEDS_APPROVAL,
                        reason="untrusted mode requires approval for non-safe tools",
                        risk_level=risk,
                        source=DecisionSource.HEURISTIC,
                        policy_evaluation=policy_eval,
                        amendment_prefix=self._derive_amendment_prefix(policy_eval),
                    )
                return allowed

        if self._approval_policy == ApprovalPolicy.NEVER:
            return PolicyDecision(kind=PolicyDecisionKind.SKIP, risk_level=risk)

        if self._approval_policy == ApprovalPolicy.ON_FAILURE:
            return PolicyDecision(kind=PolicyDecisionKind.SKIP, risk_level=risk)

        if self._approval_policy == ApprovalPolicy.ON_REQUEST:
            if risk in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                return PolicyDecision(
                    kind=PolicyDecisionKind.NEEDS_APPROVAL,
                    reason="high-risk operation requires approval",
                    risk_level=risk,
                    source=DecisionSource.HEURISTIC,
                    amendment_prefix=self._derive_shell_amendment(arguments),
                )
            return PolicyDecision(kind=PolicyDecisionKind.SKIP, risk_level=risk)

        # UNTRUSTED: require approval for non-safe actions
        if risk == RiskLevel.SAFE:
            return PolicyDecision(kind=PolicyDecisionKind.SKIP, risk_level=risk)
        return PolicyDecision(
            kind=PolicyDecisionKind.NEEDS_APPROVAL,
            reason="untrusted mode requires approval for non-safe tools",
            risk_level=risk,
            source=DecisionSource.HEURISTIC,
            amendment_prefix=self._derive_shell_amendment(arguments),
        )

    def _is_known_safe_shell(self, command: str) -> bool:
        from src.tools.exec_policy import split_shell_commands, tokenize_command

        segments = split_shell_commands(command)
        if not segments:
            return False
        for segment in segments:
            tokens = tokenize_command(segment)
            if not tokens:
                continue
            if not self._matches_any_safe_prefix(tokens):
                return False
        return True

    def _matches_any_safe_prefix(self, tokens: list[str]) -> bool:
        for prefix in self._safe_shell_prefixes:
            if len(tokens) >= len(prefix) and tuple(tokens[: len(prefix)]) == prefix:
                return True
        return False

    def _is_known_risky_shell(self, command: str) -> bool:
        from src.tools.exec_policy import split_shell_commands, tokenize_command

        risky_prefixes: list[tuple[str, ...]] = [
            ("rm",),
            ("mv",),
            ("chmod",),
            ("chown",),
            ("dd",),
            ("sudo",),
            ("git", "push"),
            ("git", "reset"),
            ("python",),
            ("pip",),
            ("uv", "pip"),
        ]
        for segment in split_shell_commands(command):
            tokens = tokenize_command(segment)
            if not tokens:
                continue
            if any(
                len(tokens) >= len(prefix) and tuple(tokens[: len(prefix)]) == prefix
                for prefix in risky_prefixes
            ):
                return True
            lower_segment = segment.lower()
            if ">" in lower_segment or ">>" in lower_segment:
                return True
        return False

    @staticmethod
    def _derive_amendment_prefix(policy_eval: ExecPolicyEvaluation | None) -> list[str] | None:
        if not policy_eval or not policy_eval.checked_commands:
            return None
        for cmd in policy_eval.checked_commands:
            if cmd:
                return cmd
        return None

    def _derive_shell_amendment(self, arguments: dict[str, Any]) -> list[str] | None:
        command = (arguments.get("command") or "").strip()
        if not command:
            return None
        policy_eval = self._exec_policy.evaluate_shell_command(command)
        return self._derive_amendment_prefix(policy_eval)

