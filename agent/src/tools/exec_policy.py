"""Minimal exec-policy engine for shell command approvals."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import shlex
from typing import Iterable, List, Optional


class ExecPolicyDecision(str, Enum):
    ALLOW = "allow"
    PROMPT = "prompt"
    FORBID = "forbid"


@dataclass
class ExecPolicyRule:
    decision: ExecPolicyDecision
    prefix: list[str]
    source: str


@dataclass
class ExecPolicyEvaluation:
    decision: ExecPolicyDecision
    matched_rule: Optional[ExecPolicyRule] = None
    checked_commands: list[list[str]] | None = None


def split_shell_commands(command: str) -> list[str]:
    """Split a shell script into plain command segments."""
    if not command.strip():
        return []

    # Support common wrapped form: bash -lc "cmd1 && cmd2"
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.strip().split()

    if len(tokens) >= 3 and tokens[0] in ("bash", "sh") and tokens[1] in ("-lc", "-c"):
        script = " ".join(tokens[2:])
        script = script.strip('"').strip("'")
    else:
        script = command

    segments: list[str] = []
    current = ""
    i = 0
    while i < len(script):
        ch = script[i]
        nxt = script[i + 1] if i + 1 < len(script) else ""
        if (ch == "&" and nxt == "&") or (ch == "|" and nxt == "|"):
            if current.strip():
                segments.append(current.strip())
            current = ""
            i += 2
            continue
        if ch == ";":
            if current.strip():
                segments.append(current.strip())
            current = ""
            i += 1
            continue
        current += ch
        i += 1
    if current.strip():
        segments.append(current.strip())
    return segments


def tokenize_command(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.strip().split()


class ExecPolicyEngine:
    """Loads and evaluates prefix rules from project and user layers."""

    PROJECT_RELATIVE_RULES = Path(".agent/rules/default.rules")
    USER_RULES = Path.home() / ".superagent" / "rules" / "default.rules"

    def __init__(self, cwd: Path):
        self.cwd = cwd
        self.rules: list[ExecPolicyRule] = []
        self.reload()

    def reload(self) -> None:
        self.rules = []
        project_rules = self.cwd / self.PROJECT_RELATIVE_RULES
        self.rules.extend(self._load_rules(project_rules, "project"))
        self.rules.extend(self._load_rules(self.USER_RULES, "user"))

    def append_allow_prefix(self, prefix: Iterable[str]) -> None:
        prefix_tokens = [p for p in prefix if p]
        if not prefix_tokens:
            return
        path = self.USER_RULES
        path.parent.mkdir(parents=True, exist_ok=True)
        line = f"allow:{' '.join(prefix_tokens)}\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
        self.reload()

    def evaluate_shell_command(self, command: str) -> ExecPolicyEvaluation:
        split = split_shell_commands(command)
        tokenized = [tokenize_command(cmd) for cmd in split if cmd.strip()]
        decision = ExecPolicyDecision.ALLOW
        matched: Optional[ExecPolicyRule] = None

        for cmd_tokens in tokenized:
            if not cmd_tokens:
                continue
            match = self._match_rules(cmd_tokens)
            if not match:
                continue
            if match.decision == ExecPolicyDecision.FORBID:
                return ExecPolicyEvaluation(
                    decision=ExecPolicyDecision.FORBID,
                    matched_rule=match,
                    checked_commands=tokenized,
                )
            if match.decision == ExecPolicyDecision.PROMPT:
                decision = ExecPolicyDecision.PROMPT
                matched = match
            elif decision == ExecPolicyDecision.ALLOW:
                matched = match

        return ExecPolicyEvaluation(
            decision=decision,
            matched_rule=matched,
            checked_commands=tokenized,
        )

    def _match_rules(self, command_tokens: list[str]) -> Optional[ExecPolicyRule]:
        best: Optional[ExecPolicyRule] = None
        for rule in self.rules:
            if len(command_tokens) < len(rule.prefix):
                continue
            if command_tokens[: len(rule.prefix)] != rule.prefix:
                continue
            if best is None or len(rule.prefix) > len(best.prefix):
                best = rule
            elif best and len(rule.prefix) == len(best.prefix):
                # Strictness precedence for equal specificity.
                precedence = {
                    ExecPolicyDecision.FORBID: 3,
                    ExecPolicyDecision.PROMPT: 2,
                    ExecPolicyDecision.ALLOW: 1,
                }
                if precedence[rule.decision] > precedence[best.decision]:
                    best = rule
        return best

    @staticmethod
    def _load_rules(path: Path, source: str) -> list[ExecPolicyRule]:
        if not path.exists():
            return []
        rules: list[ExecPolicyRule] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            raw_decision, raw_prefix = line.split(":", 1)
            decision_map = {
                "allow": ExecPolicyDecision.ALLOW,
                "prompt": ExecPolicyDecision.PROMPT,
                "forbid": ExecPolicyDecision.FORBID,
            }
            decision = decision_map.get(raw_decision.strip().lower())
            if not decision:
                continue
            try:
                prefix_tokens = shlex.split(raw_prefix.strip())
            except ValueError:
                prefix_tokens = raw_prefix.strip().split()
            if not prefix_tokens:
                continue
            rules.append(
                ExecPolicyRule(
                    decision=decision,
                    prefix=prefix_tokens,
                    source=source,
                )
            )
        return rules

