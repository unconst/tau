"""Path and command guards for tool execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _normalize_roots(roots: Iterable[str], fallback: Path) -> list[Path]:
    resolved: list[Path] = []
    base = fallback.expanduser().resolve()
    for root in roots:
        p = Path(root).expanduser()
        if not p.is_absolute():
            p = base / p
        p = p.resolve()
        resolved.append(p)
    if not resolved:
        resolved.append(base)
    return resolved


@dataclass
class GuardConfig:
    cwd: Path
    readable_roots: list[Path]
    writable_roots: list[Path]
    readonly: bool = False
    enabled: bool = True

    @classmethod
    def from_paths(
        cls,
        cwd: Path,
        readable_roots: list[str] | None = None,
        writable_roots: list[str] | None = None,
        readonly: bool = False,
        enabled: bool = True,
    ) -> "GuardConfig":
        readable = _normalize_roots(readable_roots or [], cwd)
        writable = _normalize_roots(writable_roots or [], cwd)
        return cls(
            cwd=cwd.resolve(),
            readable_roots=readable,
            writable_roots=writable,
            readonly=readonly,
            enabled=enabled,
        )


class GuardError(ValueError):
    """Raised when a tool call violates guard policies."""


class PathGuards:
    """Validates path access and mutation permissions."""

    def __init__(self, config: GuardConfig):
        self.config = config

    @staticmethod
    def _is_within(path: Path, roots: list[Path]) -> bool:
        for root in roots:
            try:
                path.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def require_read(self, path: Path) -> Path:
        resolved = path.expanduser().resolve()
        if not self.config.enabled:
            return resolved
        if not self._is_within(resolved, self.config.readable_roots):
            raise GuardError(f"Read denied outside readable roots: {resolved}")
        return resolved

    def require_write(self, path: Path) -> Path:
        resolved = path.expanduser().resolve()
        if not self.config.enabled:
            return resolved
        if self.config.readonly:
            raise GuardError("Write denied in readonly mode")
        if not self._is_within(resolved, self.config.writable_roots):
            raise GuardError(f"Write denied outside writable roots: {resolved}")
        return resolved

    def require_mutation_allowed(self, tool_name: str) -> None:
        if not self.config.enabled:
            return
        if self.config.readonly:
            raise GuardError(f"Tool `{tool_name}` denied in readonly mode")

