"""Hashline utilities for line-level addressing with integrity hashes.

Each line is tagged as ``line_number:hash|content`` where *hash* is a short
(2-char hex) fingerprint of the line content.  This lets the LLM reference
specific lines unambiguously even when surrounding code shifts.
"""

from __future__ import annotations

import hashlib
from typing import Sequence


def line_hash(content: str) -> str:
    """Return a 2-char hex hash for a line of content.

    The hash is the first two hex digits of the MD5 of the stripped line.
    """
    stripped = content.rstrip("\n\r")
    return hashlib.md5(stripped.encode("utf-8", errors="replace")).hexdigest()[:2]


def format_hashline(line_number: int, content: str) -> str:
    """Format a single line with its hashline tag.

    Returns ``'line_number:hash|content'``.
    """
    h = line_hash(content)
    return f"{line_number}:{h}|{content}"


def format_lines(
    lines: Sequence[str],
    start: int = 1,
) -> list[str]:
    """Format multiple lines with hashline tags.

    Parameters
    ----------
    lines:
        Sequence of line strings (newlines are stripped automatically).
    start:
        1-based line number for the first line.

    Returns
    -------
    list[str]
        Each element is ``'line_number:hash|content'``.
    """
    return [
        format_hashline(start + i, line.rstrip("\n\r"))
        for i, line in enumerate(lines)
    ]


def compute_hashes(lines: Sequence[str]) -> list[str]:
    """Compute hashes for every line in *lines*.

    Returns a list of 2-char hex hashes aligned by index (0-based).
    """
    return [line_hash(line) for line in lines]


def parse_ref(ref: str) -> tuple[int, str]:
    """Parse a ``'line_number:hash'`` reference string.

    Returns ``(line_number, hash)`` where *line_number* is 1-based.
    Raises ``ValueError`` on malformed input.
    """
    if ":" not in ref:
        raise ValueError(f"Invalid hashline ref (missing ':'): {ref!r}")
    parts = ref.split(":", 1)
    try:
        line_number = int(parts[0])
    except ValueError:
        raise ValueError(f"Invalid line number in ref: {ref!r}")
    return line_number, parts[1]


def validate_ref(ref: str, lines: Sequence[str]) -> tuple[bool, str | None]:
    """Validate a hashline reference against file content.

    Parameters
    ----------
    ref:
        A ``'line_number:hash'`` string.
    lines:
        The file's content split into lines (0-indexed list, but line numbers
        in *ref* are 1-based).

    Returns
    -------
    tuple[bool, str | None]
        ``(True, None)`` if valid, ``(False, error_message)`` otherwise.
    """
    try:
        line_number, expected_hash = parse_ref(ref)
    except ValueError as exc:
        return False, str(exc)

    if line_number < 1 or line_number > len(lines):
        return False, (
            f"Line {line_number} out of range (file has {len(lines)} lines)"
        )

    actual_content = lines[line_number - 1]
    actual_hash = line_hash(actual_content)
    if actual_hash != expected_hash:
        # Include actual hash and line content so the model can self-correct
        # without re-reading the file.
        content_preview = actual_content.rstrip("\n\r")
        if len(content_preview) > 120:
            content_preview = content_preview[:120] + "..."
        return False, (
            f"Hash mismatch at line {line_number}: "
            f"expected {expected_hash!r}, actual {actual_hash!r}. "
            f"Current content: {content_preview!r}"
        )

    return True, None
