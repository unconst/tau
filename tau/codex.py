"""Codex CLI wrapper — centralises command construction and event parsing.

All subprocess invocations of `codex exec` should go through the helpers in
this module so that flag/model changes only need to happen in one place.
"""

from __future__ import annotations

import os
import re as _re
from typing import Any

# ---------------------------------------------------------------------------
# Model mapping — env-var overrides with sensible defaults
# Two tiers: a fast model for read-only queries and a coding-agent model.
# ---------------------------------------------------------------------------

# Lightweight model for ask/chat/plan/summary (read-only queries)
CHAT_MODEL = os.getenv("TAU_CODEX_CHAT_MODEL", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")

# Coding agent model for /adapt, agent loop, cron jobs (writes code + runs tools)
AGENT_MODEL = os.getenv("TAU_CODEX_AGENT_MODEL", "zai-org/GLM-4.7-TEE")


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------

def build_cmd(
    prompt: str,
    *,
    model: str | None = None,
    json_mode: bool = False,
    readonly: bool = False,
    extra_flags: list[str] | None = None,
) -> list[str]:
    """Build the ``codex exec`` command list.

    Parameters
    ----------
    prompt:
        The user/agent prompt (passed as the last positional argument).
    model:
        Model name to pass via ``--model``.  When *None* the codex
        config.toml default is used.
    json_mode:
        If ``True``, append ``--json`` so stdout emits JSONL events.
    readonly:
        If ``True``, use ``--sandbox read-only`` (no ``--full-auto``).
        If ``False`` (default), use ``--full-auto`` which implies
        ``--sandbox workspace-write`` with auto-approval.
    extra_flags:
        Any additional CLI flags (e.g. ``["-c", "key=value"]``).
    """
    cmd: list[str] = ["codex", "exec"]

    if readonly:
        cmd += ["--sandbox", "read-only"]
    else:
        cmd.append("--full-auto")

    if model:
        cmd += ["--model", model]

    if json_mode:
        cmd.append("--json")

    if extra_flags:
        cmd.extend(extra_flags)

    cmd.append(prompt)
    return cmd


# ---------------------------------------------------------------------------
# Think-tag stripper
# ---------------------------------------------------------------------------
_THINK_RE = _re.compile(r"<think>[\s\S]*?</think>", _re.IGNORECASE)


def strip_think_tags(text: str) -> str:
    """Remove ``<think>...</think>`` blocks emitted by reasoning models."""
    return _THINK_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# JSONL event normaliser
# ---------------------------------------------------------------------------
# Codex emits events shaped like:
#   {"type": "item.started",   "item": {"type": "agent_message", ...}}
#   {"type": "item.completed", "item": {"type": "command_execution", ...}}
#   {"type": "turn.completed", "usage": {...}}
#
# The streaming loops in __init__.py consume normalised events from
# parse_event() below, which translates raw Codex JSONL into a
# uniform internal format.
#
# parse_event() translates Codex events into a uniform dict:
#   {"kind": "tool_start"|"tool_done"|"message"|"message_done"|"turn_done"|"error",
#    "text": str | None,
#    "tool_name": str | None,
#    "tool_detail": str | None,
#    "item_type": str | None}

def parse_event(event: dict[str, Any]) -> dict[str, Any] | None:
    """Normalise a Codex JSONL event into an internal dict.

    Returns ``None`` for events we don't care about (e.g. thread.started).
    """
    etype = event.get("type", "")
    item = event.get("item") or {}
    item_type = item.get("type", "")

    # -- Item started ----------------------------------------------------------
    if etype == "item.started":
        if item_type == "agent_message":
            return {"kind": "message", "text": strip_think_tags(item.get("text", "")), "tool_name": None, "tool_detail": None, "item_type": item_type}
        if item_type == "command_execution":
            cmd_str = item.get("command", "")
            return {"kind": "tool_start", "text": None, "tool_name": "Shell", "tool_detail": cmd_str, "item_type": item_type}
        if item_type == "file_change":
            changes = item.get("changes") or []
            detail = ", ".join(c.get("path", "") for c in changes[:3])
            return {"kind": "tool_start", "text": None, "tool_name": "FileChange", "tool_detail": detail, "item_type": item_type}
        if item_type == "mcp_tool_call":
            tool = item.get("tool", "mcp")
            return {"kind": "tool_start", "text": None, "tool_name": f"MCP:{tool}", "tool_detail": None, "item_type": item_type}
        if item_type == "reasoning":
            return {"kind": "message", "text": item.get("text", ""), "tool_name": None, "tool_detail": None, "item_type": item_type}
        if item_type == "web_search":
            return {"kind": "tool_start", "text": None, "tool_name": "WebSearch", "tool_detail": None, "item_type": item_type}
        if item_type == "error":
            return {"kind": "error", "text": item.get("message", item.get("text", "Unknown error")), "tool_name": None, "tool_detail": None, "item_type": item_type}
        if item_type == "todo_list":
            return None  # skip
        # Fallback
        return {"kind": "tool_start", "text": None, "tool_name": item_type, "tool_detail": None, "item_type": item_type}

    # -- Item updated ----------------------------------------------------------
    if etype == "item.updated":
        if item_type == "agent_message":
            return {"kind": "message", "text": strip_think_tags(item.get("text", "")), "tool_name": None, "tool_detail": None, "item_type": item_type}
        if item_type == "command_execution":
            return {"kind": "tool_start", "text": None, "tool_name": "Shell", "tool_detail": item.get("aggregated_output", ""), "item_type": item_type}
        if item_type == "reasoning":
            return {"kind": "message", "text": item.get("text", ""), "tool_name": None, "tool_detail": None, "item_type": item_type}
        return None

    # -- Item completed --------------------------------------------------------
    if etype == "item.completed":
        if item_type == "agent_message":
            return {"kind": "message_done", "text": strip_think_tags(item.get("text", "")), "tool_name": None, "tool_detail": None, "item_type": item_type}
        if item_type == "command_execution":
            return {"kind": "tool_done", "text": None, "tool_name": "Shell", "tool_detail": item.get("command", ""), "item_type": item_type}
        if item_type == "file_change":
            changes = item.get("changes") or []
            detail = ", ".join(c.get("path", "") for c in changes[:3])
            return {"kind": "tool_done", "text": None, "tool_name": "FileChange", "tool_detail": detail, "item_type": item_type}
        if item_type == "mcp_tool_call":
            tool = item.get("tool", "mcp")
            return {"kind": "tool_done", "text": None, "tool_name": f"MCP:{tool}", "tool_detail": None, "item_type": item_type}
        if item_type == "reasoning":
            return {"kind": "message_done", "text": item.get("text", ""), "tool_name": None, "tool_detail": None, "item_type": item_type}
        if item_type == "web_search":
            return {"kind": "tool_done", "text": None, "tool_name": "WebSearch", "tool_detail": None, "item_type": item_type}
        if item_type == "error":
            return {"kind": "error", "text": item.get("message", item.get("text", "Unknown error")), "tool_name": None, "tool_detail": None, "item_type": item_type}
        # Fallback
        return {"kind": "tool_done", "text": None, "tool_name": item_type, "tool_detail": None, "item_type": item_type}

    # -- Turn events -----------------------------------------------------------
    if etype == "turn.completed":
        return {"kind": "turn_done", "text": None, "tool_name": None, "tool_detail": None, "item_type": None}

    if etype == "turn.failed":
        err = event.get("error") or {}
        return {"kind": "error", "text": err.get("message", "Turn failed"), "tool_name": None, "tool_detail": None, "item_type": None}

    if etype == "error":
        return {"kind": "error", "text": event.get("message", "Unknown error"), "tool_name": None, "tool_detail": None, "item_type": None}

    # thread.started, turn.started — no-ops
    return None


def format_tool_update(parsed: dict[str, Any]) -> str | None:
    """Return a human-friendly one-liner for a tool_start event, or None."""
    if parsed["kind"] != "tool_start":
        return None

    name = parsed.get("tool_name") or "tool"
    detail = (parsed.get("tool_detail") or "")[:40]

    if name == "Shell":
        return f"💻 {detail}..." if detail else "💻 Running command..."
    if name == "FileChange":
        return f"✍️ Editing {detail}..." if detail else "✍️ Editing files..."
    if name == "WebSearch":
        return "🌐 Searching the web..."
    if name.startswith("MCP:"):
        tool = name.split(":", 1)[1]
        return f"🔧 {tool}..."
    return f"🔧 {name}..."
