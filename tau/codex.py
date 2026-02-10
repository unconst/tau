"""Agent wrapper — in-process LLM and agent interface.

All agent invocations go through in-process calls to the ``agent``
package.  Three entry-points are provided:

* :func:`llm_chat`  — simple text completion (no tools, no agent loop).
* :func:`run_baseagent` — blocking agent execution, returns final text.
* :func:`run_baseagent_streaming` — threaded agent with JSONL event queue.

Helpers (:func:`parse_event`, :func:`format_tool_update`,
:func:`strip_think_tags`) support JSONL event consumption across ``tau``.
"""

from __future__ import annotations

import json
import os
import queue
import re as _re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Model mapping — env-var overrides with sensible defaults
# ---------------------------------------------------------------------------

# Lightweight model for ask/chat/plan/summary (read-only queries)
CHAT_MODEL = os.getenv("TAU_CHAT_MODEL") or os.getenv(
    "TAU_CODEX_CHAT_MODEL",  # legacy fallback
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
)

# Coding agent model for /adapt, agent loop, cron jobs (writes code + runs tools)
AGENT_MODEL = os.getenv("TAU_AGENT_MODEL") or os.getenv(
    "TAU_CODEX_AGENT_MODEL",  # legacy fallback
    "zai-org/GLM-4.7-TEE",
)

# Workspace root (same as the rest of tau)
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Think-tag stripper
# ---------------------------------------------------------------------------
_THINK_RE = _re.compile(r"<think>[\s\S]*?</think>", _re.IGNORECASE)


def strip_think_tags(text: str) -> str:
    """Remove ``<think>...</think>`` blocks emitted by reasoning models."""
    return _THINK_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Helpers – ensure the agent package is importable
# ---------------------------------------------------------------------------

def _ensure_agent_on_path() -> None:
    """Add the agent directory to ``sys.path`` if not already present."""
    agent_dir = os.path.join(WORKSPACE, "agent")
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)


def _make_llm_client(model: str, timeout: float = 300.0):
    """Create an LLMClient from the agent package."""
    _ensure_agent_on_path()
    from src.llm.client import LLMClient  # type: ignore[import-untyped]

    return LLMClient(
        model=model,
        temperature=0.0,
        max_tokens=16384,
        timeout=timeout,
    )


def _make_config(readonly: bool = False) -> Dict[str, Any]:
    """Build the config dict consumed by ``run_agent_loop``."""
    return {
        "model": AGENT_MODEL,
        "provider": "chutes",
        "reasoning_effort": "none",
        "max_tokens": 16384,
        "temperature": 0.0,
        "max_iterations": 200 if not readonly else 50,
        "max_output_tokens": 2500,
        "shell_timeout": 60,
        "model_context_limit": 200_000,
        "output_token_max": 32_000,
        "auto_compact_threshold": 0.85,
        "prune_protect": 40_000,
        "prune_minimum": 20_000,
        "cache_enabled": True,
        "bypass_approvals": True,
        "bypass_sandbox": True,
        "skip_git_check": True,
        "unified_exec": True,
        "json_output": True,
        "require_completion_confirmation": False,
    }


# ---------------------------------------------------------------------------
# Register Tau-specific tools on a ToolRegistry
# ---------------------------------------------------------------------------

def _register_tau_tools(tools) -> None:
    """Register Tau-specific tools on *tools* (a ``ToolRegistry``)."""
    _ensure_agent_on_path()
    from src.tools.base import ToolResult  # type: ignore[import-untyped]

    venv_prefix = f"source {WORKSPACE}/.venv/bin/activate && "

    def _run_tau_tool(module: str, args_str: str = "") -> ToolResult:
        """Helper – run a tau tool module as a subprocess."""
        cmd = f"{venv_prefix}python -m tau.tools.{module} {args_str}"
        try:
            result = subprocess.run(
                ["sh", "-c", cmd],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=WORKSPACE,
            )
            output = (result.stdout + "\n" + result.stderr).strip()
            return ToolResult(success=result.returncode == 0, output=output)
        except subprocess.TimeoutExpired:
            return ToolResult.fail("Tool timed out")
        except Exception as e:
            return ToolResult.fail(str(e))

    # send_message
    tools.register_tool(
        "send_message",
        {
            "name": "send_message",
            "description": "Send a text message to the owner via Telegram.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "The message text to send."},
                },
                "required": ["message"],
            },
        },
        lambda args: _run_tau_tool("send_message", json.dumps(args.get("message", ""))),
    )

    # send_voice
    tools.register_tool(
        "send_voice",
        {
            "name": "send_voice",
            "description": "Send a TTS voice message to the owner via Telegram.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Text to speak."},
                },
                "required": ["message"],
            },
        },
        lambda args: _run_tau_tool("send_voice", json.dumps(args.get("message", ""))),
    )

    # search_skills
    tools.register_tool(
        "search_skills",
        {
            "name": "search_skills",
            "description": (
                "Search the creative AI skills catalog (image/video/audio generation, etc.). "
                "Pass a query string, or leave blank to list all skills."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (optional)."},
                    "category": {
                        "type": "string",
                        "description": "Filter by category: image, video, audio, social, utility.",
                    },
                },
            },
        },
        lambda args: _run_tau_tool(
            "search_skills",
            (args.get("query") or "")
            + (f" --category {args['category']}" if args.get("category") else ""),
        ),
    )

    # create_task
    tools.register_tool(
        "create_task",
        {
            "name": "create_task",
            "description": "Create a task for Tau to process later.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short task title."},
                    "description": {"type": "string", "description": "Detailed description."},
                },
                "required": ["title", "description"],
            },
        },
        lambda args: _run_tau_tool(
            "create_task",
            f"{json.dumps(args.get('title', ''))} {json.dumps(args.get('description', ''))}",
        ),
    )

    # schedule_message
    tools.register_tool(
        "schedule_message",
        {
            "name": "schedule_message",
            "description": (
                "Schedule a future message.  Use --in '2h' for relative or "
                "--at '14:00' for absolute time, or --cron '0 9 * * *' for recurring."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Message text."},
                    "delay": {
                        "type": "string",
                        "description": "Relative delay like '30m', '2h', '1d'.",
                    },
                    "at": {"type": "string", "description": "Absolute time like '14:00'."},
                    "cron": {"type": "string", "description": "Cron expression."},
                },
                "required": ["message"],
            },
        },
        lambda args: _run_tau_tool(
            "schedule_message",
            (f"--in {json.dumps(args['delay'])} " if args.get("delay") else "")
            + (f"--at {json.dumps(args['at'])} " if args.get("at") else "")
            + (f"--cron {json.dumps(args['cron'])} " if args.get("cron") else "")
            + json.dumps(args.get("message", "")),
        ),
    )

    # commands (catch-all for bot commands)
    tools.register_tool(
        "commands",
        {
            "name": "commands",
            "description": (
                "Execute any tau bot command directly.  Available: "
                "task, plan, status, adapt, cron, crons, uncron, clear, restart, kill, debug."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command name."},
                    "args": {"type": "string", "description": "Command arguments."},
                },
                "required": ["command"],
            },
        },
        lambda args: _run_tau_tool(
            "commands",
            f"{args.get('command', '')} {args.get('args', '')}",
        ),
    )


# ---------------------------------------------------------------------------
# 1) llm_chat — simple text completion (no tools)
# ---------------------------------------------------------------------------

def llm_chat(prompt: str, *, model: str | None = None, timeout: float = 300.0) -> str:
    """Run a simple text completion — no tools, no agent loop.

    Used for summaries, fact extraction, and other pure-LLM tasks.
    Returns the raw text response (think-tags stripped).
    """
    llm = _make_llm_client(model or CHAT_MODEL, timeout=timeout)
    try:
        response = llm.chat(
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=4096,
        )
        return strip_think_tags(response.text or "")
    except Exception as e:
        return f"Error: {e}"
    finally:
        llm.close()


# ---------------------------------------------------------------------------
# 2) run_baseagent — blocking agent execution
# ---------------------------------------------------------------------------

def run_baseagent(
    prompt: str,
    *,
    model: str | None = None,
    readonly: bool = False,
    cwd: str | None = None,
    system_prompt_override: str | None = None,
) -> str:
    """Run the baseagent loop in-process, blocking until done.

    Returns the final agent message text (think-tags stripped).
    """
    _ensure_agent_on_path()
    from src.core.loop import run_agent_loop  # type: ignore[import-untyped]
    from src.llm.client import LLMClient  # type: ignore[import-untyped]
    from src.tools.registry import ToolRegistry  # type: ignore[import-untyped]
    from src.output.jsonl import set_event_callback  # type: ignore[import-untyped]

    effective_model = model or (CHAT_MODEL if readonly else AGENT_MODEL)
    effective_cwd = cwd or WORKSPACE
    config = _make_config(readonly=readonly)

    llm = LLMClient(
        model=effective_model,
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        timeout=300.0,
    )

    tools = ToolRegistry(cwd=Path(effective_cwd))
    if not readonly:
        _register_tau_tools(tools)

    # Minimal AgentContext (duck-typed for run_agent_loop)
    class _Ctx:
        def __init__(self):
            self.instruction = prompt
            self.cwd = effective_cwd
            self.is_done = False
            self._start = time.time()

        @property
        def elapsed_secs(self):
            return time.time() - self._start

        def shell(self, cmd, timeout=120):
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                               timeout=timeout, cwd=self.cwd)
            class _R:
                def __init__(self, out, rc, stdout="", stderr=""):
                    self.output = out
                    self.exit_code = rc
                    self.stdout = stdout
                    self.stderr = stderr
            return _R(r.stdout + r.stderr, r.returncode, r.stdout, r.stderr)

        def done(self):
            self.is_done = True

    ctx = _Ctx()

    # If a custom system prompt was given, monkey-patch get_system_prompt
    # in the loop's import namespace so it uses ours.
    _orig_get_system_prompt = None
    if system_prompt_override:
        import src.core.loop as _loop_mod  # type: ignore[import-untyped]
        _orig_get_system_prompt = _loop_mod.get_system_prompt
        _loop_mod.get_system_prompt = lambda cwd=None, shell=None, **kw: system_prompt_override

    # Capture the last agent message from events
    captured_messages: list[str] = []

    def _capture(event: Dict[str, Any]) -> None:
        if event.get("type") == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message" and item.get("text"):
                captured_messages.append(item["text"])

    set_event_callback(_capture)

    try:
        run_agent_loop(llm=llm, tools=tools, ctx=ctx, config=config)
    except Exception as e:
        return f"Error: {e}"
    finally:
        set_event_callback(None)
        llm.close()
        if _orig_get_system_prompt is not None:
            import src.core.loop as _loop_mod  # type: ignore[import-untyped]
            _loop_mod.get_system_prompt = _orig_get_system_prompt

    final = captured_messages[-1] if captured_messages else ""
    return strip_think_tags(final)


# ---------------------------------------------------------------------------
# 3) run_baseagent_streaming — threaded agent with event queue
# ---------------------------------------------------------------------------

def run_baseagent_streaming(
    prompt: str,
    *,
    model: str | None = None,
    readonly: bool = False,
    cwd: str | None = None,
    system_prompt_override: str | None = None,
    event_queue: queue.Queue | None = None,
    timeout_seconds: int = 3600,
) -> threading.Thread:
    """Run the baseagent loop in a background thread.

    JSONL event dicts are pushed to *event_queue*.  A ``None`` sentinel is
    pushed when the loop finishes.  Returns the started ``Thread``.
    """
    if event_queue is None:
        raise ValueError("event_queue is required")

    _ensure_agent_on_path()

    def _worker():
        from src.core.loop import run_agent_loop  # type: ignore[import-untyped]
        from src.llm.client import LLMClient  # type: ignore[import-untyped]
        from src.tools.registry import ToolRegistry  # type: ignore[import-untyped]
        from src.output.jsonl import set_event_callback  # type: ignore[import-untyped]

        effective_model = model or (CHAT_MODEL if readonly else AGENT_MODEL)
        effective_cwd = cwd or WORKSPACE
        config = _make_config(readonly=readonly)

        llm = LLMClient(
            model=effective_model,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            timeout=300.0,
        )

        tools = ToolRegistry(cwd=Path(effective_cwd))
        if not readonly:
            _register_tau_tools(tools)

        class _Ctx:
            def __init__(self):
                self.instruction = prompt
                self.cwd = effective_cwd
                self.is_done = False
                self._start = time.time()

            @property
            def elapsed_secs(self):
                return time.time() - self._start

            def shell(self, cmd, timeout=120):
                r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                   timeout=timeout, cwd=self.cwd)
                class _R:
                    def __init__(self, out, rc, stdout="", stderr=""):
                        self.output = out
                        self.exit_code = rc
                        self.stdout = stdout
                        self.stderr = stderr
                return _R(r.stdout + r.stderr, r.returncode, r.stdout, r.stderr)

            def done(self):
                self.is_done = True

        ctx = _Ctx()

        # Monkey-patch system prompt if override given
        _orig = None
        if system_prompt_override:
            import src.core.loop as _lm  # type: ignore[import-untyped]
            _orig = _lm.get_system_prompt
            _lm.get_system_prompt = lambda cwd=None, shell=None, **kw: system_prompt_override

        def _emit_to_queue(event_dict):
            event_queue.put(event_dict)

        set_event_callback(_emit_to_queue)

        try:
            run_agent_loop(llm=llm, tools=tools, ctx=ctx, config=config)
        except Exception as e:
            event_queue.put({"type": "error", "message": str(e)})
        finally:
            set_event_callback(None)
            llm.close()
            if _orig is not None:
                import src.core.loop as _lm  # type: ignore[import-untyped]
                _lm.get_system_prompt = _orig
            event_queue.put(None)  # sentinel

    t = threading.Thread(target=_worker, daemon=True, name="BaseAgentWorker")
    t.start()
    return t


# ---------------------------------------------------------------------------
# JSONL event normaliser (kept for backward compatibility)
# ---------------------------------------------------------------------------
# BaseAgent emits JSONL events in this format:
#   {"type": "item.started",   "item": {"type": "agent_message", ...}}
#   {"type": "item.completed", "item": {"type": "command_execution", ...}}
#   {"type": "turn.completed", "usage": {...}}


def parse_event(event: dict[str, Any]) -> dict[str, Any] | None:
    """Normalise a JSONL event into an internal dict.

    Returns ``None`` for events we don't care about.
    """
    etype = event.get("type", "")
    item = event.get("item") or {}
    item_type = item.get("type", "")

    # -- Item started --
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
            return None
        return {"kind": "tool_start", "text": None, "tool_name": item_type, "tool_detail": None, "item_type": item_type}

    # -- Item updated --
    if etype == "item.updated":
        if item_type == "agent_message":
            return {"kind": "message", "text": strip_think_tags(item.get("text", "")), "tool_name": None, "tool_detail": None, "item_type": item_type}
        if item_type == "command_execution":
            return {"kind": "tool_start", "text": None, "tool_name": "Shell", "tool_detail": item.get("aggregated_output", ""), "item_type": item_type}
        if item_type == "reasoning":
            return {"kind": "message", "text": item.get("text", ""), "tool_name": None, "tool_detail": None, "item_type": item_type}
        return None

    # -- Item completed --
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
        return {"kind": "tool_done", "text": None, "tool_name": item_type, "tool_detail": None, "item_type": item_type}

    # -- Turn events --
    if etype == "turn.completed":
        return {"kind": "turn_done", "text": None, "tool_name": None, "tool_detail": None, "item_type": None}

    if etype == "turn.failed":
        err = event.get("error") or {}
        return {"kind": "error", "text": err.get("message", "Turn failed"), "tool_name": None, "tool_detail": None, "item_type": None}

    if etype == "error":
        return {"kind": "error", "text": event.get("message", "Unknown error"), "tool_name": None, "tool_detail": None, "item_type": None}

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


# ---------------------------------------------------------------------------
# Backward-compat shim: build_cmd