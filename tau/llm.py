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
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Model mapping — env-var overrides with sensible defaults
# ---------------------------------------------------------------------------

# Lightweight model for ask/chat/plan/summary (read-only queries)
CHAT_MODEL = os.getenv(
    "TAU_CHAT_MODEL",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
)

# Coding agent model for /adapt, agent loop, cron jobs (writes code + runs tools)
AGENT_MODEL = os.getenv(
    "TAU_AGENT_MODEL",
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


# ---------------------------------------------------------------------------
# Cached singletons — reuse httpx connection pool across requests
# ---------------------------------------------------------------------------
import threading as _threading

_llm_client_lock = _threading.Lock()
_llm_client_cache: Dict[str, Any] = {}  # model -> LLMClient

_tool_registry_lock = _threading.Lock()
_tool_registry_cache: Dict[str, Any] = {}  # key -> ToolRegistry


def _get_cached_llm_client(model: str, timeout: float = 300.0):
    """Return a cached LLMClient, creating one if needed.

    Reuses the underlying httpx connection pool to avoid
    ~100-300 ms TCP/TLS setup on every request.
    """
    with _llm_client_lock:
        if model in _llm_client_cache:
            return _llm_client_cache[model]
        client = _make_llm_client(model, timeout=timeout)
        _llm_client_cache[model] = client
        return client


def _get_cached_tool_registry(cwd: str, *, register_tau: bool = False):
    """Return a cached ToolRegistry, creating one if needed."""
    cache_key = f"{cwd}:{register_tau}"
    with _tool_registry_lock:
        if cache_key in _tool_registry_cache:
            return _tool_registry_cache[cache_key]
        _ensure_agent_on_path()
        from src.tools.registry import ToolRegistry  # type: ignore[import-untyped]
        tools = ToolRegistry(cwd=Path(cwd))
        if register_tau:
            _register_tau_tools(tools)
        _tool_registry_cache[cache_key] = tools
        return tools


def _make_config(readonly: bool = False, chat_mode: bool = False) -> Dict[str, Any]:
    """Build the config dict consumed by ``run_agent_loop``.

    Only keys actually read by ``loop.py`` are included here.
    Context-management constants (model_context_limit, prune thresholds,
    etc.) live in ``agent/src/core/compaction.py`` as module-level defaults.

    When *chat_mode* is True, the config is tuned for conversational
    Telegram messages: fewer iterations, skip filesystem state injection,
    and streaming enabled.
    """
    if chat_mode:
        return {
            "model": AGENT_MODEL,
            "provider": "chutes",
            "reasoning_effort": "none",
            "max_tokens": 16384,
            "temperature": 0.0,
            "max_iterations": 30,
            "max_output_tokens": 2500,
            "shell_timeout": 60,
            "cache_enabled": True,
            "skip_initial_state": True,
            "skip_verification": True,
            "streaming": True,
        }
    return {
        "model": AGENT_MODEL,
        "provider": "chutes",
        "reasoning_effort": "none",
        "max_tokens": 16384,
        "temperature": 0.0,
        "max_iterations": 200 if not readonly else 50,
        "max_output_tokens": 2500,
        "shell_timeout": 60,
        "cache_enabled": True,
    }


# ---------------------------------------------------------------------------
# Skill loader — keyword-match skill files from context/skills/
# ---------------------------------------------------------------------------

# Keyword → skill file mapping (ported from .cursor/rules/context-system.mdc)
_SKILL_KEYWORDS: Dict[str, str] = {
    "lium": "lium-skills.md",
    "gpu": "lium-skills.md",
    "rent": "lium-skills.md",
    "instance": "lium-skills.md",
    "eve": "eve-skills.md",
    "creative": "eve-skills.md",
    "art": "eve-skills.md",
    "image": "eve-skills.md",
    "agent": "agent.md",
    "cursor": "agent.md",
    "cli": "agent.md",
    "basilica": "basilica.md",
    "targon": "targon.md",
    "remind": "self-scheduling.md",
    "reminder": "self-scheduling.md",
    "schedule": "self-scheduling.md",
    "cron": "self-scheduling.md",
    "later": "self-scheduling.md",
    "task": "self-scheduling.md",
    "self-message": "self-scheduling.md",
    "follow-up": "self-scheduling.md",
    "checkpoint": "self-scheduling.md",
    "defer": "self-scheduling.md",
}


def _load_skills_for_message(message: str) -> str:
    """Return concatenated skill file contents matching keywords in *message*.

    Uses the same keyword-to-skill mapping as ``.cursor/rules/context-system.mdc``
    but works at runtime (outside of Cursor IDE).
    """
    skills_dir = os.path.join(WORKSPACE, "context", "skills")
    if not os.path.isdir(skills_dir):
        return ""

    message_lower = message.lower()
    matched_files: set[str] = set()

    for keyword, filename in _SKILL_KEYWORDS.items():
        if keyword in message_lower:
            matched_files.add(filename)

    if not matched_files:
        return ""

    parts: list[str] = []
    for filename in sorted(matched_files):
        filepath = os.path.join(skills_dir, filename)
        if os.path.isfile(filepath):
            content = _read_file_cached(filepath, ttl=120.0)  # skills change rarely
            if content:
                parts.append(f"## Skill: {filename}\n\n{content}")

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# TTL file cache — avoids re-reading unchanged context files on every message
# ---------------------------------------------------------------------------

_file_cache: Dict[str, tuple[float, str]] = {}  # filepath -> (timestamp, content)
_FILE_CACHE_TTL = 30.0  # seconds; identity/skills rarely change


def _read_file_cached(filepath: str, ttl: float = _FILE_CACHE_TTL) -> str:
    """Read a file with in-memory caching.  Returns '' on error."""
    now = time.time()
    cached = _file_cache.get(filepath)
    if cached is not None:
        ts, content = cached
        if now - ts < ttl:
            return content
    try:
        content = Path(filepath).read_text(encoding="utf-8").strip()
    except Exception:
        content = ""
    _file_cache[filepath] = (now, content)
    return content


# ---------------------------------------------------------------------------
# Tau system prompt builder
# ---------------------------------------------------------------------------

def _read_context_file(relative_path: str, max_lines: int | None = None) -> str:
    """Read a file relative to the workspace root, returning '' on error."""
    filepath = os.path.join(WORKSPACE, relative_path)
    text = _read_file_cached(filepath)
    if max_lines is not None and text:
        lines = text.split("\n")
        text = "\n".join(lines[-max_lines:])
    return text


def build_tau_system_prompt(
    chat_id: int | str | None = None,
    user_message: str = "",
    chat_lines: int = 50,
) -> str:
    """Build a Tau-aware system prompt for conversational Telegram use.

    Loads identity, chat summary, recent chat history, core memory,
    and any skills matched by the user's message.

    Args:
        chat_id: Telegram chat ID (for per-chat history). None uses default.
        user_message: The user's raw message text (for skill matching).
        chat_lines: How many lines of recent chat to include.

    Returns:
        Complete system prompt string.
    """
    # --- Identity ---
    identity = _read_context_file("context/IDENTITY.md")

    # --- Chat summary (hourly auto-updated) ---
    chat_summary = _read_context_file("context/CHAT_SUMMARY.md")

    # --- Recent chat history ---
    recent_chat = ""
    try:
        from tau.telegram import get_chat_history
        if chat_id is not None:
            recent_chat = get_chat_history(chat_id=chat_id, max_lines=chat_lines)
        else:
            recent_chat = get_chat_history(max_lines=chat_lines)
    except Exception:
        pass

    # --- Core memory ---
    core_memory = _read_context_file("context/memory/CORE_MEMORY.md", max_lines=30)

    # --- Mid-term memory ---
    mid_term = _read_context_file("context/memory/MID_TERM.md", max_lines=20)

    # --- Skills (keyword-matched) ---
    skills_section = _load_skills_for_message(user_message) if user_message else ""

    # --- Assemble ---
    parts: list[str] = []

    parts.append(
        "You are Tau, an autonomous AI assistant running as a Telegram bot.\n"
        "You are NOT a generic LLM. You are Tau — a specific agent with persistent "
        "state, memory, skills, and the ability to take real actions."
    )

    if identity:
        parts.append(identity)

    if chat_summary:
        parts.append(f"# Conversation Summary (auto-updated hourly)\n\n{chat_summary}")

    if core_memory:
        parts.append(f"# Core Memory (persistent facts)\n\n{core_memory}")

    if mid_term:
        parts.append(f"# Mid-Term Memory (recent weeks)\n\n{mid_term}")

    if recent_chat:
        parts.append(f"# Recent Chat History\n\n{recent_chat}")

    if skills_section:
        parts.append(f"# Relevant Skills\n\n{skills_section}")

    parts.append("""# Tools Available

You have the following tools:
- **send_message** — Send a Telegram message to the user
- **send_voice** — Send a TTS voice message
- **create_task** — Create a task for yourself to process later
- **schedule_message** — Schedule a future message (--in '2h', --at '14:00', --cron '0 9 * * *')
- **search_skills** — Search the creative AI skills catalog
- **commands** — Execute any bot command (task, plan, status, adapt, cron, etc.)
- **shell_command** — Run shell commands
- **read_file** / **write_file** / **apply_patch** — File operations
- **grep_files** / **glob_files** / **list_dir** — Search and explore files
- **web_search** — Search the web for information

# Guidelines

- Answer directly, be concise. No preamble, no filler.
- For simple questions (factual, conversational), just answer — don't run shell commands unnecessarily.
- When the user asks for actions (reminders, tasks, searches, code changes), use the appropriate tools.
- For coding tasks, use your full file editing capability.
- Do NOT say "Is there anything else..." or similar closing phrases.
- Do NOT explain your thinking process in the response.
- Strip any <think>...</think> blocks from your output.
- If asked "what are you?" or "who are you?", say you are Tau.
- Never identify as Composer, Cursor, ChatGPT, Claude, or any other AI system.
- When asked what you can do, explain conversationally — no command syntax, just natural examples.""")

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Register Tau-specific tools on a ToolRegistry
# ---------------------------------------------------------------------------

def _register_tau_tools(tools) -> None:
    """Register Tau-specific tools on *tools* (a ``ToolRegistry``).

    Tools are called **in-process** via direct Python function calls,
    avoiding the ~300-800 ms overhead of spawning a subprocess per tool.
    """
    _ensure_agent_on_path()
    from src.tools.base import ToolResult  # type: ignore[import-untyped]

    # ------------------------------------------------------------------
    # send_message  (in-process: tau.telegram.notify)
    # ------------------------------------------------------------------
    def _handle_send_message(args: dict) -> ToolResult:
        try:
            from tau.telegram import notify
            message = args.get("message", "")
            notify(message)
            label = f"{message[:50]}..." if len(message) > 50 else message
            return ToolResult(success=True, output=f"Message sent: {label}")
        except Exception as e:
            return ToolResult(success=False, output=str(e))

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
        _handle_send_message,
    )

    # ------------------------------------------------------------------
    # send_voice  (in-process: tau.tools.send_voice)
    # ------------------------------------------------------------------
    def _handle_send_voice(args: dict) -> ToolResult:
        try:
            from tau.tools.send_voice import send_voice_message
            from tau.telegram import get_chat_id
            chat_id = get_chat_id()
            if not chat_id:
                return ToolResult(success=False, output="No chat ID found.")
            text = args.get("message", "")
            send_voice_message(chat_id, text)
            return ToolResult(success=True, output=f"Voice message sent: {text}")
        except Exception as e:
            return ToolResult(success=False, output=str(e))

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
        _handle_send_voice,
    )

    # ------------------------------------------------------------------
    # search_skills  (in-process: tau.tools.search_skills)
    # ------------------------------------------------------------------
    def _handle_search_skills(args: dict) -> ToolResult:
        try:
            from tau.tools.search_skills import search_skills, format_skill_list
            query = args.get("query") or None
            category = args.get("category") or None
            results = search_skills(query, category)
            return ToolResult(success=True, output=format_skill_list(results))
        except Exception as e:
            return ToolResult(success=False, output=str(e))

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
        _handle_search_skills,
    )

    # ------------------------------------------------------------------
    # create_task  (in-process: tau.tools.create_task)
    # ------------------------------------------------------------------
    def _handle_create_task(args: dict) -> ToolResult:
        try:
            from tau.tools.create_task import create_task
            title = args.get("title", "")
            description = args.get("description", "")
            task_id = create_task(title, description)
            return ToolResult(success=True, output=f"Created {task_id}: {title}")
        except Exception as e:
            return ToolResult(success=False, output=str(e))

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
        _handle_create_task,
    )

    # ------------------------------------------------------------------
    # schedule_message  (in-process: tau.tools.schedule_message)
    # ------------------------------------------------------------------
    def _handle_schedule_message(args: dict) -> ToolResult:
        try:
            from tau.tools.schedule_message import schedule_at, schedule_cron
            message = args.get("message", "")
            delay = args.get("delay")
            at_time = args.get("at")
            cron_expr = args.get("cron")

            if at_time:
                schedule_at(at_time, message)
                return ToolResult(success=True, output=f"Scheduled at {at_time}: {message}")
            elif delay:
                time_spec = f"now + {delay.replace('h', ' hours').replace('m', ' minutes')}"
                schedule_at(time_spec, message)
                return ToolResult(success=True, output=f"Scheduled in {delay}: {message}")
            elif cron_expr:
                schedule_cron(cron_expr, message)
                return ToolResult(success=True, output=f"Added cron job ({cron_expr}): {message}")
            else:
                return ToolResult(success=False, output="Specify --delay, --at, or --cron")
        except Exception as e:
            return ToolResult(success=False, output=str(e))

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
        _handle_schedule_message,
    )

    # ------------------------------------------------------------------
    # commands  (in-process: tau.tools.commands.run_command)
    # ------------------------------------------------------------------
    def _handle_commands(args: dict) -> ToolResult:
        try:
            from tau.tools.commands import run_command
            command = args.get("command", "")
            cmd_args = args.get("args", "")
            # Split args string into list if non-empty
            arg_list = cmd_args.split() if cmd_args else []
            result = run_command(command, *arg_list)
            return ToolResult(success=True, output=result)
        except Exception as e:
            return ToolResult(success=False, output=str(e))

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
        _handle_commands,
    )


# ---------------------------------------------------------------------------
# 1) llm_chat — simple text completion (no tools)
# ---------------------------------------------------------------------------

def llm_chat(prompt: str, *, model: str | None = None, timeout: float = 300.0) -> str:
    """Run a simple text completion — no tools, no agent loop.

    Used for summaries, fact extraction, and other pure-LLM tasks.
    Returns the raw text response (think-tags stripped).
    """
    llm = _get_cached_llm_client(model or CHAT_MODEL, timeout=timeout)
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
    from src.core.session import SimpleAgentContext  # type: ignore[import-untyped]
    from src.output.jsonl import set_event_callback  # type: ignore[import-untyped]

    effective_model = model or (CHAT_MODEL if readonly else AGENT_MODEL)
    effective_cwd = cwd or WORKSPACE
    config = _make_config(readonly=readonly)

    llm = _get_cached_llm_client(effective_model, timeout=300.0)
    tools = _get_cached_tool_registry(effective_cwd, register_tau=not readonly)

    # Use the proper AgentContext from the agent package
    ctx = SimpleAgentContext(instruction=prompt, cwd=effective_cwd)

    # Capture the last agent message from events
    captured_messages: list[str] = []

    def _capture(event: Dict[str, Any]) -> None:
        if event.get("type") == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message" and item.get("text"):
                captured_messages.append(item["text"])

    set_event_callback(_capture)

    try:
        run_agent_loop(
            llm=llm,
            tools=tools,
            ctx=ctx,
            config=config,
            system_prompt=system_prompt_override,
        )
    except Exception as e:
        return f"Error: {e}"
    finally:
        set_event_callback(None)

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
    chat_mode: bool = False,
) -> threading.Thread:
    """Run the baseagent loop in a background thread.

    JSONL event dicts are pushed to *event_queue*.  A ``None`` sentinel is
    pushed when the loop finishes.  Returns the started ``Thread``.

    When *chat_mode* is True, the config is tuned for conversational
    Telegram messages (fewer iterations, no initial filesystem state).
    """
    if event_queue is None:
        raise ValueError("event_queue is required")

    _ensure_agent_on_path()

    def _worker():
        from src.core.loop import run_agent_loop  # type: ignore[import-untyped]
        from src.core.session import SimpleAgentContext  # type: ignore[import-untyped]
        from src.output.jsonl import set_event_callback  # type: ignore[import-untyped]

        effective_model = model or (CHAT_MODEL if readonly else AGENT_MODEL)
        effective_cwd = cwd or WORKSPACE
        config = _make_config(readonly=readonly, chat_mode=chat_mode)

        llm = _get_cached_llm_client(effective_model, timeout=300.0)
        tools = _get_cached_tool_registry(effective_cwd, register_tau=not readonly)

        ctx = SimpleAgentContext(instruction=prompt, cwd=effective_cwd)

        def _emit_to_queue(event_dict):
            event_queue.put(event_dict)

        set_event_callback(_emit_to_queue)

        try:
            run_agent_loop(
                llm=llm,
                tools=tools,
                ctx=ctx,
                config=config,
                system_prompt=system_prompt_override,
            )
        except Exception as e:
            event_queue.put({"type": "error", "message": str(e)})
        finally:
            set_event_callback(None)
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