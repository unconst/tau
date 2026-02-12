"""Tool registry for SuperAgent - manages and dispatches tool calls."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from src.tools.base import ToolResult
from src.tools.guards import GuardConfig, GuardError, PathGuards
from src.tools.hashline import format_hashline, format_lines, line_hash
from src.tools.specs import get_all_tools, tool_is_mutating, tool_supports_parallel

if TYPE_CHECKING:
    pass  # AgentContext is duck-typed (has shell(), cwd, etc.)

_LIST_DIR_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", ".env",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
    ".eggs", ".cache", ".DS_Store", "dist", "build", ".agent",
})


@dataclass
class ExecutorConfig:
    """Configuration for tool execution."""

    max_concurrent: int = 4
    default_timeout: float = 120.0
    cache_enabled: bool = True
    cache_ttl: float = 300.0  # 5 minutes


@dataclass
class CachedResult:
    """A cached tool result with timestamp."""

    result: ToolResult
    cached_at: float  # timestamp from time.time()

    def is_valid(self, ttl: float) -> bool:
        """Check if the cached result is still valid."""
        return (time.time() - self.cached_at) < ttl


@dataclass
class ToolStats:
    """Per-tool execution statistics."""

    executions: int = 0
    successes: int = 0
    total_ms: int = 0

    def success_rate(self) -> float:
        """Get the success rate for this tool."""
        if self.executions == 0:
            return 0.0
        return self.successes / self.executions

    def avg_ms(self) -> float:
        """Get average execution time in milliseconds."""
        if self.executions == 0:
            return 0.0
        return self.total_ms / self.executions


@dataclass
class ExecutorStats:
    """Aggregate execution statistics."""

    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    cache_hits: int = 0
    total_duration_ms: int = 0
    by_tool: Dict[str, ToolStats] = field(default_factory=dict)

    def success_rate(self) -> float:
        """Get overall success rate."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions

    def cache_hit_rate(self) -> float:
        """Get cache hit rate."""
        if self.total_executions == 0:
            return 0.0
        return self.cache_hits / self.total_executions

    def avg_duration_ms(self) -> float:
        """Get average execution duration in milliseconds."""
        if self.total_executions == 0:
            return 0.0
        return self.total_duration_ms / self.total_executions


class _RWLock:
    """A simple read-write lock for tool execution.

    - Multiple readers (parallel-safe tools) can hold the lock simultaneously.
    - A writer (mutating tool) gets exclusive access — blocks until all
      readers release, and blocks new readers until it's done.

    This prevents race conditions where e.g. two concurrent shell_command
    calls write to the same file.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._readers = 0
        self._read_ok = threading.Condition(self._lock)
        self._write_ok = threading.Condition(self._lock)
        self._writers_waiting = 0
        self._writer_active = False

    def acquire_read(self) -> None:
        with self._lock:
            while self._writer_active or self._writers_waiting > 0:
                self._read_ok.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._lock:
            self._readers -= 1
            if self._readers == 0:
                self._write_ok.notify_all()

    def acquire_write(self) -> None:
        with self._lock:
            self._writers_waiting += 1
            while self._writer_active or self._readers > 0:
                self._write_ok.wait()
            self._writers_waiting -= 1
            self._writer_active = True

    def release_write(self) -> None:
        with self._lock:
            self._writer_active = False
            self._read_ok.notify_all()
            self._write_ok.notify_all()


class ToolRegistry:
    """Registry for managing and dispatching tool calls.

    Tools receive AgentContext for shell execution.
    Includes caching and execution statistics.
    """

    def __init__(
        self,
        cwd: Optional[Path] = None,
        config: Optional[ExecutorConfig] = None,
        allowed_tools: Optional[set[str]] = None,
        allow_subagent_spawn: bool = True,
    ):
        """Initialize the registry.

        Args:
            cwd: Current working directory for tools (optional, can be set later)
            config: Executor configuration (optional, uses defaults)
        """
        self.cwd = cwd or Path("/app")
        self._plan: list[dict[str, str]] = []
        self._plan_last_updated_iteration: int = 0
        self._config = config or ExecutorConfig()
        self._cache: Dict[str, CachedResult] = {}
        self._stats = ExecutorStats()
        self._guards = PathGuards(GuardConfig.from_paths(self.cwd))
        self._allowed_tools = set(allowed_tools) if allowed_tools is not None else None
        self._allow_subagent_spawn = allow_subagent_spawn
        # Custom tools registered at runtime (name -> (spec_dict, handler_callable))
        self._custom_tools: Dict[str, tuple[Dict[str, Any], Callable[..., ToolResult]]] = {}
        # Read/write lock — parallel-safe tools take a read lock;
        # mutating tools take a write lock (exclusive).
        self._rwlock = _RWLock()
        # ask_user callback and pending responses (instance-level to avoid sharing)
        self._ask_user_callback: Optional[Callable] = None
        self._pending_asks: Dict[str, Dict[str, Any]] = {}

    # -----------------------------------------------------------------
    # Custom tool registration
    # -----------------------------------------------------------------

    def register_tool(
        self,
        name: str,
        spec: Dict[str, Any],
        handler: Callable[..., ToolResult],
    ) -> None:
        """Register a custom tool.

        Args:
            name: Tool name (must be unique).
            spec: OpenAI-compatible tool specification dict
                  (needs ``name``, ``description``, ``parameters``).
            handler: Callable that accepts ``(arguments: dict) -> ToolResult``.
        """
        self._custom_tools[name] = (spec, handler)

    def execute(
        self,
        ctx: "AgentContext",
        name: str,
        arguments: dict[str, Any],
        enforce_guards: bool = True,
    ) -> ToolResult:
        """Execute a tool by name.

        Args:
            ctx: Agent context with shell() method
            name: Tool name
            arguments: Tool arguments

        Returns:
            ToolResult from the tool execution
        """
        start_time = time.time()

        # Check cache first if enabled
        if self._config.cache_enabled:
            cache_key = self._cache_key(name, arguments)
            cached = self._get_cached(cache_key)
            if cached is not None:
                duration_ms = int((time.time() - start_time) * 1000)
                self._record_execution(name, duration_ms, success=True, cached=True)
                return cached

        cwd = Path(ctx.cwd) if hasattr(ctx, "cwd") else self.cwd

        try:
            max_attempts = 1 if tool_is_mutating(name, arguments) else 3
            timeout_seconds = self._config.default_timeout
            if name == "shell_command":
                timeout_seconds = max(
                    timeout_seconds,
                    float(max(1, int(arguments.get("timeout_ms", 60000)) // 1000)),
                )

            result = None
            last_error_text = ""
            for attempt in range(1, max_attempts + 1):
                try:
                    result = self._execute_once(ctx, cwd, name, arguments, enforce_guards)
                    if result.success or attempt >= max_attempts:
                        break
                    last_error_text = (result.error or result.output or "").lower()
                    if not self._is_transient_failure(last_error_text):
                        break
                except FuturesTimeoutError:
                    last_error_text = "execution timed out"
                    if attempt >= max_attempts:
                        timeout_msg = f"Tool {name} timed out after {timeout_seconds:.1f}s"
                        if name == "spawn_subagent":
                            result = ToolResult.fail(
                                timeout_msg,
                                output=f"error_code=tool_timeout timeout_seconds={timeout_seconds:.1f}",
                            )
                        else:
                            result = ToolResult.fail(timeout_msg)
                        break
                except Exception as e:
                    last_error_text = str(e).lower()
                    if attempt >= max_attempts or not self._is_transient_failure(last_error_text):
                        result = ToolResult.fail(f"Tool {name} failed: {e}")
                        break
                time.sleep(min(2.0, 0.25 * attempt))

            if result is None:
                result = ToolResult.fail(f"Tool {name} failed after retries")

        except GuardError as e:
            result = ToolResult.fail(str(e))
        except Exception as e:
            result = ToolResult.fail(f"Tool {name} failed: {e}")

        # Record execution stats
        duration_ms = int((time.time() - start_time) * 1000)
        self._record_execution(name, duration_ms, success=result.success, cached=False)

        # Cache successful results
        if self._config.cache_enabled and result.success:
            cache_key = self._cache_key(name, arguments)
            self._cache_result(cache_key, result)

        return result

    def _execute_once(
        self,
        ctx: "AgentContext",
        cwd: Path,
        name: str,
        arguments: dict[str, Any],
        enforce_guards: bool,
    ) -> ToolResult:
        """Execute one tool call with timeout handling."""
        if self._allowed_tools is not None and name not in self._allowed_tools:
            return ToolResult.fail(f"Tool '{name}' is not available in this runtime.")
        if name == "spawn_subagent" and not self._allow_subagent_spawn:
            return ToolResult.fail("Nested subagents are not allowed.")
        if name in self._custom_tools:
            _, handler = self._custom_tools[name]
            return self._run_with_timeout(lambda: handler(arguments), self._config.default_timeout)
        if name == "shell_command":
            return self._execute_shell(ctx, cwd, arguments, enforce_guards)
        if name == "read_file":
            return self._run_with_timeout(
                lambda: self._execute_read_file(cwd, arguments, enforce_guards),
                self._config.default_timeout,
            )
        if name == "write_file":
            return self._run_with_timeout(
                lambda: self._execute_write_file(cwd, arguments, enforce_guards),
                self._config.default_timeout,
            )
        if name == "list_dir":
            return self._run_with_timeout(
                lambda: self._execute_list_dir(cwd, arguments, enforce_guards),
                self._config.default_timeout,
            )
        if name == "grep_files":
            return self._run_with_timeout(
                lambda: self._execute_grep(ctx, cwd, arguments, enforce_guards),
                self._config.default_timeout,
            )
        if name == "apply_patch":
            return self._run_with_timeout(
                lambda: self._execute_apply_patch(cwd, arguments, enforce_guards),
                self._config.default_timeout,
            )
        if name == "view_image":
            return self._run_with_timeout(
                lambda: self._execute_view_image(cwd, arguments, enforce_guards),
                self._config.default_timeout,
            )
        if name == "update_plan":
            return self._run_with_timeout(
                lambda: self._execute_update_plan(arguments),
                self._config.default_timeout,
            )
        if name == "web_search":
            return self._run_with_timeout(
                lambda: self._execute_web_search(arguments),
                self._config.default_timeout,
            )
        if name == "spawn_subagent":
            requested = float(arguments.get("timeout_seconds", 0) or 0)
            effective_timeout = max(self._config.default_timeout, requested + 10.0) if requested > 0 else self._config.default_timeout
            return self._run_with_timeout(
                lambda: self._execute_subagent(ctx, cwd, arguments),
                effective_timeout,
            )
        if name == "spawn_comparison":
            # Comparison runs multiple subagents — allow generous timeout
            num_approaches = len(arguments.get("approaches", []))
            effective_timeout = max(self._config.default_timeout, 150.0 * num_approaches)
            return self._run_with_timeout(
                lambda: self._execute_comparison(ctx, cwd, arguments),
                effective_timeout,
            )
        if name == "str_replace":
            return self._run_with_timeout(
                lambda: self._execute_str_replace(cwd, arguments, enforce_guards),
                self._config.default_timeout,
            )
        if name == "hashline_edit":
            return self._run_with_timeout(
                lambda: self._execute_hashline_edit(cwd, arguments, enforce_guards),
                self._config.default_timeout,
            )
        if name == "glob_files":
            return self._run_with_timeout(
                lambda: self._execute_glob_files(cwd, arguments, enforce_guards),
                self._config.default_timeout,
            )
        if name == "lint":
            return self._run_with_timeout(
                lambda: self._execute_lint(cwd, arguments, enforce_guards),
                self._config.default_timeout,
            )
        if name == "ask_user":
            # ask_user blocks for up to 5 minutes waiting for user reply
            return self._run_with_timeout(
                lambda: self._execute_ask_user(arguments),
                timeout_seconds=310.0,  # slightly more than the 5min wait
            )
        return ToolResult.fail(f"Unknown tool: {name}")

    @staticmethod
    def _is_transient_failure(text: str) -> bool:
        return any(
            marker in text
            for marker in (
                "timeout",
                "timed out",
                "temporarily unavailable",
                "connection reset",
                "connection aborted",
                "rate limit",
                "429",
                "503",
                "502",
                "500",
            )
        )

    @staticmethod
    def _run_with_timeout(fn: Callable[[], ToolResult], timeout_seconds: float) -> ToolResult:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fn)
            return future.result(timeout=timeout_seconds)

    def _execute_shell(
        self,
        ctx: "AgentContext",
        cwd: Path,
        args: dict[str, Any],
        enforce_guards: bool,
    ) -> ToolResult:
        """Execute shell command using subprocess directly."""
        command = args.get("command", "")
        workdir = args.get("workdir")
        timeout_ms = args.get("timeout_ms", 60000)

        if not command:
            return ToolResult.fail("No command provided")

        # Resolve working directory
        effective_cwd = cwd
        if workdir:
            wd = Path(workdir)
            effective_cwd = wd if wd.is_absolute() else cwd / wd
        if enforce_guards:
            self._guards.require_read(effective_cwd)
            lower = command.lower()
            mutating_hint = any(
                token in lower
                for token in (
                    "rm ",
                    "mv ",
                    "cp ",
                    "git commit",
                    "git push",
                    "apply_patch",
                    ">>",
                    ">",
                    "touch ",
                    "mkdir ",
                )
            )
            if mutating_hint:
                self._guards.require_mutation_allowed("shell_command")

        timeout_sec = max(1, timeout_ms // 1000)

        try:
            result = subprocess.run(
                ["sh", "-c", command],
                cwd=str(effective_cwd),
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )

            output = result.stdout
            if result.stderr:
                output += f"\n{result.stderr}"

            if result.returncode != 0:
                # Auto-recover: if `python` is not found, retry with `python3`.
                # This is extremely common on macOS/Darwin where `python` doesn't
                # exist by default.  Saves a full LLM round-trip that would
                # otherwise be spent discovering the right binary name.
                if result.returncode == 127 and "python" in output.lower() and "not found" in output.lower():
                    rewritten = self._rewrite_python_command(command)
                    if rewritten and rewritten != command:
                        retry = subprocess.run(
                            ["sh", "-c", rewritten],
                            cwd=str(effective_cwd),
                            capture_output=True,
                            text=True,
                            timeout=timeout_sec,
                        )
                        retry_output = retry.stdout
                        if retry.stderr:
                            retry_output += f"\n{retry.stderr}"
                        if retry.returncode == 0:
                            return ToolResult.ok(retry_output.strip())
                        # Retry also failed — return the retry error
                        return ToolResult.fail(
                            f"Command failed with exit code {retry.returncode}",
                            output=(retry_output + f"\n[exit code: {retry.returncode}]").strip(),
                        )

                return ToolResult.fail(
                    f"Command failed with exit code {result.returncode}",
                    output=(output + f"\n[exit code: {result.returncode}]").strip(),
                )

            return ToolResult.ok(output.strip())

        except subprocess.TimeoutExpired:
            return ToolResult.fail(f"Command timed out after {timeout_sec}s")
        except Exception as e:
            return ToolResult.fail(str(e))

    @staticmethod
    def _rewrite_python_command(command: str) -> str | None:
        """Rewrite a command that uses `python` to use `python3` instead.

        Only rewrites when `python` appears as a standalone command (not
        inside `python3`, `pythonw`, or similar).  Returns None if no
        rewrite is applicable.
        """
        # Match `python` as a standalone word (not part of python3, pythonw, etc.)
        # Handles: `python script.py`, `python -c "..."`, `python -m module`, etc.
        rewritten = re.sub(r'\bpython\b(?!3|\w)', 'python3', command)
        return rewritten if rewritten != command else None

    def _execute_read_file(
        self,
        cwd: Path,
        args: dict[str, Any],
        enforce_guards: bool,
    ) -> ToolResult:
        """Read file contents."""
        file_path = args.get("file_path", "")
        offset = args.get("offset", 1)
        limit = args.get("limit", 500)

        if not file_path:
            return ToolResult.fail("No file_path provided")

        path = Path(file_path)
        if not path.is_absolute():
            path = cwd / path
        if enforce_guards:
            path = self._guards.require_read(path)

        if not path.exists():
            return ToolResult.fail(f"File not found: {path}")

        if not path.is_file():
            return ToolResult.fail(f"Not a file: {path}")

        try:
            file_size = path.stat().st_size

            # Guard: very large files get metadata + preview unless specific range requested
            if file_size > 100_000 and offset <= 1 and limit >= 500:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    all_lines = f.readlines()
                total = len(all_lines)
                head = all_lines[:100]
                tail = all_lines[-50:] if total > 150 else []
                output_lines = [f"[File: {path} | {file_size:,} bytes | {total} lines]"]
                output_lines.append(f"[Showing first 100 and last {len(tail)} lines. Use offset/limit for specific sections.]\n")
                for i, line in enumerate(head, start=1):
                    output_lines.append(format_hashline(i, line.rstrip()))
                if tail:
                    output_lines.append(f"\n[... {total - 100 - len(tail)} lines omitted ...]\n")
                    for i, line in enumerate(tail, start=total - len(tail) + 1):
                        output_lines.append(format_hashline(i, line.rstrip()))
                return ToolResult.ok("\n".join(output_lines))

            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            # Apply offset and limit (1-indexed)
            start = max(0, offset - 1)
            end = start + limit
            selected = lines[start:end]

            # Format with hashline tags (line_number:hash|content)
            total_lines = len(lines)
            shown_end = min(end, total_lines)
            header_line = f"[File: {path} | {total_lines} lines | {file_size:,} bytes | showing {start + 1}-{shown_end}]"
            output_lines = [header_line]
            for i, line in enumerate(selected, start=start + 1):
                output_lines.append(format_hashline(i, line.rstrip()))

            output = "\n".join(output_lines)

            if len(lines) > end:
                output += f"\n\n[... {len(lines) - end} more lines ...]"

            return ToolResult.ok(output)

        except Exception as e:
            return ToolResult.fail(f"Failed to read file: {e}")

    def _execute_write_file(
        self,
        cwd: Path,
        args: dict[str, Any],
        enforce_guards: bool,
    ) -> ToolResult:
        """Write content to a file."""
        file_path = args.get("file_path", "")
        content = args.get("content", "")

        if not file_path:
            return ToolResult.fail("No file_path provided")

        path = Path(file_path)
        if not path.is_absolute():
            path = cwd / path
        if enforce_guards:
            self._guards.require_mutation_allowed("write_file")
            path = self._guards.require_write(path)

        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            return ToolResult.ok(f"Wrote {len(content)} bytes to {path}")

        except Exception as e:
            return ToolResult.fail(f"Failed to write file: {e}")

    def _execute_list_dir(
        self,
        cwd: Path,
        args: dict[str, Any],
        enforce_guards: bool,
    ) -> ToolResult:
        """List directory contents."""
        dir_path = args.get("dir_path", ".")
        depth = args.get("depth", 2)
        limit = args.get("limit", 50)

        path = Path(dir_path)
        if not path.is_absolute():
            path = cwd / path
        if enforce_guards:
            path = self._guards.require_read(path)

        if not path.exists():
            return ToolResult.fail(f"Directory not found: {path}")

        if not path.is_dir():
            return ToolResult.fail(f"Not a directory: {path}")

        try:
            entries = []
            self._list_recursive(path, path, entries, depth, limit)

            if not entries:
                return ToolResult.ok("(empty directory)")

            output = "\n".join(entries[:limit])
            if len(entries) > limit:
                output += f"\n\n[... {len(entries) - limit} more entries ...]"

            return ToolResult.ok(output)

        except Exception as e:
            return ToolResult.fail(f"Failed to list directory: {e}")

    def _list_recursive(
        self,
        base: Path,
        current: Path,
        entries: list,
        max_depth: int,
        max_entries: int,
        current_depth: int = 0,
    ) -> None:
        """Recursively list directory contents."""
        if current_depth > max_depth or len(entries) >= max_entries:
            return

        try:
            items = sorted(current.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))

            for item in items:
                if len(entries) >= max_entries:
                    break

                if item.is_dir() and item.name in _LIST_DIR_SKIP_DIRS:
                    continue

                rel_path = item.relative_to(base)

                if item.is_dir():
                    entries.append(f"{rel_path}/")
                    self._list_recursive(
                        base, item, entries, max_depth, max_entries, current_depth + 1
                    )
                elif item.is_symlink():
                    entries.append(f"{rel_path}@")
                else:
                    entries.append(str(rel_path))

        except PermissionError:
            pass

    def _execute_grep(
        self,
        ctx: "AgentContext",
        cwd: Path,
        args: dict[str, Any],
        enforce_guards: bool,
    ) -> ToolResult:
        """Search files using ripgrep."""
        pattern = args.get("pattern", "")
        include = args.get("include", "")
        search_path = args.get("path", ".")
        limit = int(args.get("limit", 50))
        context_lines = int(args.get("context_lines", 2))
        # Clamp context_lines to 0-5
        context_lines = max(0, min(context_lines, 5))

        if not pattern:
            return ToolResult.fail("No pattern provided")

        # Build ripgrep command — keep -C and its value as separate
        # elements so the shell-quoting logic on line ~702 doesn't
        # merge them into a single quoted token like "-C 2" which
        # ripgrep cannot parse.
        cmd_parts = ["rg", "-n", "--color=never", "-C", str(context_lines)]

        if include:
            cmd_parts.extend(["-g", include])

        cmd_parts.append(pattern)
        cmd_parts.append(search_path)
        # Add max-count to limit total matches
        cmd_parts.append("--max-count")
        cmd_parts.append(str(limit))

        cmd = " ".join(f'"{p}"' if " " in p else p for p in cmd_parts)
        if enforce_guards:
            target = Path(search_path)
            if not target.is_absolute():
                target = cwd / target
            self._guards.require_read(target)

        try:
            result = subprocess.run(
                ["sh", "-c", cmd],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=30,
            )

            # ripgrep exit codes: 0=matches found, 1=no matches, 2=error
            if result.returncode == 2:
                return ToolResult.fail(
                    f"Search error: {(result.stderr or '').strip() or 'unknown ripgrep error'}"
                )

            # ripgrep output is "filepath:linenumber:content" (match) or
            # "filepath-linenumber-content" (context).  Inject hashline tags
            # for small result sets so the model can use hashline_edit directly.
            raw_lines = [line for line in result.stdout.strip().split("\n") if line]

            if not raw_lines:
                return ToolResult.ok(f"No matches found for pattern: '{pattern}'")

            truncated = raw_lines[:limit]

            # Only inject hashes when the result set is small enough that
            # reading files to compute them is cheap.
            if len(truncated) <= 80:
                truncated = self._inject_grep_hashes(cwd, truncated)

            output = "\n".join(truncated)
            if len(raw_lines) > limit:
                output += f"\n\n[... {len(raw_lines) - limit} more matches ...]"

            return ToolResult.ok(output)

        except subprocess.TimeoutExpired:
            return ToolResult.fail("Search timed out")
        except Exception as e:
            return ToolResult.fail(f"Search failed: {e}")

    @staticmethod
    def _inject_grep_hashes(cwd: Path, lines: list[str]) -> list[str]:
        """Post-process ripgrep output lines to inject hashline tags.

        Transforms ``filepath:42:content`` into ``filepath:42:ab|content``
        where ``ab`` is the 2-char content hash.  Context lines
        (``filepath-42-content``) are similarly transformed.
        Group separators (``--``) are passed through unchanged.
        """
        # Regex for ripgrep match/context lines:
        #   match:   filepath:linenum:content
        #   context: filepath-linenum-content
        rg_line_re = re.compile(r"^(.+?)([:])(\d+)([:])(.*)$")
        rg_ctx_re = re.compile(r"^(.+?)(-)(\d+)(-)(.*)$")

        # Cache: file path -> list of lines (0-indexed)
        file_cache: dict[str, list[str]] = {}

        def _get_file_lines(fpath: str) -> list[str] | None:
            if fpath in file_cache:
                return file_cache[fpath]
            p = Path(fpath)
            if not p.is_absolute():
                p = cwd / p
            try:
                file_cache[fpath] = p.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                file_cache[fpath] = []  # type: ignore[assignment]
                return None
            return file_cache[fpath]

        result: list[str] = []
        for raw in lines:
            if raw == "--":
                result.append(raw)
                continue

            m = rg_line_re.match(raw)
            sep = ":"
            if not m:
                m = rg_ctx_re.match(raw)
                sep = "-"
            if not m:
                result.append(raw)
                continue

            fpath, _, linenum_s, _, content = m.groups()
            linenum = int(linenum_s)

            # Compute hash from the actual file content at that line
            file_lines = _get_file_lines(fpath)
            if file_lines and 1 <= linenum <= len(file_lines):
                h = line_hash(file_lines[linenum - 1])
            else:
                h = line_hash(content)

            result.append(f"{fpath}{sep}{linenum}:{h}{sep}{content}")
        return result

    def _execute_apply_patch(
        self,
        cwd: Path,
        args: dict[str, Any],
        enforce_guards: bool,
    ) -> ToolResult:
        """Apply a patch to files."""
        patch = args.get("patch", "")

        if not patch:
            return ToolResult.fail("No patch provided")

        if enforce_guards:
            self._guards.require_mutation_allowed("apply_patch")
            for patch_path in self._extract_apply_patch_targets(patch):
                self._guards.require_write(self._resolve_path(cwd, patch_path))
        from src.tools.apply_patch import ApplyPatchTool

        tool = ApplyPatchTool(cwd)
        return tool.execute(patch=patch)

    def _extract_apply_patch_targets(self, patch: str) -> list[str]:
        """Parse apply_patch envelope and extract target file paths."""
        targets: list[str] = []
        pattern = re.compile(r"^\*\*\* (?:Add|Update|Delete) File:\s+(.+?)\s*$")
        for line in patch.splitlines():
            match = pattern.match(line)
            if match:
                targets.append(match.group(1).strip())
        return targets

    @staticmethod
    def _resolve_path(cwd: Path, file_path: str) -> Path:
        path = Path(file_path)
        if not path.is_absolute():
            path = cwd / path
        return path

    def _execute_view_image(
        self,
        cwd: Path,
        args: dict[str, Any],
        enforce_guards: bool,
    ) -> ToolResult:
        """View an image file."""
        path = args.get("path", "")

        if not path:
            return ToolResult.fail("No path provided")
        if enforce_guards:
            p = Path(path)
            if not p.is_absolute():
                p = cwd / p
            self._guards.require_read(p)

        from src.tools.view_image import view_image

        return view_image(path, cwd)

    def _execute_update_plan(self, args: dict[str, Any]) -> ToolResult:
        """Update the task plan."""
        steps = args.get("steps", [])
        explanation = args.get("explanation")
        if not isinstance(steps, list) or not steps:
            return ToolResult.fail("update_plan requires a non-empty steps array")

        normalized_steps: list[dict[str, str]] = []
        allowed_statuses = {"pending", "in_progress", "completed"}
        in_progress_count = 0
        pending_count = 0
        for idx, step in enumerate(steps, 1):
            if not isinstance(step, dict):
                return ToolResult.fail(f"Step {idx} must be an object")
            description = (step.get("description") or "").strip()
            status = (step.get("status") or "").strip()
            if not description:
                return ToolResult.fail(f"Step {idx} is missing a description")
            if status not in allowed_statuses:
                return ToolResult.fail(
                    f"Step {idx} has invalid status `{status}` (expected pending, in_progress, or completed)"
                )
            if status == "in_progress":
                in_progress_count += 1
            if status == "pending":
                pending_count += 1
            normalized_steps.append({"description": description, "status": status})

        # Keep plan progression predictable for long-horizon tasks.
        if in_progress_count > 1:
            return ToolResult.fail("Plan can only contain one in_progress step")
        if pending_count > 0 and in_progress_count != 1:
            return ToolResult.fail(
                "Plan must have exactly one in_progress step while pending steps remain"
            )

        self._plan = normalized_steps
        # Bump iteration marker — the loop stamps the current iteration via
        # ``stamp_plan_iteration`` after each turn that contains an update_plan.
        # We set a flag so the loop knows it needs to refresh the stamp.
        self._plan_dirty = True

        # Emit plan update as a JSONL event for progress tracking
        from src.output.jsonl import emit_raw
        emit_raw({
            "type": "plan.updated",
            "steps": normalized_steps,
        })

        # Format plan for output
        lines = ["Plan updated:"]
        for i, step in enumerate(normalized_steps, 1):
            status_icon = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }.get(step.get("status", "pending"), "[ ]")
            lines.append(f"  {status_icon} {i}. {step.get('description', '')}")

        if explanation:
            lines.append(f"\nReason: {explanation}")

        return ToolResult.ok("\n".join(lines))

    def format_plan_for_context(self) -> str:
        """Format current plan as compact context for the model."""
        if not self._plan:
            return ""
        lines = ["Current execution plan:"]
        for i, step in enumerate(self._plan, 1):
            status = step.get("status", "pending")
            icon = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(status, "[ ]")
            lines.append(f"{icon} {i}. {step.get('description', '')}")
        lines.append("Do not call update_plan just to mark steps done — batch status changes or skip intermediate updates.")
        return "\n".join(lines)

    def stamp_plan_iteration(self, iteration: int) -> None:
        """Record the loop iteration when the plan was last updated.

        Called by the main loop after processing a turn that contained
        an ``update_plan`` tool call so we can detect staleness later.
        """
        if getattr(self, "_plan_dirty", False):
            self._plan_last_updated_iteration = iteration
            self._plan_dirty = False

    @property
    def plan_last_updated_iteration(self) -> int:
        """Return the loop iteration at which the plan was last updated."""
        return self._plan_last_updated_iteration

    def _execute_web_search(self, args: dict[str, Any]) -> ToolResult:
        """Execute a web search."""
        from src.tools.web_search import web_search
        return web_search(args)

    # -------------------------------------------------------------------------
    # Subagent execution
    # -------------------------------------------------------------------------

    def _execute_subagent(
        self,
        ctx: "AgentContext",
        cwd: Path,
        args: dict[str, Any],
    ) -> ToolResult:
        """Spawn a subagent for task delegation."""
        task = args.get("task", "")
        subagent_type = args.get("type", "explore")
        sub_cwd = args.get("cwd", str(cwd))
        requested_overall_timeout = int(args.get("timeout_seconds", 0) or 0)
        requested_llm_timeout = int(args.get("llm_timeout_seconds", 0) or 0)
        requested_tool_timeout = int(args.get("tool_timeout_seconds", 0) or 0)

        if not task:
            return ToolResult.fail("No task provided for subagent")

        if subagent_type not in ("explore", "execute"):
            return ToolResult.fail(f"Invalid subagent type: {subagent_type}. Use 'explore' or 'execute'.")
        constraints = getattr(ctx, "runtime_constraints", {}) or {}
        readonly_parent = bool(constraints.get("readonly", False))
        depth = int(constraints.get("depth", 0) or 0)
        max_depth = int(constraints.get("max_subagent_depth", 1) or 1)
        if depth >= max_depth:
            return ToolResult.fail(
                f"Subagent depth limit exceeded ({depth} >= {max_depth}). Nested subagents are not allowed."
            )
        if readonly_parent and subagent_type == "execute":
            return ToolResult.fail("readonly parent cannot spawn execute subagent")
        budget = getattr(ctx, "agent_budget", None)
        reservation_amount = 0.05 if subagent_type == "explore" else 0.25
        reservation_key = f"subagent:{uuid.uuid4().hex[:10]}"
        if budget is not None and not budget.reserve(reservation_key, reservation_amount):
            snap = budget.snapshot()
            return ToolResult.fail(
                "Insufficient shared budget for subagent spawn",
                output=(
                    f"remaining_cost={snap.remaining_cost:.4f}, "
                    f"required_reservation={reservation_amount:.4f}"
                ),
            )

        from src.tools.subagent import run_subagent

        try:
            return run_subagent(
                task=task,
                subagent_type=subagent_type,
                cwd=sub_cwd,
                parent_constraints=constraints,
                budget=budget,
                timeout_seconds=requested_overall_timeout if requested_overall_timeout > 0 else None,
                llm_timeout_seconds=requested_llm_timeout if requested_llm_timeout > 0 else None,
                tool_timeout_seconds=requested_tool_timeout if requested_tool_timeout > 0 else None,
                budget_reservation_key=reservation_key,
            )
        finally:
            if budget is not None:
                budget.release(reservation_key)

    # -------------------------------------------------------------------------
    # Comparison execution
    # -------------------------------------------------------------------------

    def _execute_comparison(
        self,
        ctx: "AgentContext",
        cwd: Path,
        args: dict[str, Any],
    ) -> ToolResult:
        """Spawn multiple subagents to compare approaches."""
        task = args.get("task", "")
        approaches = args.get("approaches", [])
        sub_cwd = args.get("cwd", str(cwd))

        if not task:
            return ToolResult.fail("No task provided for comparison")
        if not approaches or not isinstance(approaches, list):
            return ToolResult.fail("No approaches provided for comparison")
        if len(approaches) > 3:
            return ToolResult.fail("Maximum 3 approaches allowed")

        constraints = getattr(ctx, "runtime_constraints", {}) or {}
        depth = int(constraints.get("depth", 0) or 0)
        max_depth = int(constraints.get("max_subagent_depth", 1) or 1)
        if depth >= max_depth:
            return ToolResult.fail("Subagent depth limit exceeded for comparison")

        budget = getattr(ctx, "agent_budget", None)

        from src.tools.subagent import run_comparison
        return run_comparison(
            task=task,
            approaches=approaches,
            cwd=sub_cwd,
            parent_constraints=constraints,
            budget=budget,
        )

    # -------------------------------------------------------------------------
    # str_replace execution
    # -------------------------------------------------------------------------

    def _execute_str_replace(
        self,
        cwd: Path,
        args: dict[str, Any],
        enforce_guards: bool,
    ) -> ToolResult:
        """Targeted find-and-replace in a file."""
        file_path = args.get("file_path", "")
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "")
        replace_all = args.get("replace_all", False)

        if not file_path:
            return ToolResult.fail("No file_path provided")
        if not old_string:
            return ToolResult.fail("No old_string provided")

        path = Path(file_path)
        if not path.is_absolute():
            path = cwd / path
        if enforce_guards:
            self._guards.require_mutation_allowed("str_replace")
            path = self._guards.require_write(path)

        if not path.exists():
            return ToolResult.fail(f"File not found: {path}")

        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            return ToolResult.fail(f"Failed to read file: {e}")

        count = content.count(old_string)
        if count == 0:
            return ToolResult.fail(
                f"old_string not found in {file_path}. Make sure it matches the file content exactly."
            )

        if count > 1 and not replace_all:
            return ToolResult.fail(
                f"old_string found {count} times in {file_path}. "
                f"Provide more context to make it unique, or set replace_all=true."
            )

        if replace_all:
            new_content = content.replace(old_string, new_string)
            replaced = count
        else:
            new_content = content.replace(old_string, new_string, 1)
            replaced = 1

        try:
            path.write_text(new_content, encoding="utf-8")
        except Exception as e:
            return ToolResult.fail(f"Failed to write file: {e}")

        return ToolResult.ok(
            f"Replaced {replaced} occurrence(s) in {file_path}"
        )

    # -------------------------------------------------------------------------
    # hashline_edit execution
    # -------------------------------------------------------------------------

    def _execute_hashline_edit(
        self,
        cwd: Path,
        args: dict[str, Any],
        enforce_guards: bool,
    ) -> ToolResult:
        """Edit a file using hashline references (line_number:hash pairs)."""
        from src.tools.hashline import compute_hashes, parse_ref, validate_ref

        file_path = args.get("file_path", "")
        operations = args.get("operations", [])

        if not file_path:
            return ToolResult.fail("No file_path provided")
        if not operations or not isinstance(operations, list):
            return ToolResult.fail("No operations provided (expected a non-empty array)")

        path = Path(file_path)
        if not path.is_absolute():
            path = cwd / path
        if enforce_guards:
            self._guards.require_mutation_allowed("hashline_edit")
            path = self._guards.require_write(path)

        if not path.exists():
            return ToolResult.fail(f"File not found: {path}")

        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            return ToolResult.fail(f"Failed to read file: {e}")

        lines = content.splitlines()

        # ------------------------------------------------------------------
        # Validate all refs before applying anything.
        # Collect ALL errors (especially hash mismatches) so the model can
        # self-correct in a single retry instead of re-reading the file
        # across multiple turns.
        # ------------------------------------------------------------------
        parsed_ops: list[dict[str, Any]] = []
        validation_errors: list[str] = []
        for idx, op in enumerate(operations, 1):
            op_type = op.get("op", "")
            start_ref = op.get("start", "")
            end_ref = op.get("end", "")
            new_content = op.get("content", "")

            if op_type not in ("replace", "insert", "delete"):
                # Structural errors are fatal — fail immediately
                return ToolResult.fail(
                    f"Operation {idx}: invalid op '{op_type}' (expected replace, insert, or delete)"
                )
            if not start_ref:
                return ToolResult.fail(f"Operation {idx}: missing 'start' reference")

            # Validate start ref — collect errors, don't fail yet
            ok, err = validate_ref(start_ref, lines)
            if not ok:
                validation_errors.append(f"Operation {idx}: {err}")
                # Still parse the ref to continue validation of other ops
                try:
                    start_line, _ = parse_ref(start_ref)
                except ValueError:
                    start_line = 1
            else:
                start_line, _ = parse_ref(start_ref)

            # Validate end ref if present
            end_line = start_line
            if end_ref:
                ok, err = validate_ref(end_ref, lines)
                if not ok:
                    validation_errors.append(f"Operation {idx}: {err}")
                    try:
                        end_line, _ = parse_ref(end_ref)
                    except ValueError:
                        end_line = start_line
                else:
                    end_line, _ = parse_ref(end_ref)
                if end_line < start_line and not validation_errors:
                    validation_errors.append(
                        f"Operation {idx}: end line {end_line} < start line {start_line}"
                    )

            # Content required for replace/insert
            if op_type in ("replace", "insert") and new_content is None:
                return ToolResult.fail(
                    f"Operation {idx}: '{op_type}' requires 'content'"
                )

            parsed_ops.append({
                "op": op_type,
                "start": start_line,
                "end": end_line,
                "content": new_content or "",
            })

        # Report ALL validation errors at once so the model can fix them
        # in a single retry rather than discovering them one at a time.
        if validation_errors:
            error_summary = "; ".join(validation_errors)
            return ToolResult.fail(
                f"{len(validation_errors)} validation error(s): {error_summary}"
            )

        # ------------------------------------------------------------------
        # Sort operations bottom-up (highest line number first) so earlier
        # line numbers remain valid as we modify the list.
        # ------------------------------------------------------------------
        parsed_ops.sort(key=lambda o: o["start"], reverse=True)

        # ------------------------------------------------------------------
        # Apply operations
        # ------------------------------------------------------------------
        changes_made = 0
        for op in parsed_ops:
            op_type = op["op"]
            start = op["start"]  # 1-indexed
            end = op["end"]      # 1-indexed, inclusive
            new_content = op["content"]

            if op_type == "replace":
                new_lines = new_content.splitlines() if new_content else []
                lines[start - 1 : end] = new_lines
                changes_made += 1

            elif op_type == "insert":
                new_lines = new_content.splitlines() if new_content else []
                # Insert after the referenced line
                lines[start : start] = new_lines
                changes_made += 1

            elif op_type == "delete":
                lines[start - 1 : end] = []
                changes_made += 1

        # ------------------------------------------------------------------
        # Write result
        # ------------------------------------------------------------------
        new_file_content = "\n".join(lines)
        # Preserve trailing newline if original had one
        if content.endswith("\n"):
            new_file_content += "\n"

        try:
            path.write_text(new_file_content, encoding="utf-8")
        except Exception as e:
            return ToolResult.fail(f"Failed to write file: {e}")

        return ToolResult.ok(
            f"Applied {changes_made} operation(s) to {file_path}"
        )

    # -------------------------------------------------------------------------
    # glob_files execution
    # -------------------------------------------------------------------------

    def _execute_glob_files(
        self,
        cwd: Path,
        args: dict[str, Any],
        enforce_guards: bool,
    ) -> ToolResult:
        """Find files matching a glob pattern."""
        pattern = args.get("pattern", "")
        path = args.get("path", ".")
        limit = args.get("limit", 100)

        if not pattern:
            return ToolResult.fail("No pattern provided")

        search_path = Path(path)
        if not search_path.is_absolute():
            search_path = cwd / search_path
        if enforce_guards:
            search_path = self._guards.require_read(search_path)

        if not search_path.exists():
            return ToolResult.fail(f"Path not found: {search_path}")

        try:
            # Prepend **/ if the pattern doesn't start with it (for recursive search)
            effective_pattern = pattern
            if not pattern.startswith("**/"):
                effective_pattern = f"**/{pattern}"

            matches = sorted(
                search_path.glob(effective_pattern),
                key=lambda p: p.stat().st_mtime if p.exists() else 0,
                reverse=True,
            )

            # Filter to files only
            files = [str(m.relative_to(search_path)) for m in matches if m.is_file()]

            if not files:
                return ToolResult.ok("No files matched the pattern.")

            output = "\n".join(files[:limit])
            if len(files) > limit:
                output += f"\n\n[... {len(files) - limit} more files ...]"

            return ToolResult.ok(output)

        except Exception as e:
            return ToolResult.fail(f"Glob search failed: {e}")

    # -------------------------------------------------------------------------
    # lint execution
    # -------------------------------------------------------------------------

    def _execute_lint(
        self,
        cwd: Path,
        args: dict[str, Any],
        enforce_guards: bool,
    ) -> ToolResult:
        """Run linter on specified files and return diagnostics."""
        files = args.get("files", [])
        linter = args.get("linter")  # auto-detect if not specified

        if not files:
            return ToolResult.fail("No files provided. Pass a list of file paths to lint.")
        if enforce_guards:
            for file_path in files:
                p = Path(file_path)
                if not p.is_absolute():
                    p = cwd / p
                self._guards.require_read(p)

        # Auto-detect linter if not specified
        if not linter:
            if (cwd / "pyproject.toml").exists() or (cwd / "ruff.toml").exists():
                linter = "ruff"
            elif (cwd / ".eslintrc.json").exists() or (cwd / "eslint.config.js").exists():
                linter = "eslint"
            elif (cwd / "setup.cfg").exists() or (cwd / ".flake8").exists():
                linter = "flake8"
            else:
                # Default to ruff for Python, eslint for JS/TS
                ext = Path(files[0]).suffix if files else ""
                if ext in (".py",):
                    linter = "ruff"
                elif ext in (".js", ".ts", ".jsx", ".tsx"):
                    linter = "eslint"
                else:
                    linter = "ruff"  # fallback

        file_args = " ".join(f'"{f}"' for f in files)

        if linter == "ruff":
            cmd = f"ruff check --output-format=text {file_args}"
        elif linter == "eslint":
            cmd = f"npx eslint --format=compact {file_args}"
        elif linter == "flake8":
            cmd = f"flake8 {file_args}"
        else:
            cmd = f"{linter} {file_args}"

        try:
            result = subprocess.run(
                ["sh", "-c", cmd],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=60,
            )
            output = result.stdout.strip()
            if result.stderr.strip():
                output += f"\n{result.stderr.strip()}"

            if not output:
                return ToolResult.ok("No linter errors found.")

            return ToolResult.ok(output)

        except subprocess.TimeoutExpired:
            return ToolResult.fail("Lint timed out after 60s")
        except Exception as e:
            return ToolResult.fail(f"Lint failed: {e}")

    # -------------------------------------------------------------------------
    # ask_user execution
    # -------------------------------------------------------------------------

    def set_ask_user_callback(
        self,
        callback: Optional[Callable[[str, Optional[list[str]], str], str]],
    ) -> None:
        """Register a callback for the ask_user tool.

        The callback signature is:
            callback(question: str, options: list[str] | None, request_id: str) -> str

        It should block until the user responds and return the answer text.
        If no callback is registered, ask_user will emit a JSONL event and
        wait for the answer to be pushed via :meth:`push_user_answer`.
        """
        self._ask_user_callback = callback

    def push_user_answer(self, request_id: str, answer: str) -> None:
        """Push a user answer for a pending ask_user request."""
        pending = self._pending_asks.get(request_id)
        if pending:
            pending["answer"] = answer
            pending["event"].set()

    def _execute_ask_user(self, args: dict[str, Any]) -> ToolResult:
        """Ask the user a question and wait for a response."""
        question = args.get("question", "")
        options = args.get("options")

        if not question:
            return ToolResult.fail("No question provided")

        request_id = f"ask_{uuid.uuid4().hex[:8]}"

        # If a callback is registered (e.g., from Telegram), use it directly
        if self._ask_user_callback is not None:
            try:
                answer = self._ask_user_callback(question, options, request_id)
                return ToolResult.ok(f"User answered: {answer}")
            except Exception as e:
                return ToolResult.fail(f"ask_user callback failed: {e}")

        # Otherwise, emit event and wait for push_user_answer()
        from src.output.jsonl import emit, UserInputRequestedEvent

        wait_event = threading.Event()
        self._pending_asks[request_id] = {
            "event": wait_event,
            "answer": "",
        }

        emit(UserInputRequestedEvent(
            question=question,
            options=options,
            request_id=request_id,
        ))

        # Wait up to 5 minutes for user response
        answered = wait_event.wait(timeout=300)
        pending = self._pending_asks.pop(request_id, {})

        if not answered:
            return ToolResult.fail(
                "User did not respond within 5 minutes. "
                "Make a reasonable decision and proceed."
            )

        answer = pending.get("answer", "")
        return ToolResult.ok(f"User answered: {answer}")

    # -------------------------------------------------------------------------
    # Caching methods
    # -------------------------------------------------------------------------

    def _cache_key(self, name: str, arguments: dict[str, Any]) -> str:
        """Generate a cache key for a tool call."""
        args_json = json.dumps(arguments, sort_keys=True, default=str)
        content = f"{name}:{args_json}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def _get_cached(self, key: str) -> Optional[ToolResult]:
        """Get a cached result if valid."""
        cached = self._cache.get(key)
        if cached is not None and cached.is_valid(self._config.cache_ttl):
            return cached.result
        return None

    def _cache_result(self, key: str, result: ToolResult) -> None:
        """Cache a tool result."""
        self._cache[key] = CachedResult(result=result, cached_at=time.time())

        # Evict old entries if cache is too large
        if len(self._cache) > 1000:
            self._evict_expired_cache()

    def _evict_expired_cache(self) -> None:
        """Remove expired entries from cache."""
        now = time.time()
        expired_keys = [
            key
            for key, cached in self._cache.items()
            if not cached.is_valid(self._config.cache_ttl)
        ]
        for key in expired_keys:
            del self._cache[key]

    def clear_cache(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()

    # -------------------------------------------------------------------------
    # Statistics methods
    # -------------------------------------------------------------------------

    def _record_execution(
        self,
        tool_name: str,
        duration_ms: int,
        success: bool,
        cached: bool,
    ) -> None:
        """Record execution statistics."""
        self._stats.total_executions += 1
        self._stats.total_duration_ms += duration_ms

        if success:
            self._stats.successful_executions += 1
        else:
            self._stats.failed_executions += 1

        if cached:
            self._stats.cache_hits += 1

        # Per-tool stats
        if tool_name not in self._stats.by_tool:
            self._stats.by_tool[tool_name] = ToolStats()

        tool_stats = self._stats.by_tool[tool_name]
        tool_stats.executions += 1
        tool_stats.total_ms += duration_ms
        if success:
            tool_stats.successes += 1

    def stats(self) -> ExecutorStats:
        """Get execution statistics."""
        return self._stats

    # -------------------------------------------------------------------------
    # Batch execution
    # -------------------------------------------------------------------------

    def _execute_with_lock(
        self,
        ctx: "AgentContext",
        name: str,
        args: dict[str, Any],
    ) -> ToolResult:
        """Execute a tool with the appropriate read or write lock."""
        is_write = tool_is_mutating(name, args)
        if is_write:
            self._rwlock.acquire_write()
        else:
            self._rwlock.acquire_read()
        try:
            return self.execute(ctx, name, args)
        finally:
            if is_write:
                self._rwlock.release_write()
            else:
                self._rwlock.release_read()

    def execute_batch(
        self,
        ctx: "AgentContext",
        calls: List[Tuple[str, dict]],
    ) -> List[ToolResult]:
        """Execute multiple tool calls with read/write locking.

        Parallel-safe (read-only) tools run concurrently under a shared read
        lock.  Mutating tools acquire an exclusive write lock, blocking until
        all concurrent reads finish and preventing new reads until done.

        Args:
            ctx: Agent context with shell() method
            calls: List of (tool_name, arguments) tuples

        Returns:
            List of ToolResults in the same order as input calls
        """
        if not calls:
            return []

        # For single call, just execute directly (still use lock)
        if len(calls) == 1:
            name, args = calls[0]
            return [self._execute_with_lock(ctx, name, args)]

        # Execute in parallel using ThreadPoolExecutor with RW locking
        results: List[Optional[ToolResult]] = [None] * len(calls)

        with ThreadPoolExecutor(max_workers=self._config.max_concurrent) as executor:
            future_to_index = {
                executor.submit(self._execute_with_lock, ctx, name, args): i
                for i, (name, args) in enumerate(calls)
            }

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    results[index] = ToolResult.fail(f"Batch execution failed: {e}")

        # Ensure all results are filled (shouldn't happen, but just in case)
        return [r if r is not None else ToolResult.fail("No result") for r in results]

    def get_plan(self) -> list[dict[str, str]]:
        """Get the current plan."""
        return self._plan.copy()

    def configure_guards(
        self,
        *,
        readable_roots: list[str] | None = None,
        writable_roots: list[str] | None = None,
        readonly: bool = False,
        enabled: bool = True,
    ) -> None:
        """Configure path/mutation guard behavior for this registry."""
        self._guards = PathGuards(
            GuardConfig.from_paths(
                cwd=self.cwd,
                readable_roots=readable_roots,
                writable_roots=writable_roots,
                readonly=readonly,
                enabled=enabled,
            )
        )

    def get_tools_for_llm(self) -> list:
        """Get tool specifications formatted for the LLM.

        Returns tools in OpenAI-compatible format, including any
        custom tools registered via :meth:`register_tool`.
        """
        specs = get_all_tools()
        tools = []

        for spec in specs:
            if self._allowed_tools is not None and spec["name"] not in self._allowed_tools:
                continue
            if spec["name"] == "spawn_subagent" and not self._allow_subagent_spawn:
                continue
            tools.append(
                {
                    "name": spec["name"],
                    "description": spec.get("description", ""),
                    "parameters": spec.get("parameters", {}),
                }
            )

        # Append custom-registered tools
        for _name, (spec, _handler) in self._custom_tools.items():
            tools.append(
                {
                    "name": spec["name"],
                    "description": spec.get("description", ""),
                    "parameters": spec.get("parameters", {}),
                }
            )

        return tools
