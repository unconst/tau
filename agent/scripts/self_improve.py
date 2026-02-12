#!/usr/bin/env python3
"""
Self-improvement loop for the agent.

Iteratively discovers weaknesses, implements improvements via Cursor CLI,
verifies them (syntax, imports, benchmarks), and accumulates successful
changes via git commits.

Features:
- Institutional memory tracking all past iterations and outcomes
- Benchmark suite for measuring agent quality before/after changes
- Adaptive focus selection (weakest areas get priority)
- Quality gates (syntax check, import check, benchmark regression check)
- Git accumulation (successful improvements are committed, not thrown away)
- Score-based acceptance threshold
- Failure mode adaptation (auto-adjusts based on recent failures)

Usage:
    python3 agent/scripts/self_improve.py
    python3 agent/scripts/self_improve.py --iterations 5 --skip-impl
    python3 agent/scripts/self_improve.py --skip-benchmarks
    python3 agent/scripts/self_improve.py --model claude-4.5-sonnet
"""

from __future__ import annotations

import argparse
import ast
import datetime
import json
import os
import re
import selectors
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = REPO_ROOT / "agent"
LOG_DIR = AGENT_DIR / ".agent" / "self_improve"
MEMORY_FILE = LOG_DIR / "memory.json"

# Make benchmark_suite importable from the same directory
sys.path.insert(0, str(Path(__file__).parent))

# Directories that git-clean should never touch (relative to REPO_ROOT)
CLEAN_EXCLUDES = [
    "agent/scripts/",
    "agent/.agent/",
]

# ---------------------------------------------------------------------------
# Focus areas
# ---------------------------------------------------------------------------
FOCUS_AREAS = [
    "error handling and recovery",
    "search and navigation",
    "planning and task decomposition",
    "context management",
    "tool implementation",
    "LLM interaction",
]

FOCUS_DESCRIPTIONS = {
    "error handling and recovery": (
        "how the agent handles tool failures, bad outputs, retries, and error messages"
    ),
    "search and navigation": (
        "how the agent finds files, searches code, navigates and understands project structure"
    ),
    "planning and task decomposition": (
        "how the agent breaks down complex tasks, tracks progress, uses update_plan"
    ),
    "context management": (
        "how the agent manages conversation history, compaction, token limits"
    ),
    "tool implementation": (
        "quality of tool execution, output handling, edge cases in _execute_* methods"
    ),
    "LLM interaction": (
        "prompt design, caching strategy, model selection, cost efficiency"
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log(msg: str, level: str = "INFO"):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def banner(text: str):
    print(flush=True)
    print("=" * 70, flush=True)
    print(f"  {text}", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)


# ===========================================================================
# MEMORY SYSTEM
# ===========================================================================

def _empty_memory() -> dict:
    """Return a fresh memory structure."""
    return {
        "iterations_total": 0,
        "improvements_landed": 0,
        "improvements_reverted": 0,
        "history": [],
        "focus_scores": {
            area: {
                "attempts": 0,
                "avg_score": 0.0,
                "best": 0,
                "last_landed": None,
                "landed_count": 0,
            }
            for area in FOCUS_AREAS
        },
        "blacklist": [],
        "failure_counts": {
            "parse_error": 0,
            "timeout": 0,
            "crash": 0,
            "regression": 0,
            "syntax_error": 0,
            "import_error": 0,
        },
        "last_benchmark": None,
    }


def load_memory() -> dict:
    """Load persistent memory from disk."""
    if MEMORY_FILE.exists():
        try:
            data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
            # Forward-compat: ensure all keys exist
            empty = _empty_memory()
            for k, v in empty.items():
                data.setdefault(k, v)
            # Ensure all focus areas exist
            for area in FOCUS_AREAS:
                if area not in data["focus_scores"]:
                    data["focus_scores"][area] = {
                        "attempts": 0, "avg_score": 0.0,
                        "best": 0, "last_landed": None, "landed_count": 0,
                    }
            return data
        except (json.JSONDecodeError, KeyError, TypeError):
            log("Corrupt memory.json — starting fresh", "WARN")
    return _empty_memory()


def save_memory(memory: dict):
    """Persist memory to disk."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(
        json.dumps(memory, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _title_similar(a: str, b: str) -> bool:
    """Check if two task titles are similar enough to be the same idea."""
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    if not a_words or not b_words:
        return False
    overlap = len(a_words & b_words)
    return overlap / max(len(a_words), len(b_words)) > 0.6


def update_memory(
    memory: dict,
    *,
    iteration: int,
    focus: str,
    task_title: str,
    score: int | None,
    outcome: str,
    files_changed: list[str] | None = None,
    summary: str = "",
):
    """Update memory after an iteration completes."""
    memory["iterations_total"] += 1

    entry = {
        "iteration": iteration,
        "global_iteration": memory["iterations_total"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "focus": focus,
        "task_title": task_title,
        "score": score,
        "outcome": outcome,
        "files_changed": files_changed or [],
        "summary": summary,
    }
    memory["history"].append(entry)

    # Keep history bounded (last 100 entries)
    if len(memory["history"]) > 100:
        memory["history"] = memory["history"][-100:]

    # Update focus scores
    if focus in memory["focus_scores"]:
        fs = memory["focus_scores"][focus]
        fs["attempts"] += 1
        if score is not None:
            n = fs["attempts"]
            fs["avg_score"] = round(
                ((fs["avg_score"] * (n - 1)) + score) / n, 2
            )
            if score > fs["best"]:
                fs["best"] = score
        if outcome == "landed":
            fs["last_landed"] = f"iter_{iteration:04d}"
            fs["landed_count"] = fs.get("landed_count", 0) + 1
            memory["improvements_landed"] += 1
        elif outcome == "reverted":
            memory["improvements_reverted"] += 1

    # Update failure counts
    fc = memory.get("failure_counts", {})
    if outcome in fc:
        fc[outcome] += 1

    # Auto-blacklist: if same idea has failed 3+ times
    similar_failures = sum(
        1 for h in memory["history"]
        if h.get("outcome") in ("crash", "timeout", "syntax_error", "import_error", "regression")
        and _title_similar(h.get("task_title", ""), task_title)
    )
    if similar_failures >= 3:
        bl_entry = f"{task_title} (failed {similar_failures}x)"
        if not any(_title_similar(b, task_title) for b in memory["blacklist"]):
            memory["blacklist"].append(bl_entry)
            log(f"Auto-blacklisted: {bl_entry}")

    save_memory(memory)


def format_history_for_prompt(memory: dict, last_n: int = 15) -> str:
    """Format recent history for inclusion in prompts."""
    history = memory.get("history", [])[-last_n:]
    if not history:
        return "No previous iterations yet."
    lines = []
    for h in history:
        score = h.get("score")
        score_str = f"{score}/10" if score is not None else "N/A"
        lines.append(
            f"- \"{h.get('task_title', '?')}\" "
            f"(focus: {h.get('focus', '?')}, score: {score_str}, "
            f"outcome: {h.get('outcome', '?')})"
        )
    return "\n".join(lines)


# ===========================================================================
# SCORE EXTRACTION
# ===========================================================================

def extract_score(text: str) -> int | None:
    """Extract a score from critique text (N/10 or Score: N patterns)."""
    if not text:
        return None
    # Try "N/10" pattern (most common)
    match = re.search(r'(\d+)\s*/\s*10', text)
    if match:
        return min(int(match.group(1)), 10)
    # Try "Score: N" pattern
    match = re.search(r'[Ss]core[:\s]+(\d+)', text)
    if match:
        return min(int(match.group(1)), 10)
    return None


# ===========================================================================
# ADAPTIVE FOCUS SELECTION
# ===========================================================================

def select_focus(memory: dict) -> str:
    """Select the focus area most in need of improvement.

    Priority favors areas with lower average scores and fewer landed
    improvements, while deprioritizing areas that have seen recent success.
    """
    focus_scores = memory.get("focus_scores", {})
    if not focus_scores:
        return FOCUS_AREAS[0]

    candidates = []
    for area in FOCUS_AREAS:
        stats = focus_scores.get(area, {})
        avg = stats.get("avg_score", 0.0)
        landed = stats.get("landed_count", 0)
        attempts = stats.get("attempts", 0)

        # Lower priority score = selected first (weaker area)
        # - Low avg_score → higher priority
        # - More landed → lower priority (area is improving)
        # - Many attempts without landings → higher priority
        priority = avg + (landed * 3.0) - max(0, attempts - landed) * 0.3
        candidates.append((priority, area))

    candidates.sort(key=lambda x: x[0])

    # Rotate within the weakest half for variety
    pool_size = max(2, len(candidates) // 2)
    pool = candidates[:pool_size]
    idx = memory.get("iterations_total", 0) % len(pool)
    return pool[idx][1]


# ===========================================================================
# FAILURE ADAPTATION
# ===========================================================================

def get_adaptive_params(memory: dict) -> dict:
    """Adjust parameters based on recent failure patterns."""
    recent = memory.get("history", [])[-10:]

    recent_parse = sum(1 for h in recent if h.get("outcome") == "parse_error")
    recent_timeout = sum(1 for h in recent if h.get("outcome") == "timeout")
    recent_crash = sum(1 for h in recent if h.get("outcome") == "crash")
    recent_syntax = sum(1 for h in recent if h.get("outcome") in ("syntax_error", "import_error"))

    return {
        "add_format_examples": recent_parse >= 2,
        "simplify_tasks": recent_crash >= 2 or recent_syntax >= 3,
        "increase_timeout": recent_timeout >= 2,
        "emphasize_syntax": recent_syntax >= 2,
    }


# ===========================================================================
# CURSOR CLI WRAPPER
# ===========================================================================

_cursor_cli_consecutive_rate_limits = 0


def cursor_cli(
    prompt: str,
    *,
    mode: str = "ask",
    model: str = "opus-4.6-thinking",
    max_retries: int = 3,
    base_backoff: float = 30.0,
) -> str:
    """Run Cursor CLI, stream events, return final text.

    Includes rate-limit detection with exponential backoff.
    """
    global _cursor_cli_consecutive_rate_limits

    for attempt in range(1, max_retries + 1):
        cmd = [
            "agent", "-p",
            "--model", model,
            "--mode", mode,
            "--output-format", "stream-json",
            "--force",
            prompt,
        ]

        log(f"Cursor ({mode}, attempt {attempt}/{max_retries}) → {prompt[:120]}...")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=str(REPO_ROOT),
        )

        result_text = ""

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type", "")
            subtype = event.get("subtype", "")

            if etype == "system" and subtype == "init":
                print(f"  ▸ init model={event.get('model', '?')}", flush=True)

            elif etype == "assistant":
                msg = event.get("message", {})
                content = msg.get("content", [])
                text = "".join(
                    c.get("text", "") for c in content if c.get("type") == "text"
                )
                result_text = text
                snippet = text.replace("\n", " ")[:120]
                print(f"  ▸ assistant: {snippet}…", flush=True)

            elif etype == "tool_call":
                td = event.get("tool_call", {})
                tool_name = "unknown"
                args = {}
                for k, v in td.items():
                    if isinstance(v, dict):
                        tool_name = k.replace("ToolCall", "").replace("Call", "")
                        args = v.get("args", {})
                        break
                if subtype == "started":
                    print(f"  ▸ tool.start: {tool_name}({json.dumps(args)[:80]})", flush=True)
                elif subtype == "completed":
                    print(f"  ▸ tool.done:  {tool_name}", flush=True)

            elif etype == "result":
                result_text = event.get("result", result_text)
                print(f"  ▸ result ({event.get('duration_ms', 0)}ms)", flush=True)

        stderr_text = proc.stderr.read()
        proc.wait()

        # Rate-limit detection
        rate_limited = False
        if proc.returncode != 0 and stderr_text:
            signals = ("rate limit", "rate_limit", "429", "throttled", "too many requests")
            if any(s in stderr_text.lower() for s in signals):
                rate_limited = True

        if rate_limited and attempt < max_retries:
            _cursor_cli_consecutive_rate_limits += 1
            wait = min(base_backoff * (2 ** (attempt - 1)), 180.0)
            log(
                f"Rate-limited (consecutive: {_cursor_cli_consecutive_rate_limits}). "
                f"Pausing {wait:.0f}s before retry...",
                "WARN",
            )
            time.sleep(wait)
            continue

        if not rate_limited:
            _cursor_cli_consecutive_rate_limits = 0

        return result_text

    log("Cursor CLI retries exhausted after rate-limiting", "WARN")
    return result_text


# ===========================================================================
# GIT HELPERS
# ===========================================================================

def git_save_patch() -> str:
    """Save current uncommitted changes as a unified diff string."""
    result = subprocess.run(
        ["git", "diff"], cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    patch = result.stdout
    if patch.strip():
        log("Saved working-tree patch")
    return patch


def git_apply_patch(patch: str) -> bool:
    """Apply a saved patch to restore the working tree."""
    if not patch.strip():
        return True
    result = subprocess.run(
        ["git", "apply", "--allow-empty"],
        input=patch, cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    if result.returncode == 0:
        log("Restored working-tree from patch")
        return True
    log(f"Patch apply failed: {result.stderr.strip()}", "WARN")
    return False


def git_diff_stat() -> str:
    result = subprocess.run(
        ["git", "diff", "--stat"], cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    return result.stdout.strip()


def git_diff_content() -> str:
    """Return the full git diff content."""
    result = subprocess.run(
        ["git", "diff"], cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    return result.stdout


def git_diff_name_only() -> set[str]:
    """Return the set of files currently modified in the working tree."""
    result = subprocess.run(
        ["git", "diff", "--name-only"], cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    return {f.strip() for f in result.stdout.splitlines() if f.strip()}


def git_undo_changes():
    """Reset working tree to HEAD, protecting our own files."""
    subprocess.run(
        ["git", "checkout", "--", "."], cwd=str(REPO_ROOT), capture_output=True,
    )
    clean_cmd = ["git", "clean", "-fd"]
    for exc in CLEAN_EXCLUDES:
        clean_cmd += ["--exclude", exc]
    subprocess.run(clean_cmd, cwd=str(REPO_ROOT), capture_output=True)


def git_revert_to_patch(pre_patch: str):
    """Revert all working tree changes and restore from a saved patch."""
    git_undo_changes()
    if pre_patch.strip():
        git_apply_patch(pre_patch)


def git_commit_improvement(title: str, files: set[str], iteration: int) -> bool:
    """Commit specific files with a self-improvement commit message.

    Only commits files that exist and have changes staged.
    """
    if not files:
        log("No files to commit")
        return False

    valid_files = []
    for f in sorted(files):
        full = REPO_ROOT / f
        if full.exists():
            valid_files.append(f)

    if not valid_files:
        log("No valid files to commit")
        return False

    # Stage only the implementation files
    for f in valid_files:
        subprocess.run(["git", "add", f], cwd=str(REPO_ROOT), capture_output=True)

    msg = f"self-improve(iter {iteration}): {title}"
    result = subprocess.run(
        ["git", "commit", "-m", msg],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    if result.returncode == 0:
        log(f"Committed: {msg}")
        return True

    log(f"Commit failed: {result.stderr.strip()}", "WARN")
    return False


# ===========================================================================
# AGENT RUNNER
# ===========================================================================

def _venv_python() -> str:
    """Return the .venv Python interpreter path."""
    for candidate in (REPO_ROOT, AGENT_DIR):
        venv_py = candidate / ".venv" / "bin" / "python"
        if venv_py.is_file():
            return str(venv_py)
    return sys.executable


def _parse_agent_jsonl(stdout: str) -> dict:
    """Parse JSONL events from agent stdout to extract structured results."""
    turns_completed = 0
    turns_failed = 0
    errors: list[str] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}
    session_id = None
    completion_reason = ""

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        etype = event.get("type", "")

        if etype == "thread.started":
            session_id = event.get("thread_id")
        elif etype == "turn.completed":
            turns_completed += 1
            u = event.get("usage", {})
            usage["input_tokens"] += int(u.get("input_tokens", 0) or 0)
            usage["output_tokens"] += int(u.get("output_tokens", 0) or 0)
            usage["cached_input_tokens"] += int(u.get("cached_input_tokens", 0) or 0)
        elif etype == "turn.failed":
            turns_failed += 1
            errors.append(event.get("error", {}).get("message", "unknown error"))
        elif etype == "stream.error":
            errors.append(event.get("message", "unknown stream error"))
        elif etype == "turn.metrics":
            completion_reason = event.get("completion_reason", completion_reason)

    return {
        "session_id": session_id,
        "turns_completed": turns_completed,
        "turns_failed": turns_failed,
        "errors": errors,
        "usage": usage,
        "completion_reason": completion_reason,
    }


def run_agent(
    instruction: str,
    timeout: int = 1800,
    pre_modified_files: set[str] | None = None,
    max_iterations: int = 0,
) -> dict:
    """Run agent.py with an instruction, stream output live, return results."""
    python = _venv_python()
    cmd = [python, str(AGENT_DIR / "agent.py"), "--instruction", instruction]
    if max_iterations > 0:
        cmd.extend(["--max-iterations", str(max_iterations)])

    log(f"Agent → {instruction[:100]}...")
    started = time.time()

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    timed_out = False

    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ)
    sel.register(proc.stderr, selectors.EVENT_READ)

    try:
        open_streams = 2
        while open_streams > 0:
            remaining = timeout - (time.time() - started)
            if remaining <= 0:
                timed_out = True
                proc.kill()
                break
            events = sel.select(timeout=min(remaining, 1.0))
            for key, _ in events:
                line = key.fileobj.readline()
                if not line:
                    sel.unregister(key.fileobj)
                    open_streams -= 1
                    continue
                if key.fileobj is proc.stdout:
                    stdout_lines.append(line)
                    print(line, end="", flush=True)
                else:
                    stderr_lines.append(line)
                    print(line, end="", flush=True)
    finally:
        sel.close()

    proc.wait()
    elapsed = time.time() - started
    exit_code = proc.returncode if not timed_out else -1
    stdout = "".join(stdout_lines)
    stderr = "".join(stderr_lines)

    jsonl = _parse_agent_jsonl(stdout)
    turns = jsonl["turns_completed"]
    fails = jsonl["turns_failed"]

    if turns == 0 and fails == 0:
        log("Agent produced no turns — likely crashed", "WARN")
    elif fails > 0 and turns == 0:
        log(f"Agent failed all {fails} turn(s)", "WARN")
    else:
        log(f"Agent completed {turns} turn(s), {fails} failed")

    # Diff stat
    post_modified = git_diff_name_only()
    pre = pre_modified_files or set()
    agent_new = post_modified - pre
    diff_stat = ""
    if agent_new:
        stat = subprocess.run(
            ["git", "diff", "--stat", "--"] + sorted(agent_new),
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        diff_stat = stat.stdout.strip()

    log(f"Agent finished: {elapsed:.1f}s, timed_out={timed_out}, exit={exit_code}")

    return {
        "elapsed": elapsed,
        "timed_out": timed_out,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "diff_stat": diff_stat,
        "turns": turns,
        "turns_failed": fails,
        "errors": jsonl["errors"],
        "usage": jsonl["usage"],
        "completion_reason": jsonl["completion_reason"],
    }


# ===========================================================================
# TASK PARSING
# ===========================================================================

def parse_task(text: str) -> tuple[str, str]:
    """Extract TASK_TITLE and TASK_INSTRUCTION from discovery text."""
    title = ""
    instruction = ""

    if "TASK_TITLE:" in text:
        parts = text.split("TASK_TITLE:", 1)
        after = parts[1]
        if "TASK_INSTRUCTION:" in after:
            title = after.split("TASK_INSTRUCTION:", 1)[0].strip()
        else:
            title = after.split("\n", 1)[0].strip()

    if "TASK_INSTRUCTION:" in text:
        instruction = text.split("TASK_INSTRUCTION:", 1)[1].strip()

    return title, instruction


# ===========================================================================
# VERIFICATION PIPELINE
# ===========================================================================

def verify_syntax(files: set[str]) -> tuple[bool, list[str]]:
    """Check syntax of all modified Python files.

    Returns (all_ok, list_of_errors).
    """
    errors = []
    for fpath in sorted(files):
        if not fpath.endswith(".py"):
            continue
        full = REPO_ROOT / fpath
        if not full.exists():
            continue
        try:
            ast.parse(full.read_text(encoding="utf-8"))
        except SyntaxError as e:
            errors.append(f"{fpath}:{e.lineno}: {e.msg}")
    return len(errors) == 0, errors


def verify_imports(files: set[str]) -> tuple[bool, list[str]]:
    """Compile-check modified Python modules (no side effects).

    Uses compile() rather than import to avoid executing module-level code.
    Returns (all_ok, list_of_errors).
    """
    errors = []
    for fpath in sorted(files):
        if not fpath.endswith(".py") or "__pycache__" in fpath:
            continue
        full = REPO_ROOT / fpath
        if not full.exists():
            continue
        try:
            source = full.read_text(encoding="utf-8")
            compile(source, str(full), "exec")
        except Exception as e:
            errors.append(f"{fpath}: {e}")
    return len(errors) == 0, errors


# ===========================================================================
# BENCHMARK INTEGRATION
# ===========================================================================

def run_benchmarks_safely() -> dict | None:
    """Run the benchmark suite without affecting the working tree.

    Saves git state before benchmarks, restores it after (regardless of
    what the agent might have done during benchmark tasks).
    """
    try:
        from benchmark_suite import run_all_benchmarks
    except ImportError:
        log("benchmark_suite.py not found — skipping benchmarks", "WARN")
        return None

    log("Running benchmark suite...")
    pre_patch = git_save_patch()

    try:
        results = run_all_benchmarks(
            run_agent_fn=lambda instruction, timeout=120, max_iterations=5: run_agent(
                instruction=instruction,
                timeout=timeout,
                max_iterations=max_iterations,
            ),
        )
    except Exception as e:
        log(f"Benchmark suite failed: {e}", "WARN")
        results = None
    finally:
        # Restore working tree to pre-benchmark state
        git_undo_changes()
        if pre_patch.strip():
            git_apply_patch(pre_patch)

    if results:
        log(f"Benchmark score: {results['score']} ({results['passed']}/{results['total']})")

    return results


# ===========================================================================
# PROMPT BUILDERS
# ===========================================================================

def build_discovery_prompt(
    focus: str,
    memory: dict,
    benchmark_scores: dict | None,
    adaptive: dict,
) -> str:
    """Build a context-rich discovery prompt with memory, blacklist, benchmarks."""
    history_text = format_history_for_prompt(memory)

    blacklist = memory.get("blacklist", [])
    blacklist_text = "\n".join(f"- {b}" for b in blacklist) if blacklist else "None yet."

    # Benchmark scores
    bench_text = "Not yet computed."
    if benchmark_scores and "results" in benchmark_scores:
        lines = []
        for name, result in benchmark_scores["results"].items():
            status = "PASS" if result.get("passed") else "FAIL"
            lines.append(f"- {name}: {status}")
        bench_text = "\n".join(lines)
        bench_text += f"\nOverall: {benchmark_scores.get('score', '?')}"

    desc = FOCUS_DESCRIPTIONS.get(focus, focus)

    simplify_note = ""
    if adaptive.get("simplify_tasks"):
        simplify_note = (
            "\n\nIMPORTANT: Recent iterations have FAILED because tasks were too "
            "ambitious. Propose a SMALL, focused change — modify 1-2 functions "
            "in a single file. Prefer safe, incremental improvements."
        )

    syntax_note = ""
    if adaptive.get("emphasize_syntax"):
        syntax_note = (
            "\n\nIMPORTANT: Recent implementations had syntax errors. Ensure your "
            "proposal is straightforward and describe the exact code changes needed."
        )

    prompt = (
        f"You are reviewing a repository containing an autonomous coding agent "
        f"(at `agent/`). Your focus area is: **{focus}** — {desc}.\n\n"
        f"## Previous Iterations (what has already been tried)\n"
        f"{history_text}\n\n"
        f"## Blacklisted Proposals (do NOT propose these again)\n"
        f"{blacklist_text}\n\n"
        f"## Current Benchmark Scores\n"
        f"{bench_text}\n\n"
        "## Codebase Architecture\n"
        "- `agent/src/tools/registry.py` — ToolRegistry class with `_execute_*` methods "
        "(active tool implementations: shell, grep, read_file, write_file, etc.)\n"
        "- `agent/src/tools/grep_files.py` — UNUSED LEGACY code, do NOT propose changes\n"
        "- `agent/src/core/loop.py` — Main agent loop (LLM calls, retries, context mgmt)\n"
        "- `agent/src/core/turn_runtime.py` — Per-turn tool call processing and output truncation\n"
        "- `agent/src/core/compaction.py` — Context pruning and AI-assisted compaction\n"
        "- `agent/src/llm/router.py` — Model selection (fast/default/strong) and fallback chains\n"
        "- `agent/src/llm/client.py` — LLM API client (Chutes provider)\n"
        "- `agent/src/tools/policy.py` — Tool approval policy and execution policy\n"
        "- `agent/src/tools/orchestrator.py` — Tool execution orchestration and guard checks\n"
        "- `agent/src/prompts/system.py` — System prompt (SYSTEM_PROMPT constant)\n"
        "- `agent/src/config/defaults.py` — Default configuration\n\n"
        "## Your Task\n"
        "Read the agent source code in `agent/src/` and identify ONE specific, "
        "high-impact improvement that:\n"
        "1. Is DIFFERENT from all previous attempts listed above\n"
        "2. Is NOT in the blacklist\n"
        "3. Would meaningfully improve the agent's capability in the focus area\n"
        "4. Is concrete and implementable in a single session\n"
        "5. Targets `agent/src/` files (not test files, docs, or scripts)\n"
        "6. Affects actual behavior (not cosmetic/style changes)\n"
        f"{simplify_note}{syntax_note}\n\n"
        "Output EXACTLY this format (no other text before or after):\n\n"
        "TASK_TITLE: <concise title>\n\n"
        "TASK_INSTRUCTION: <detailed instruction: which files, which functions, "
        "current behavior, problem, desired behavior, step-by-step implementation plan>"
    )

    if adaptive.get("add_format_examples"):
        prompt += (
            "\n\n## Example (follow this format exactly)\n\n"
            "TASK_TITLE: Add exponential backoff to tool failure retries\n\n"
            "TASK_INSTRUCTION: In `agent/src/core/loop.py`, the tool failure "
            "handling around line 340 uses a fixed 1-second delay between retries. "
            "This causes retry storms when tools are temporarily unavailable. "
            "Change: (1) In `_handle_tool_result()`, track consecutive failures "
            "in `self._consecutive_failures`. (2) Compute delay as "
            "`min(1.0 * 2**failures, 30.0)`. (3) Apply the delay before the "
            "next LLM call. Files: loop.py only."
        )

    return prompt


def build_impl_prompt(
    title: str,
    instruction: str,
    adaptive: dict,
) -> str:
    """Build the implementation prompt for Cursor CLI agent mode."""
    syntax_note = ""
    if adaptive.get("emphasize_syntax"):
        syntax_note = (
            "\n- CRITICAL: After EVERY file edit, verify syntax with: "
            "`python -c 'import ast; ast.parse(open(\"FILE\").read())'`\n"
            "- If syntax check fails, fix immediately before moving to the next file"
        )

    return (
        "Implement this specific improvement to the coding agent.\n\n"
        f"## What to Do\n"
        f"**Title**: {title}\n\n"
        f"**Instruction**: {instruction}\n\n"
        "## Codebase Architecture\n"
        "- `agent/src/tools/registry.py` — ToolRegistry with `_execute_*` methods "
        "(active tool implementations)\n"
        "- `agent/src/tools/grep_files.py` / `GrepFilesTool` — UNUSED LEGACY, "
        "do NOT modify\n"
        "- `agent/src/core/loop.py` — Main agent loop\n"
        "- `agent/src/core/turn_runtime.py` — Per-turn tool call processing\n"
        "- `agent/src/core/compaction.py` — Context management\n"
        "- `agent/src/llm/router.py` — Model selection/fallback\n"
        "- `agent/src/llm/client.py` — LLM API client\n"
        "- `agent/src/prompts/system.py` — System prompt\n\n"
        "## Rules\n"
        "- Only modify files under `agent/src/` or `agent/agent.py`\n"
        "- Do NOT modify `agent/scripts/` (self-improvement infrastructure)\n"
        "- Make minimal, focused changes\n"
        "- Verify syntax after editing: "
        "`python -c 'import ast; ast.parse(open(\"FILE\").read())'`\n"
        f"{syntax_note}\n"
        "- Do NOT commit changes\n"
        "- Do NOT try to test by running the agent (requires process restart)\n"
        "- Verify correctness by reading code and checking syntax only"
    )


def build_critique_prompt(
    title: str,
    instruction: str,
    diff_stat: str,
    diff_content: str,
) -> str:
    """Build critique prompt to evaluate the implementation diff."""
    return (
        "You are an expert evaluating a code change to an autonomous coding agent.\n\n"
        f"## Task\n"
        f"**Title**: {title}\n"
        f"**Instruction**: {instruction[:800]}\n\n"
        f"## Changes Made\n```\n{diff_stat}\n```\n\n"
        f"## Diff (last 4000 chars)\n```\n{diff_content[-4000:]}\n```\n\n"
        "Evaluate:\n"
        "1. **Correctness** — Does the change implement what was requested?\n"
        "2. **Quality** — Is the code clean, safe, and well-integrated?\n"
        "3. **Impact** — Will this meaningfully improve the agent?\n"
        "4. **Risk** — Could this break existing functionality?\n\n"
        "End your evaluation with exactly: **Score: N/10** (where N is 0-10)"
    )


# ===========================================================================
# ITERATION LOG
# ===========================================================================

def save_iteration_log(iteration: int, data: dict):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = LOG_DIR / f"iter_{iteration:04d}_{ts}.json"
    filepath.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    log(f"Log saved: {filepath}")


# ===========================================================================
# SINGLE ITERATION
# ===========================================================================

def run_iteration(
    iteration: int,
    memory: dict,
    model: str,
    agent_timeout: int,
    skip_impl: bool,
    skip_benchmarks: bool,
    max_agent_iterations: int = 20,
    baseline_benchmark: dict | None = None,
    no_commit: bool = False,
) -> dict:
    """Run one full improvement iteration.

    Flow:
        1. Select focus (adaptive) + build discovery prompt with memory context
        2. Discovery via Cursor CLI → TASK_TITLE + TASK_INSTRUCTION
        3. Implementation via Cursor CLI (agent mode)
        4. Verification: syntax check + import/compile check
        5. Benchmark: run suite, compare to baseline (optional)
        6. Critique: evaluate the diff, extract score
        7. Decision: accept (commit) or revert based on quality gates
        8. Update memory
    """
    data: dict = {
        "iteration": iteration,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # ── Adaptive parameters ───────────────────────────────────────────────
    adaptive = get_adaptive_params(memory)
    if adaptive.get("increase_timeout"):
        agent_timeout = max(agent_timeout, 2400)

    # ── Focus selection ───────────────────────────────────────────────────
    focus = select_focus(memory)
    data["focus"] = focus
    banner(f"ITERATION {iteration} — FOCUS: {focus}")
    log(f"Adaptive params: {adaptive}")

    # ── Step 1: Discovery ─────────────────────────────────────────────────
    banner(f"ITERATION {iteration} — STEP 1: Discovery")

    discovery_prompt = build_discovery_prompt(
        focus=focus,
        memory=memory,
        benchmark_scores=baseline_benchmark,
        adaptive=adaptive,
    )
    discovery_result = cursor_cli(discovery_prompt, mode="ask", model=model)
    log(f"Discovery result:\n{discovery_result[:500]}")
    data["discovery"] = discovery_result

    title, instruction = parse_task(discovery_result)
    if not title or not instruction:
        log("Parse failure — could not extract TASK_TITLE/TASK_INSTRUCTION", "WARN")
        data["error"] = "parse_failure"
        data["outcome"] = "parse_error"
        update_memory(
            memory, iteration=iteration, focus=focus,
            task_title="(parse failure)", score=None, outcome="parse_error",
        )
        return data

    log(f"Task: {title}")
    log(f"Instruction: {instruction[:200]}...")
    data["task_title"] = title
    data["task_instruction"] = instruction

    # ── Skip-impl early exit ──────────────────────────────────────────────
    if skip_impl:
        log("Skipping implementation (--skip-impl)")
        data["implementation"] = "skipped"
        data["outcome"] = "skipped"
        update_memory(
            memory, iteration=iteration, focus=focus,
            task_title=title, score=None, outcome="skipped",
        )
        return data

    # Brief cooldown if rate-limited recently
    if _cursor_cli_consecutive_rate_limits > 0:
        cooldown = min(15.0 * _cursor_cli_consecutive_rate_limits, 60.0)
        log(f"Cooldown {cooldown:.0f}s (rate-limit history)")
        time.sleep(cooldown)

    # ── Step 2: Implementation ────────────────────────────────────────────
    banner(f"ITERATION {iteration} — STEP 2: Implementation")

    pre_patch = git_save_patch()
    pre_modified = git_diff_name_only()

    impl_prompt = build_impl_prompt(title, instruction, adaptive)
    impl_result = cursor_cli(impl_prompt, mode="agent", model=model)

    data["implementation"] = impl_result[:2000]

    post_modified = git_diff_name_only()
    impl_files = post_modified - pre_modified
    # Also check files that were already modified but might have new changes
    all_impl_files = impl_files  # For verification we check all modified agent files

    diff_stat = git_diff_stat()
    data["implementation_diff"] = diff_stat
    data["files_changed"] = sorted(impl_files)

    log(f"Implementation diff:\n{diff_stat}")
    log(f"New files modified: {sorted(impl_files)}")

    if not impl_files:
        # Check if ANY file content changed (implementation may have modified
        # an already-dirty file)
        new_patch = git_save_patch()
        if new_patch.strip() == pre_patch.strip():
            log("Implementation produced no changes at all", "WARN")
            data["outcome"] = "no_changes"
            update_memory(
                memory, iteration=iteration, focus=focus,
                task_title=title, score=0, outcome="crash",
                summary="Implementation produced no file changes",
            )
            return data
        else:
            log("Implementation modified pre-existing files (cannot commit separately)")
            # Still proceed with verification — changes exist but in pre-modified files
            all_impl_files = post_modified

    # ── Step 3: Verification ──────────────────────────────────────────────
    banner(f"ITERATION {iteration} — STEP 3: Verification")

    # Filter to only agent source files for verification
    agent_files = {f for f in (impl_files | post_modified) if f.startswith("agent/")}

    # 3a: Syntax check
    syntax_ok, syntax_errors = verify_syntax(agent_files)
    data["syntax_ok"] = syntax_ok
    data["syntax_errors"] = syntax_errors

    if not syntax_ok:
        log(f"SYNTAX ERRORS — reverting:\n  " + "\n  ".join(syntax_errors), "WARN")
        git_revert_to_patch(pre_patch)
        data["outcome"] = "syntax_error"
        update_memory(
            memory, iteration=iteration, focus=focus,
            task_title=title, score=0, outcome="syntax_error",
            files_changed=sorted(impl_files),
            summary=f"Syntax errors: {'; '.join(syntax_errors[:3])}",
        )
        return data

    # 3b: Import/compile check
    import_ok, import_errors = verify_imports(agent_files)
    data["import_ok"] = import_ok
    data["import_errors"] = import_errors

    if not import_ok:
        log(f"IMPORT ERRORS — reverting:\n  " + "\n  ".join(import_errors), "WARN")
        git_revert_to_patch(pre_patch)
        data["outcome"] = "import_error"
        update_memory(
            memory, iteration=iteration, focus=focus,
            task_title=title, score=0, outcome="import_error",
            files_changed=sorted(impl_files),
            summary=f"Import errors: {'; '.join(import_errors[:3])}",
        )
        return data

    log("Verification PASSED (syntax + imports)")

    # ── Step 4: Benchmark (optional) ──────────────────────────────────────
    benchmark_result = None
    if not skip_benchmarks and baseline_benchmark is not None:
        banner(f"ITERATION {iteration} — STEP 4: Benchmark")

        benchmark_result = run_benchmarks_safely()
        data["benchmark"] = benchmark_result

        if benchmark_result:
            baseline_score = baseline_benchmark.get("passed", 0)
            new_score = benchmark_result.get("passed", 0)
            log(
                f"Benchmark: {new_score}/{benchmark_result['total']} "
                f"(baseline: {baseline_score}/{baseline_benchmark['total']})"
            )
            if new_score < baseline_score:
                log("BENCHMARK REGRESSION — reverting", "WARN")
                git_revert_to_patch(pre_patch)
                data["outcome"] = "regression"
                update_memory(
                    memory, iteration=iteration, focus=focus,
                    task_title=title, score=0, outcome="regression",
                    files_changed=sorted(impl_files),
                    summary=f"Benchmark regression: {new_score} < {baseline_score}",
                )
                return data

    # ── Step 5: Critique ──────────────────────────────────────────────────
    if _cursor_cli_consecutive_rate_limits > 0:
        cooldown = min(15.0 * _cursor_cli_consecutive_rate_limits, 60.0)
        log(f"Cooldown {cooldown:.0f}s before critique")
        time.sleep(cooldown)

    banner(f"ITERATION {iteration} — STEP 5: Critique")

    diff_content = git_diff_content()
    critique_prompt = build_critique_prompt(title, instruction, diff_stat, diff_content)
    critique_result = cursor_cli(critique_prompt, mode="ask", model=model)

    data["critique"] = critique_result
    score = extract_score(critique_result)
    data["score"] = score
    log(f"Critique score: {score}/10" if score is not None else "Critique score: N/A")

    # ── Step 6: Decision ──────────────────────────────────────────────────
    banner(f"ITERATION {iteration} — STEP 6: Decision")

    # Quality gate: require score >= 3 (or accept if score couldn't be parsed)
    min_score = 3
    if score is not None and score < min_score:
        log(f"Score {score}/10 below threshold {min_score} — reverting", "WARN")
        git_revert_to_patch(pre_patch)
        data["outcome"] = "low_score"
        update_memory(
            memory, iteration=iteration, focus=focus,
            task_title=title, score=score, outcome="reverted",
            files_changed=sorted(impl_files),
            summary=f"Score {score}/10 below threshold {min_score}",
        )
        return data

    # All gates passed — accept the improvement
    committed = False
    if impl_files and not no_commit:
        committed = git_commit_improvement(title, impl_files, iteration)

    outcome = "landed"
    data["outcome"] = outcome
    data["committed"] = committed
    log(
        f"ACCEPTED: {title} "
        f"(score: {score}/10, committed: {committed}, "
        f"files: {sorted(impl_files) or 'pre-existing only'})"
    )

    update_memory(
        memory, iteration=iteration, focus=focus,
        task_title=title, score=score, outcome="landed",
        files_changed=sorted(impl_files),
        summary=f"Score {score}/10, committed={committed}",
    )

    data["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return data


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Agent self-improvement loop with memory, benchmarks, and quality gates",
    )
    parser.add_argument(
        "--iterations", type=int, default=0,
        help="Number of iterations (0 = infinite, default: 0)",
    )
    parser.add_argument(
        "--timeout", type=int, default=1800,
        help="Agent execution timeout in seconds (default: 1800)",
    )
    parser.add_argument(
        "--skip-impl", action="store_true",
        help="Discovery only — no implementation",
    )
    parser.add_argument(
        "--skip-benchmarks", action="store_true",
        help="Skip benchmark suite (faster iterations)",
    )
    parser.add_argument(
        "--model", default="opus-4.6-thinking",
        help="Cursor model for discovery/critique/implementation (default: opus-4.6-thinking)",
    )
    parser.add_argument(
        "--pause", type=int, default=10,
        help="Pause between iterations in seconds (default: 10)",
    )
    parser.add_argument(
        "--max-agent-iterations", type=int, default=20,
        help="Max agent loop iterations per step (default: 20)",
    )
    parser.add_argument(
        "--no-commit", action="store_true",
        help="Don't commit successful improvements (keep in working tree)",
    )
    args = parser.parse_args()

    banner("SELF-IMPROVEMENT LOOP — PREFLIGHT")

    # ── Load memory ───────────────────────────────────────────────────────
    memory = load_memory()
    log(f"Repo root:       {REPO_ROOT}")
    log(f"Memory:          {memory['iterations_total']} total iterations, "
        f"{memory['improvements_landed']} landed, "
        f"{memory['improvements_reverted']} reverted, "
        f"{len(memory['blacklist'])} blacklisted")
    log(f"Model:           {args.model}")
    log(f"Iterations:      {'infinite' if args.iterations == 0 else args.iterations}")
    log(f"Agent timeout:   {args.timeout}s")
    log(f"Agent max iter:  {args.max_agent_iterations}")
    log(f"Benchmarks:      {'enabled' if not args.skip_benchmarks else 'disabled'}")
    log(f"Auto-commit:     {'yes' if not args.no_commit else 'no'}")

    # ── Focus scores summary ──────────────────────────────────────────────
    for area, stats in memory.get("focus_scores", {}).items():
        landed = stats.get("landed_count", 0)
        avg = stats.get("avg_score", 0.0)
        attempts = stats.get("attempts", 0)
        log(f"  {area}: {landed} landed / {attempts} attempts (avg score: {avg:.1f})")

    # ── Baseline benchmarks ───────────────────────────────────────────────
    baseline_benchmark = None
    if not args.skip_benchmarks and not args.skip_impl:
        banner("BASELINE BENCHMARK")
        baseline_benchmark = run_benchmarks_safely()
        if baseline_benchmark:
            log(f"Baseline score: {baseline_benchmark['score']}")
            memory["last_benchmark"] = baseline_benchmark
            save_memory(memory)
        else:
            log("Baseline benchmark failed or unavailable — continuing without", "WARN")

    # ── Main loop ─────────────────────────────────────────────────────────
    iteration = memory.get("iterations_total", 0)
    results = []

    try:
        i = 0
        while True:
            i += 1
            iteration += 1
            if args.iterations > 0 and i > args.iterations:
                break

            result = run_iteration(
                iteration=iteration,
                memory=memory,
                model=args.model,
                agent_timeout=args.timeout,
                skip_impl=args.skip_impl,
                skip_benchmarks=args.skip_benchmarks,
                max_agent_iterations=args.max_agent_iterations,
                baseline_benchmark=baseline_benchmark,
                no_commit=args.no_commit,
            )
            results.append(result)
            save_iteration_log(iteration, result)

            # Update baseline if improvement landed and we have new benchmark data
            if result.get("outcome") == "landed" and result.get("benchmark"):
                baseline_benchmark = result["benchmark"]
                log("Updated baseline benchmark after successful improvement")

            if args.iterations > 0 and i >= args.iterations:
                break

            log(f"Pausing {args.pause}s before next iteration...")
            time.sleep(args.pause)

    except KeyboardInterrupt:
        log("\nInterrupted by user", "WARN")

    # ── Summary ───────────────────────────────────────────────────────────
    banner("SUMMARY")
    log(f"Completed {len(results)} iterations this session")

    for r in results:
        title = r.get("task_title", "unknown")
        outcome = r.get("outcome", "?")
        score = r.get("score")
        score_str = f"{score}/10" if score is not None else "N/A"
        committed = "✓" if r.get("committed") else "✗"
        log(
            f"  Iter {r.get('iteration', '?')}: [{outcome}] {title} "
            f"(score: {score_str}, committed: {committed})"
        )

    landed = sum(1 for r in results if r.get("outcome") == "landed")
    failed = sum(1 for r in results if r.get("outcome") not in ("landed", "skipped"))
    log(f"\nSession: {landed} landed, {failed} failed out of {len(results)} total")
    log(f"All-time: {memory['iterations_total']} iterations, "
        f"{memory['improvements_landed']} landed")
    log(f"Memory: {MEMORY_FILE}")
    log(f"Logs: {LOG_DIR}")


if __name__ == "__main__":
    main()
