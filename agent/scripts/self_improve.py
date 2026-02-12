#!/usr/bin/env python3
"""
Rollout-driven self-improvement loop for the agent.

Generates coding tasks, runs the agent on them, captures the full rollout
trajectory, analyzes the agent's behavior to find weaknesses, and implements
targeted improvements via Cursor CLI.

Flow per iteration:
    1. Task generation — Cursor CLI generates a coding task for the agent
    2. Agent execution — agent.py runs the task, producing a full rollout
    3. Rollout review — Cursor CLI reads the trajectory, identifies weaknesses,
       and implements improvements to agent source code
    4. Verification — syntax + import checks with retry loop
    5. Decision — accept (commit) or revert

Features:
- Rollout-driven discovery (real agent behavior informs improvements)
- Institutional memory tracking all past iterations and outcomes
- Pre-flight health check (detects agent startup failures before wasting cycles)
- Quality gates: syntax + import verification
- Implementation retry loop (2 fix attempts before revert)
- Adaptive focus selection (curriculum-gated, weakest areas get priority)
- Git accumulation (successful improvements are committed)
- Failure mode adaptation (auto-adjusts based on recent failures)

Usage:
    python3 agent/scripts/self_improve.py
    python3 agent/scripts/self_improve.py --iterations 5
    python3 agent/scripts/self_improve.py --model claude-4.5-sonnet
"""

from __future__ import annotations

import argparse
import ast
import datetime
import json
import os
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


# Directories that git-clean should never touch (relative to REPO_ROOT)
CLEAN_EXCLUDES = [
    "agent/scripts/",
    "agent/.agent/",
]

# ---------------------------------------------------------------------------
# Focus areas
# ---------------------------------------------------------------------------
FOCUS_AREAS = [
    "turn efficiency and speed",
    "token and cost efficiency",
    "tool call precision",
    "first-attempt accuracy",
    "error recovery speed",
    "prompt and reasoning quality",
]

FOCUS_DESCRIPTIONS = {
    "turn efficiency and speed": (
        "minimizing the number of turns/iterations to complete a task — the agent "
        "should solve tasks in as few turns as possible by making decisive, high-impact "
        "actions each turn instead of exploratory or redundant ones"
    ),
    "token and cost efficiency": (
        "reducing input/output token usage per task — aggressive caching, shorter prompts, "
        "compact tool outputs, avoiding re-reading files unnecessarily, smarter context "
        "management to keep conversations lean"
    ),
    "tool call precision": (
        "making every tool call count — no wasted shell commands, no redundant reads, "
        "no grep-then-read-then-grep cycles. The agent should use the right tool with "
        "the right arguments the first time"
    ),
    "first-attempt accuracy": (
        "getting the correct answer/implementation on the first try — writing correct "
        "code without needing fix-up iterations, reading instructions carefully, producing "
        "complete and correct outputs"
    ),
    "error recovery speed": (
        "when errors occur, recovering in minimal turns — fast diagnosis, targeted fixes, "
        "no flailing. Should identify root cause immediately rather than trying random fixes"
    ),
    "prompt and reasoning quality": (
        "system prompt design, preamble quality, reasoning chains — does the agent think "
        "clearly and efficiently before acting, or does it waste tokens on verbose reasoning "
        "and unnecessary planning"
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
            "syntax_error": 0,
            "import_error": 0,
        },
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
# CURRICULUM & ADAPTIVE FOCUS SELECTION
# ===========================================================================

# Progression tiers: each tier gates which focus areas are eligible.
#
# Tier 0 (STARTUP):  Agent can't even start → fix imports/startup errors
# Tier 1 (BASIC):    Agent starts → all areas eligible (default)

CURRICULUM_TIERS = {
    0: {
        "name": "startup",
        "description": "Agent cannot start — fix import/startup errors",
        "focus_areas": [
            "error handling and recovery",
            "tool implementation",
        ],
    },
    1: {
        "name": "basic",
        "description": "Agent starts — all areas eligible",
        "focus_areas": FOCUS_AREAS,
    },
}


def get_curriculum_tier(agent_healthy: bool = True) -> int:
    """Determine the current curriculum tier based on agent health."""
    return 0 if not agent_healthy else 1


def select_focus(
    memory: dict,
    agent_healthy: bool = True,
) -> str:
    """Select the focus area most in need of improvement.

    Uses the curriculum tier to gate which focus areas are eligible,
    then prioritizes areas with lower scores and fewer landed improvements.
    """
    tier = get_curriculum_tier(agent_healthy)
    tier_info = CURRICULUM_TIERS.get(tier, CURRICULUM_TIERS[1])
    eligible_areas = tier_info["focus_areas"]

    log(f"Curriculum tier: {tier} ({tier_info['name']}) — "
        f"eligible areas: {eligible_areas}")

    focus_scores = memory.get("focus_scores", {})

    candidates = []
    for area in eligible_areas:
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

    if not candidates:
        return FOCUS_AREAS[0]

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
            "--output-format", "stream-json",
            "--force",
            prompt,
        ]
        # --mode only supports "plan" and "ask". The default (no --mode)
        # is full agent mode with read+write access. Passing an invalid
        # mode like "agent" causes the CLI to silently exit with no output.
        if mode in ("ask", "plan"):
            cmd.insert(4, "--mode")
            cmd.insert(5, mode)

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

        # Log exit code and stderr for diagnostics
        if proc.returncode != 0:
            log(f"Cursor CLI exited with code {proc.returncode} (mode={mode})", "WARN")
            if stderr_text:
                # Show last 500 chars of stderr for debugging
                stderr_tail = stderr_text.strip()[-500:]
                log(f"Cursor CLI stderr: {stderr_tail}", "WARN")

        # Warn on empty result (common failure mode)
        if not result_text.strip():
            log(f"Cursor CLI returned EMPTY result (mode={mode}, "
                f"exit={proc.returncode})", "WARN")
            if stderr_text:
                stderr_tail = stderr_text.strip()[-300:]
                log(f"  stderr hint: {stderr_tail}", "WARN")

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

        # Retry on empty result with non-zero exit (likely transient failure)
        if not result_text.strip() and proc.returncode != 0 and attempt < max_retries:
            log(f"Empty result with non-zero exit — retrying (attempt {attempt}/{max_retries})",
                "WARN")
            time.sleep(5.0)
            continue

        if not rate_limited:
            _cursor_cli_consecutive_rate_limits = 0

        return result_text

    log("Cursor CLI retries exhausted", "WARN")
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
# AGENT-SPECIFIC DIFF HELPERS
# ===========================================================================

def _compute_agent_diff(
    pre_contents: dict[str, str],
    impl_files: set[str],
) -> str:
    """Compute a unified diff showing ONLY changes the agent made.

    For pre-dirty files, this diffs the pre-agent content against the current
    content — isolating just the agent's edits from pre-existing modifications.
    For new files, it diffs against an empty string.
    """
    import difflib

    diffs: list[str] = []
    for f in sorted(impl_files):
        old = pre_contents.get(f, "")
        full = REPO_ROOT / f
        try:
            new = full.read_text(encoding="utf-8") if full.exists() else ""
        except Exception:
            new = ""
        if old == new:
            continue
        diff = "\n".join(difflib.unified_diff(
            old.splitlines(), new.splitlines(),
            fromfile=f"a/{f}", tofile=f"b/{f}", lineterm="",
        ))
        if diff:
            diffs.append(diff)
    return "\n".join(diffs)


def _summarize_diff(diff_content: str, impl_files: set[str]) -> str:
    """Build a short stat-like summary from a unified diff string."""
    if not diff_content:
        return "(no diff)"
    insertions = sum(1 for line in diff_content.splitlines() if line.startswith("+") and not line.startswith("+++"))
    deletions = sum(1 for line in diff_content.splitlines() if line.startswith("-") and not line.startswith("---"))
    files_str = ", ".join(sorted(impl_files))
    return f"{len(impl_files)} file(s) changed ({files_str}), {insertions} insertions(+), {deletions} deletions(-)"


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
        "session_id": jsonl["session_id"],
    }


# ===========================================================================
# ROLLOUT CAPTURE
# ===========================================================================

def capture_rollout(
    session_id: str | None,
    cwd: Path | None = None,
) -> tuple[dict | None, str]:
    """Locate and read the agent's session rollout file.

    Returns (raw_rollout_dict, summary_text).
    The summary is a human-readable trajectory suitable for inclusion in prompts.
    The raw dict contains the full message history for deep analysis.
    """
    if not session_id:
        return None, "(no session_id — rollout unavailable)"

    base = (cwd or REPO_ROOT) / ".agent" / "sessions"
    rollout_path = base / f"{session_id}.json"

    if not rollout_path.exists():
        log(f"Rollout file not found: {rollout_path}", "WARN")
        return None, f"(rollout file not found at {rollout_path})"

    try:
        raw = json.loads(rollout_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        log(f"Failed to read rollout: {e}", "WARN")
        return None, f"(failed to read rollout: {e})"

    summary = _summarize_rollout(raw)
    return raw, summary


def _summarize_rollout(rollout: dict) -> str:
    """Produce a structured text summary of a rollout for prompt inclusion.

    Extracts assistant reasoning, tool calls, tool results (truncated),
    and errors from the full message history.
    """
    lines: list[str] = []

    session_id = rollout.get("session_id", "?")
    iteration = rollout.get("iteration", "?")
    usage = rollout.get("usage", {})
    tool_call_count = rollout.get("tool_call_count", 0)

    lines.append(f"Session: {session_id}")
    lines.append(f"Iterations: {iteration}")
    lines.append(f"Tool calls: {tool_call_count}")
    lines.append(
        f"Tokens: input={usage.get('input_tokens', 0)}, "
        f"output={usage.get('output_tokens', 0)}, "
        f"cached={usage.get('cached_input_tokens', 0)}"
    )
    lines.append("")
    lines.append("## Trajectory")

    messages = rollout.get("messages", [])
    turn_num = 0

    for msg in messages:
        role = msg.get("role", "")

        if role == "assistant":
            turn_num += 1
            # Extract reasoning text
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                snippet = content.strip().replace("\n", " ")[:300]
                lines.append(f"\n--- Turn {turn_num} ---")
                lines.append(f"[assistant] {snippet}")
            elif isinstance(content, list):
                texts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        texts.append(block.get("text", ""))
                combined = " ".join(texts).strip().replace("\n", " ")[:300]
                if combined:
                    lines.append(f"\n--- Turn {turn_num} ---")
                    lines.append(f"[assistant] {combined}")

            # Extract tool calls
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "?")
                args_raw = func.get("arguments", "")
                # Parse and truncate arguments
                try:
                    args_dict = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    # Truncate long argument values
                    if isinstance(args_dict, dict):
                        truncated_args = {}
                        for k, v in args_dict.items():
                            sv = str(v)
                            truncated_args[k] = sv[:200] + "..." if len(sv) > 200 else sv
                        args_str = json.dumps(truncated_args, ensure_ascii=False)
                    else:
                        args_str = str(args_dict)[:300]
                except (json.JSONDecodeError, TypeError):
                    args_str = str(args_raw)[:300]
                lines.append(f"  [tool_call] {name}({args_str})")

        elif role == "tool":
            # Tool result — truncate heavily
            content = msg.get("content", "")
            if isinstance(content, str):
                snippet = content.strip().replace("\n", " ")[:200]
            else:
                snippet = str(content)[:200]
            if snippet:
                lines.append(f"  [tool_result] {snippet}")

    return "\n".join(lines)


def get_rollout_path(session_id: str | None, cwd: Path | None = None) -> Path | None:
    """Return the path to a rollout file, or None if it doesn't exist."""
    if not session_id:
        return None
    base = (cwd or REPO_ROOT) / ".agent" / "sessions"
    path = base / f"{session_id}.json"
    return path if path.exists() else None


# ===========================================================================
# PRE-FLIGHT HEALTH CHECK
# ===========================================================================

def preflight_check() -> tuple[bool, str]:
    """Verify the agent can at least start up without import errors.

    Returns (healthy, error_message).
    """
    python = _venv_python()
    agent_py = AGENT_DIR / "agent.py"
    if not agent_py.exists():
        return False, f"agent.py not found at {agent_py}"

    # Try to import the agent's main modules
    check_script = (
        "import sys; sys.path.insert(0, 'agent'); "
        "from src.tools.registry import ToolRegistry; "
        "from src.core.loop import run_agent_loop; "
        "from src.llm.client import LLMClient; "
        "print('OK')"
    )
    result = subprocess.run(
        [python, "-c", check_script],
        cwd=str(REPO_ROOT),
        capture_output=True, text=True,
        timeout=30,
    )
    if result.returncode == 0 and "OK" in result.stdout:
        return True, ""

    error = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "Unknown error"
    return False, error


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
# PROMPT BUILDERS
# ===========================================================================

CODEBASE_ARCHITECTURE = (
    "## Codebase Architecture\n"
    "- `agent/src/tools/registry.py` — ToolRegistry class with `_execute_*` methods "
    "(active tool implementations: shell, grep, read_file, write_file, etc.)\n"
    "- `agent/src/tools/grep_files.py` — UNUSED LEGACY code, do NOT modify\n"
    "- `agent/src/core/loop.py` — Main agent loop (LLM calls, retries, context mgmt)\n"
    "- `agent/src/core/turn_runtime.py` — Per-turn tool call processing and output truncation\n"
    "- `agent/src/core/compaction.py` — Context pruning and AI-assisted compaction\n"
    "- `agent/src/llm/router.py` — Model selection (fast/default/strong) and fallback chains\n"
    "- `agent/src/llm/client.py` — LLM API client (Chutes provider)\n"
    "- `agent/src/tools/policy.py` — Tool approval policy and execution policy\n"
    "- `agent/src/tools/orchestrator.py` — Tool execution orchestration and guard checks\n"
    "- `agent/src/prompts/system.py` — System prompt (SYSTEM_PROMPT constant)\n"
    "- `agent/src/config/defaults.py` — Default configuration\n"
)


def build_task_generation_prompt(
    focus: str,
    memory: dict,
    adaptive: dict,
    preflight_error: str = "",
) -> str:
    """Build a prompt for Cursor CLI to generate a coding task for the agent.

    The task is a real coding challenge the agent will attempt to solve.
    The improvement comes from analyzing HOW the agent solves (or fails to
    solve) the task, not from the task itself.
    """
    history_text = format_history_for_prompt(memory)

    blacklist = memory.get("blacklist", [])
    blacklist_text = "\n".join(f"- {b}" for b in blacklist) if blacklist else "None yet."

    desc = FOCUS_DESCRIPTIONS.get(focus, focus)

    # Pre-flight error override
    preflight_section = ""
    if preflight_error:
        preflight_section = (
            f"\n## CRITICAL: Agent Startup Failure\n"
            f"The agent CANNOT START due to: `{preflight_error}`\n"
            f"Generate a very simple task (e.g., create a single file) so we can "
            f"observe the startup failure in the rollout.\n"
        )

    simplify_note = ""
    if adaptive.get("simplify_tasks"):
        simplify_note = (
            "\n\nIMPORTANT: Recent iterations have FAILED. Generate a SIMPLER task "
            "that the agent should be able to handle — single-file operations, "
            "straightforward instructions."
        )

    return (
        "You are generating a coding task for an autonomous coding agent to attempt. "
        "The agent will try to solve this task, and we will analyze its behavior "
        "(the full rollout of tool calls, reasoning, and errors) to find weaknesses "
        "and improve the agent's code.\n\n"
        f"The current improvement focus area is: **{focus}** — {desc}.\n"
        f"{preflight_section}\n"
        f"## Previous Iterations\n"
        f"{history_text}\n\n"
        f"## Blacklisted Tasks (do NOT generate these)\n"
        f"{blacklist_text}\n\n"
        "## Task Design Guidelines\n"
        "Generate a task that will EXERCISE the focus area and expose weaknesses:\n"
        f"- For **{focus}**: design a task that requires the agent to demonstrate "
        f"strong {focus} skills\n"
        "- The task should be completable by a competent agent in under 5 minutes\n"
        "- The task should work in a sandbox directory (the agent will be given a "
        "working directory path)\n"
        "- Make the task concrete and verifiable (specific expected outputs)\n"
        "- Vary the difficulty and type from previous iterations\n"
        "- The task should be DIFFERENT from tasks already tried above\n"
        f"{simplify_note}\n\n"
        "Output EXACTLY this format (no other text before or after):\n\n"
        "TASK_INSTRUCTION: <a complete, self-contained instruction for the agent "
        "to execute. Be specific about expected files, content, and behavior.>"
    )


def build_rollout_review_prompt(
    task_instruction: str,
    rollout_summary: str,
    rollout_path: str | None,
    focus: str,
    memory: dict,
    adaptive: dict,
    agent_result: dict | None = None,
) -> str:
    """Build prompt for Cursor CLI to review an agent rollout and implement improvements.

    This is the core of the new rollout-driven self-improvement loop. Cursor
    reads the agent's full trajectory, identifies concrete weaknesses, traces
    them to specific code, and implements targeted fixes.
    """
    history_text = format_history_for_prompt(memory)

    blacklist = memory.get("blacklist", [])
    blacklist_text = "\n".join(f"- {b}" for b in blacklist) if blacklist else "None yet."

    desc = FOCUS_DESCRIPTIONS.get(focus, focus)

    # Agent execution summary
    exec_summary = ""
    if agent_result:
        exec_summary = (
            f"\n## Agent Execution Summary\n"
            f"- Duration: {agent_result.get('elapsed', 0):.1f}s\n"
            f"- Turns completed: {agent_result.get('turns', 0)}\n"
            f"- Turns failed: {agent_result.get('turns_failed', 0)}\n"
            f"- Exit code: {agent_result.get('exit_code', -1)}\n"
            f"- Timed out: {agent_result.get('timed_out', False)}\n"
            f"- Errors: {agent_result.get('errors', [])}\n"
        )
        usage = agent_result.get("usage", {})
        if usage:
            exec_summary += (
                f"- Tokens: input={usage.get('input_tokens', 0)}, "
                f"output={usage.get('output_tokens', 0)}, "
                f"cached={usage.get('cached_input_tokens', 0)}\n"
            )

    # Rollout file section
    rollout_file_section = ""
    if rollout_path:
        rollout_file_section = (
            f"\n## Full Rollout File\n"
            f"The complete agent trajectory (all messages, tool calls, and results) "
            f"is saved at: `{rollout_path}`\n"
            f"**Read this file** to see the full sequence of actions the agent took.\n"
        )

    syntax_note = ""
    if adaptive.get("emphasize_syntax"):
        syntax_note = (
            "\n- CRITICAL: After EVERY file edit, verify syntax with: "
            "`python -c 'import ast; ast.parse(open(\"FILE\").read())'`\n"
            "- If syntax check fails, fix immediately before moving to the next file"
        )

    return (
        "You are a performance engineer analyzing an autonomous coding agent's behavior. "
        "Your PRIMARY objective is to make this agent FASTER — fewer turns, fewer tokens, "
        "less wall-clock time per task. Your SECONDARY objective is to make it more ACCURATE "
        "— correct output on the first attempt, fewer retries, fewer errors.\n\n"
        "Speed is the #1 priority. Every wasted turn, every redundant file read, every "
        "unnecessary grep, every verbose reasoning block is a bug to fix.\n\n"
        f"## Specific Focus: **{focus}** — {desc}\n"
        f"{exec_summary}"
        f"\n## Task Given to the Agent\n"
        f"```\n{task_instruction}\n```\n"
        f"\n## Rollout Summary (agent's trajectory)\n"
        f"```\n{rollout_summary}\n```\n"
        f"{rollout_file_section}\n"
        f"\n{CODEBASE_ARCHITECTURE}\n"
        f"\n## Previous Improvement Iterations\n"
        f"{history_text}\n\n"
        f"## Blacklisted Improvements (do NOT implement these)\n"
        f"{blacklist_text}\n\n"
        "## Your Analysis Task\n"
        "1. **Read the full rollout file** at the path above\n"
        "2. **Quantify the waste** — answer these questions:\n"
        "   - How many turns did the agent take? What's the MINIMUM turns a skilled "
        "developer would need?\n"
        "   - How many tool calls were redundant or could be combined?\n"
        "   - How many tokens were wasted on verbose reasoning, re-reading files, "
        "or unnecessary exploration?\n"
        "   - Did the agent get the right answer on the first attempt? If not, how "
        "many fix-up iterations did it need?\n"
        "   - Did the agent do things sequentially that could be parallelized?\n"
        "3. **Identify the root cause in code** — trace the inefficiency to specific "
        "code in `agent/src/`. Common culprits:\n"
        "   - System prompt encouraging over-exploration or verbose thinking\n"
        "   - Tool implementations that return too much data\n"
        "   - Missing tool output truncation or caching\n"
        "   - Loop logic that doesn't exit early enough\n"
        "   - Context management not aggressive enough (bloated conversations)\n"
        "   - Redundant verification steps\n"
        "   - Model selection using expensive models for simple operations\n"
        "4. **Implement a targeted performance fix** — modify the agent's source code "
        "to eliminate the waste you identified. The fix should measurably reduce either "
        "turns, tokens, or wall-clock time.\n\n"
        "## Rules\n"
        "- Only modify files under `agent/src/` or `agent/agent.py`\n"
        "- Do NOT modify `agent/scripts/` (self-improvement infrastructure)\n"
        "- Make minimal, focused changes — fix ONE performance bottleneck per iteration\n"
        "- Prefer changes that reduce turns/tokens over cosmetic refactors\n"
        "- The improvement must be DIFFERENT from previous iterations listed above\n"
        "- The improvement must NOT be in the blacklist\n"
        "- Verify syntax after editing: "
        "`python -c 'import ast; ast.parse(open(\"FILE\").read())'`\n"
        f"{syntax_note}\n"
        "- Do NOT commit changes\n"
        "- Do NOT try to test by running the agent"
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
    max_agent_iterations: int = 20,
    no_commit: bool = False,
    agent_healthy: bool = True,
    preflight_error: str = "",
) -> dict:
    """Run one full rollout-driven improvement iteration.

    Flow:
        1. Select focus (adaptive + curriculum tier)
        2. Task generation via Cursor CLI (ask mode)
        3. Agent execution on the task (with rollout capture + revert)
        4. Rollout review + improvement via Cursor CLI (agent mode)
        5. Verification: syntax check + import/compile check (with retries)
        6. Decision: accept (commit) or revert
        7. Update memory
    """
    data: dict = {
        "iteration": iteration,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # ── Adaptive parameters ───────────────────────────────────────────────
    adaptive = get_adaptive_params(memory)
    if adaptive.get("increase_timeout"):
        agent_timeout = max(agent_timeout, 2400)

    # ── Step 1: Focus selection (curriculum-aware) ─────────────────────────
    focus = select_focus(memory, agent_healthy=agent_healthy)
    data["focus"] = focus
    banner(f"ITERATION {iteration} — FOCUS: {focus}")
    log(f"Adaptive params: {adaptive}")

    # ── Step 2: Task generation ───────────────────────────────────────────
    banner(f"ITERATION {iteration} — STEP 2: Task Generation")

    task_prompt = build_task_generation_prompt(
        focus=focus,
        memory=memory,
        adaptive=adaptive,
        preflight_error=preflight_error,
    )
    task_result = cursor_cli(task_prompt, mode="ask", model=model)
    log(f"Task generation result:\n{task_result[:500]}")
    data["task_generation"] = task_result

    # Extract the task instruction
    _, task_instruction = parse_task(task_result)
    if not task_instruction:
        log("Parse failure — could not extract TASK_INSTRUCTION", "WARN")
        data["error"] = "parse_failure"
        data["outcome"] = "parse_error"
        update_memory(
            memory, iteration=iteration, focus=focus,
            task_title="(parse failure)", score=None, outcome="parse_error",
        )
        return data

    # Use a truncated version as the title for logging/memory
    task_title = task_instruction[:80].replace("\n", " ").strip()
    if len(task_instruction) > 80:
        task_title += "..."

    log(f"Task: {task_title}")
    log(f"Instruction: {task_instruction[:200]}...")
    data["task_title"] = task_title
    data["task_instruction"] = task_instruction

    # Brief cooldown if rate-limited recently
    if _cursor_cli_consecutive_rate_limits > 0:
        cooldown = min(15.0 * _cursor_cli_consecutive_rate_limits, 60.0)
        log(f"Cooldown {cooldown:.0f}s (rate-limit history)")
        time.sleep(cooldown)

    # ── Step 3: Agent execution (with rollout isolation) ──────────────────
    banner(f"ITERATION {iteration} — STEP 3: Agent Execution")

    # Save git state so we can revert agent's task artifacts
    pre_patch = git_save_patch()

    agent_result = run_agent(
        instruction=task_instruction,
        timeout=agent_timeout,
        max_iterations=max_agent_iterations,
    )

    data["agent_result"] = {
        "elapsed": agent_result.get("elapsed"),
        "timed_out": agent_result.get("timed_out"),
        "exit_code": agent_result.get("exit_code"),
        "turns": agent_result.get("turns"),
        "turns_failed": agent_result.get("turns_failed"),
        "errors": agent_result.get("errors"),
        "usage": agent_result.get("usage"),
        "session_id": agent_result.get("session_id"),
    }

    # Capture the rollout before reverting
    session_id = agent_result.get("session_id")
    rollout_raw, rollout_summary = capture_rollout(session_id)
    rollout_path = get_rollout_path(session_id)

    data["rollout_summary"] = rollout_summary[:2000]
    data["rollout_path"] = str(rollout_path) if rollout_path else None

    log(f"Agent completed: {agent_result.get('turns', 0)} turns, "
        f"{agent_result.get('elapsed', 0):.1f}s, "
        f"exit={agent_result.get('exit_code', -1)}")

    if not rollout_raw:
        log("No rollout captured — agent may have crashed before producing output", "WARN")

    # Revert ALL agent changes (task artifacts, not improvements)
    git_undo_changes()
    if pre_patch.strip():
        git_apply_patch(pre_patch)
    log("Reverted agent task artifacts — working tree restored")

    # ── Step 4: Rollout review + improvement ──────────────────────────────
    banner(f"ITERATION {iteration} — STEP 4: Rollout Review + Improvement")

    if _cursor_cli_consecutive_rate_limits > 0:
        cooldown = min(15.0 * _cursor_cli_consecutive_rate_limits, 60.0)
        log(f"Cooldown {cooldown:.0f}s (rate-limit history)")
        time.sleep(cooldown)

    # Save state again for the improvement phase
    pre_patch = git_save_patch()
    pre_modified = git_diff_name_only()

    # Snapshot file contents so we can detect changes to pre-dirty files
    pre_contents: dict[str, str] = {}
    for f in pre_modified:
        full = REPO_ROOT / f
        if full.exists() and full.is_file():
            try:
                pre_contents[f] = full.read_text(encoding="utf-8")
            except Exception:
                pass

    review_prompt = build_rollout_review_prompt(
        task_instruction=task_instruction,
        rollout_summary=rollout_summary,
        rollout_path=str(rollout_path) if rollout_path else None,
        focus=focus,
        memory=memory,
        adaptive=adaptive,
        agent_result=agent_result,
    )
    review_result = cursor_cli(review_prompt, mode="agent", model=model)

    data["review"] = review_result[:2000]

    # ── Detect changes made by Cursor (the improvements) ──────────────────
    post_modified = git_diff_name_only()

    new_files = post_modified - pre_modified
    changed_predirty: set[str] = set()
    for f in pre_modified & post_modified:
        full = REPO_ROOT / f
        if full.exists() and full.is_file():
            try:
                current = full.read_text(encoding="utf-8")
            except Exception:
                continue
            if current != pre_contents.get(f, ""):
                changed_predirty.add(f)

    impl_files = new_files | changed_predirty

    data["files_changed"] = sorted(impl_files)
    log(f"Files changed by Cursor: {sorted(impl_files)}")

    if not impl_files:
        log("Rollout review produced no code changes", "WARN")
        data["outcome"] = "no_changes"
        update_memory(
            memory, iteration=iteration, focus=focus,
            task_title=task_title, score=None, outcome="crash",
            summary="Rollout review produced no file changes",
        )
        return data

    # Compute diff for logging
    agent_diff = _compute_agent_diff(pre_contents, impl_files)
    agent_diff_stat = _summarize_diff(agent_diff, impl_files)

    data["implementation_diff"] = agent_diff_stat
    log(f"Improvement diff:\n{agent_diff_stat}")

    # ── Step 5: Verification (with retry loop) ────────────────────────────
    banner(f"ITERATION {iteration} — STEP 5: Verification")

    max_fix_attempts = 2
    verification_passed = False

    for fix_attempt in range(max_fix_attempts + 1):
        agent_files = {f for f in impl_files if f.startswith("agent/")}

        # 5a: Syntax check
        syntax_ok, syntax_errors = verify_syntax(agent_files)
        data["syntax_ok"] = syntax_ok
        data["syntax_errors"] = syntax_errors

        if not syntax_ok:
            if fix_attempt < max_fix_attempts:
                log(f"SYNTAX ERRORS (attempt {fix_attempt + 1}/{max_fix_attempts + 1}) "
                    f"— asking Cursor to fix:\n  " + "\n  ".join(syntax_errors), "WARN")
                fix_prompt = (
                    f"Your previous implementation has syntax errors. Fix them:\n\n"
                    + "\n".join(f"- {e}" for e in syntax_errors) + "\n\n"
                    "Fix ONLY the syntax errors. Do not make other changes."
                )
                cursor_cli(fix_prompt, mode="agent", model=model)
                post_modified_fix = git_diff_name_only()
                impl_files = (post_modified_fix - pre_modified) | changed_predirty
                continue
            else:
                log(f"SYNTAX ERRORS — exhausted {max_fix_attempts} fix attempts, "
                    f"reverting:\n  " + "\n  ".join(syntax_errors), "WARN")
                git_revert_to_patch(pre_patch)
                data["outcome"] = "syntax_error"
                update_memory(
                    memory, iteration=iteration, focus=focus,
                    task_title=task_title, score=None, outcome="syntax_error",
                    files_changed=sorted(impl_files),
                    summary=f"Syntax errors after {max_fix_attempts} fix attempts: "
                            f"{'; '.join(syntax_errors[:3])}",
                )
                return data

        # 5b: Import/compile check
        import_ok, import_errors = verify_imports(agent_files)
        data["import_ok"] = import_ok
        data["import_errors"] = import_errors

        if not import_ok:
            if fix_attempt < max_fix_attempts:
                log(f"IMPORT ERRORS (attempt {fix_attempt + 1}/{max_fix_attempts + 1}) "
                    f"— asking Cursor to fix:\n  " + "\n  ".join(import_errors), "WARN")
                fix_prompt = (
                    f"Your previous implementation has compile/import errors. Fix them:\n\n"
                    + "\n".join(f"- {e}" for e in import_errors) + "\n\n"
                    "Fix ONLY the import/compile errors. Do not make other changes."
                )
                cursor_cli(fix_prompt, mode="agent", model=model)
                post_modified_fix = git_diff_name_only()
                impl_files = (post_modified_fix - pre_modified) | changed_predirty
                continue
            else:
                log(f"IMPORT ERRORS — exhausted {max_fix_attempts} fix attempts, "
                    f"reverting:\n  " + "\n  ".join(import_errors), "WARN")
                git_revert_to_patch(pre_patch)
                data["outcome"] = "import_error"
                update_memory(
                    memory, iteration=iteration, focus=focus,
                    task_title=task_title, score=None, outcome="import_error",
                    files_changed=sorted(impl_files),
                    summary=f"Import errors after {max_fix_attempts} fix attempts: "
                            f"{'; '.join(import_errors[:3])}",
                )
                return data

        verification_passed = True
        break

    if not verification_passed:
        git_revert_to_patch(pre_patch)
        data["outcome"] = "verification_failed"
        update_memory(
            memory, iteration=iteration, focus=focus,
            task_title=task_title, score=None, outcome="crash",
            summary="Verification loop exited without passing",
        )
        return data

    # Recompute diff after possible fix attempts
    agent_diff = _compute_agent_diff(pre_contents, impl_files)
    agent_diff_stat = _summarize_diff(agent_diff, impl_files)
    data["implementation_diff"] = agent_diff_stat

    log("Verification PASSED (syntax + imports)")

    # ── Accept + commit ───────────────────────────────────────────────────
    banner(f"ITERATION {iteration} — Accept")

    committed = False
    if impl_files and not no_commit:
        committed = git_commit_improvement(task_title, impl_files, iteration)

    data["outcome"] = "landed"
    data["committed"] = committed
    log(f"ACCEPTED (committed: {committed}, files: {sorted(impl_files)})")

    update_memory(
        memory, iteration=iteration, focus=focus,
        task_title=task_title, score=None, outcome="landed",
        files_changed=sorted(impl_files),
        summary=f"Rollout-driven improvement, committed={committed}",
    )

    data["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return data


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Rollout-driven agent self-improvement loop",
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
        "--model", default="opus-4.6-thinking",
        help="Cursor model for task generation and rollout review (default: opus-4.6-thinking)",
    )
    parser.add_argument(
        "--pause", type=int, default=10,
        help="Pause between iterations in seconds (default: 10)",
    )
    parser.add_argument(
        "--max-agent-iterations", type=int, default=20,
        help="Max agent loop iterations per task (default: 20)",
    )
    parser.add_argument(
        "--no-commit", action="store_true",
        help="Don't commit successful improvements (keep in working tree)",
    )
    args = parser.parse_args()

    banner("ROLLOUT-DRIVEN SELF-IMPROVEMENT — PREFLIGHT")

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
    log(f"Auto-commit:     {'yes' if not args.no_commit else 'no'}")

    # ── Focus scores summary ──────────────────────────────────────────────
    for area, stats in memory.get("focus_scores", {}).items():
        landed = stats.get("landed_count", 0)
        avg = stats.get("avg_score", 0.0)
        attempts = stats.get("attempts", 0)
        log(f"  {area}: {landed} landed / {attempts} attempts (avg score: {avg:.1f})")

    # ── Pre-flight health check ───────────────────────────────────────────
    banner("PRE-FLIGHT HEALTH CHECK")
    agent_healthy, preflight_error = preflight_check()
    if agent_healthy:
        log("Agent health check: PASSED")
    else:
        log(f"Agent health check: FAILED — {preflight_error}", "WARN")
        log("The agent cannot start. Self-improvement will focus on fixing this.")

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
                max_agent_iterations=args.max_agent_iterations,
                no_commit=args.no_commit,
                agent_healthy=agent_healthy,
                preflight_error=preflight_error,
            )
            results.append(result)
            save_iteration_log(iteration, result)

            # Re-check agent health after a landed improvement
            if result.get("outcome") == "landed":
                agent_healthy, preflight_error = preflight_check()
                if agent_healthy:
                    log("Agent health: PASSED (post-improvement)")

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
        committed = "✓" if r.get("committed") else "✗"
        log(
            f"  Iter {r.get('iteration', '?')}: [{outcome}] {title} "
            f"(committed: {committed})"
        )

    landed = sum(1 for r in results if r.get("outcome") == "landed")
    failed = sum(1 for r in results if r.get("outcome") not in ("landed",))
    log(f"\nSession: {landed} landed, {failed} failed out of {len(results)} total")
    log(f"All-time: {memory['iterations_total']} iterations, "
        f"{memory['improvements_landed']} landed")
    log(f"Memory: {MEMORY_FILE}")
    log(f"Logs: {LOG_DIR}")


if __name__ == "__main__":
    main()
