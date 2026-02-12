#!/usr/bin/env python3
"""
Self-improvement loop for the agent.

Iteratively discovers weaknesses, implements improvements via Cursor CLI,
verifies them (syntax, imports, benchmarks), and accumulates successful
changes via git commits.

Features:
- Institutional memory tracking all past iterations and outcomes
- Benchmark suite for measuring agent quality before/after changes
- Curriculum-based progression (startup → basic → correct → efficient → capable)
- Rollout-driven discovery (benchmark errors feed directly into proposals)
- Pre-flight health check (detects agent startup failures before wasting cycles)
- Quality gates: syntax, import, benchmark regression, score >= 6/10
- Strict critic: runs in agent mode with full file access & efficiency rubric
- Implementation retry loop (2 fix attempts before revert)
- Benchmark trend tracking with regression detection
- Adaptive focus selection (curriculum-gated, weakest areas get priority)
- Git accumulation (successful improvements are committed)
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
        "benchmark_trend": [],  # list of {iteration, passed, total, timestamp}
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


def record_benchmark_trend(memory: dict, iteration: int, benchmark: dict | None):
    """Record a benchmark score in the rolling trend tracker."""
    if not benchmark:
        return
    trend = memory.setdefault("benchmark_trend", [])
    trend.append({
        "iteration": iteration,
        "passed": benchmark.get("passed", 0),
        "total": benchmark.get("total", 0),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    # Keep last 50 data points
    if len(trend) > 50:
        memory["benchmark_trend"] = trend[-50:]
    save_memory(memory)


def check_benchmark_trend(memory: dict) -> tuple[str, bool]:
    """Analyze the benchmark trend and detect if performance is declining.

    Returns (description, should_halt).
    should_halt is True if there's a clear declining trend over the last N points.
    """
    trend = memory.get("benchmark_trend", [])
    if len(trend) < 3:
        return "Not enough data points for trend analysis.", False

    recent = trend[-5:]  # last 5 benchmark runs
    scores = [t["passed"] for t in recent]

    # Check for monotonic decline
    declining = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    # But only halt if it's a real decline (not flat at 0)
    actually_declining = declining and scores[0] > scores[-1]

    # Check for overall regression from peak
    all_scores = [t["passed"] for t in trend]
    peak = max(all_scores) if all_scores else 0
    current = scores[-1] if scores else 0
    regressed_from_peak = peak > 0 and current < peak * 0.5  # dropped below 50% of peak

    description = (
        f"Trend ({len(trend)} points): "
        f"recent={scores}, peak={peak}, current={current}"
    )

    if actually_declining and len(recent) >= 3:
        return (
            f"{description} — DECLINING over last {len(recent)} runs. "
            "Consider reverting recent changes.",
            True,
        )
    if regressed_from_peak:
        return (
            f"{description} — REGRESSED from peak ({peak}) to {current}. "
            "Recent changes may have degraded the agent.",
            True,
        )

    return f"{description} — stable or improving.", False


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
# CURRICULUM & ADAPTIVE FOCUS SELECTION
# ===========================================================================

# Progression tiers: each tier gates which focus areas are eligible.
# The agent must graduate through tiers based on benchmark performance.
#
# Tier 0 (STARTUP):  Agent can't even start → fix imports/startup errors
# Tier 1 (BASIC):    Agent starts but fails tasks → basic reliability
# Tier 2 (CORRECT):  Agent passes some tasks → correctness & robustness
# Tier 3 (EFFICIENT): Agent passes most tasks → optimize for speed/tokens
# Tier 4 (CAPABLE):  Agent is efficient → enhance capabilities

CURRICULUM_TIERS = {
    0: {
        "name": "startup",
        "description": "Agent cannot start — fix import/startup errors",
        "min_benchmark_pass_rate": -1,  # always eligible if agent can't start
        "focus_areas": [
            "error handling and recovery",
            "tool implementation",
        ],
    },
    1: {
        "name": "basic",
        "description": "Agent starts but fails most tasks — basic reliability",
        "min_benchmark_pass_rate": 0.0,
        "focus_areas": [
            "error handling and recovery",
            "tool implementation",
            "LLM interaction",
        ],
    },
    2: {
        "name": "correct",
        "description": "Agent passes some tasks — improve correctness",
        "min_benchmark_pass_rate": 0.33,
        "focus_areas": [
            "error handling and recovery",
            "search and navigation",
            "planning and task decomposition",
            "tool implementation",
        ],
    },
    3: {
        "name": "efficient",
        "description": "Agent passes most tasks — optimize efficiency",
        "min_benchmark_pass_rate": 0.66,
        "focus_areas": [
            "context management",
            "LLM interaction",
            "planning and task decomposition",
            "search and navigation",
        ],
    },
    4: {
        "name": "capable",
        "description": "Agent is efficient — enhance capabilities",
        "min_benchmark_pass_rate": 0.85,
        "focus_areas": FOCUS_AREAS,  # all areas eligible
    },
}


def get_curriculum_tier(
    benchmark_scores: dict | None,
    agent_healthy: bool = True,
) -> int:
    """Determine the current curriculum tier based on benchmark performance."""
    if not agent_healthy:
        return 0

    if not benchmark_scores or "results" not in benchmark_scores:
        return 1  # no benchmark data, assume basic tier

    total = benchmark_scores.get("total", 0)
    passed = benchmark_scores.get("passed", 0)
    if total == 0:
        return 1

    pass_rate = passed / total

    # Find the highest tier we qualify for
    current_tier = 1
    for tier_num in sorted(CURRICULUM_TIERS.keys()):
        if tier_num == 0:
            continue  # tier 0 is special (startup failure)
        min_rate = CURRICULUM_TIERS[tier_num]["min_benchmark_pass_rate"]
        if pass_rate >= min_rate:
            current_tier = tier_num

    return current_tier


def select_focus(
    memory: dict,
    benchmark_scores: dict | None = None,
    agent_healthy: bool = True,
) -> str:
    """Select the focus area most in need of improvement.

    Uses the curriculum tier to gate which focus areas are eligible,
    then prioritizes areas with lower scores and fewer landed improvements.
    """
    tier = get_curriculum_tier(benchmark_scores, agent_healthy)
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
    }


# ===========================================================================
# ROLLOUT ANALYSIS
# ===========================================================================

def analyze_rollout(agent_result: dict) -> dict:
    """Extract structured performance metrics from an agent run result.

    Parses the JSONL output and stderr to identify:
    - Token efficiency (input/output/cached)
    - Turn efficiency (completed vs failed)
    - Error patterns (recurring errors, crash types)
    - Stuck-loop indicators (high turns with no progress)
    """
    usage = agent_result.get("usage", {})
    turns = agent_result.get("turns", 0)
    turns_failed = agent_result.get("turns_failed", 0)
    errors = agent_result.get("errors", [])
    stderr = agent_result.get("stderr", "") or ""
    elapsed = agent_result.get("elapsed", 0)
    exit_code = agent_result.get("exit_code", -1)
    timed_out = agent_result.get("timed_out", False)

    # Classify errors
    error_categories: dict[str, int] = {}
    for err in errors:
        err_lower = err.lower()
        if "none" in err_lower and "iterable" in err_lower:
            cat = "null_response"
        elif "rate" in err_lower or "429" in err_lower:
            cat = "rate_limit"
        elif "timeout" in err_lower:
            cat = "timeout"
        elif "module" in err_lower and "not found" in err_lower:
            cat = "import_error"
        elif "connection" in err_lower or "network" in err_lower:
            cat = "network_error"
        else:
            cat = "other"
        error_categories[cat] = error_categories.get(cat, 0) + 1

    # Check stderr for import/startup errors
    startup_error = ""
    if exit_code != 0 and stderr:
        for line in stderr.strip().splitlines()[-5:]:
            if "ModuleNotFoundError" in line or "ImportError" in line:
                startup_error = line.strip()
                break
            if "SyntaxError" in line:
                startup_error = line.strip()
                break

    # Compute efficiency metrics
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cached_tokens = usage.get("cached_input_tokens", 0)
    total_tokens = input_tokens + output_tokens
    cache_hit_rate = (
        round(cached_tokens / input_tokens, 2) if input_tokens > 0 else 0.0
    )

    return {
        "turns_completed": turns,
        "turns_failed": turns_failed,
        "total_turns": turns + turns_failed,
        "elapsed_seconds": round(elapsed, 1),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "total_tokens": total_tokens,
        "cache_hit_rate": cache_hit_rate,
        "error_categories": error_categories,
        "error_count": len(errors),
        "startup_error": startup_error,
        "stuck_loop": turns > 5 and turns_failed == 0 and exit_code != 0,
    }


def format_benchmark_diagnostics(benchmark_scores: dict | None) -> str:
    """Format benchmark results WITH error details for inclusion in prompts.

    Unlike the old bench_text which just showed PASS/FAIL, this includes
    stderr, error messages, and performance metrics so the discovery prompt
    can target REAL problems.
    """
    if not benchmark_scores or "results" not in benchmark_scores:
        return "Not yet computed."

    lines = []
    for name, result in benchmark_scores["results"].items():
        status = "PASS" if result.get("passed") else "FAIL"
        elapsed = result.get("elapsed", 0)
        turns = result.get("turns", 0)
        turns_failed = result.get("turns_failed", 0)
        exit_code = result.get("exit_code", -1)
        errors = result.get("errors", [])
        stderr_tail = result.get("stderr_tail", "")

        line = f"- **{name}**: {status} ({elapsed:.1f}s, {turns} turns, exit={exit_code})"
        if not result.get("passed"):
            # Include WHY it failed
            if stderr_tail:
                # Extract the most informative error line
                for sline in stderr_tail.strip().splitlines():
                    if any(kw in sline for kw in (
                        "Error", "error", "Traceback", "ModuleNotFound",
                        "ImportError", "SyntaxError", "Exception",
                    )):
                        line += f"\n  ERROR: {sline.strip()[:200]}"
                        break
            if errors:
                line += f"\n  ERRORS: {'; '.join(errors[:3])}"
            if turns == 0 and turns_failed == 0:
                line += "\n  NOTE: Agent produced zero turns (likely crashed on startup)"
        lines.append(line)

    total = benchmark_scores.get("total", 0)
    passed = benchmark_scores.get("passed", 0)
    lines.append(f"\nOverall: {passed}/{total}")

    # Add aggregate diagnosis
    all_same_error = True
    first_error = None
    for result in benchmark_scores["results"].values():
        stderr = result.get("stderr_tail", "")
        if stderr and not result.get("passed"):
            for sline in stderr.strip().splitlines():
                if "Error" in sline:
                    if first_error is None:
                        first_error = sline.strip()
                    elif sline.strip() != first_error:
                        all_same_error = False
                    break

    if first_error and all_same_error and passed == 0:
        lines.append(
            f"\n** ALL benchmarks fail with the same error: {first_error[:200]}\n"
            f"** FIX THIS FIRST — it blocks all other improvements."
        )

    return "\n".join(lines)


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
    preflight_error: str = "",
) -> str:
    """Build a context-rich discovery prompt with memory, blacklist, benchmarks.

    Now includes full benchmark diagnostics (errors, stderr, metrics) so the
    discovery targets REAL observable failures rather than hypothetical ones.
    """
    history_text = format_history_for_prompt(memory)

    blacklist = memory.get("blacklist", [])
    blacklist_text = "\n".join(f"- {b}" for b in blacklist) if blacklist else "None yet."

    # Benchmark diagnostics with error details
    bench_text = format_benchmark_diagnostics(benchmark_scores)

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

    # Pre-flight error override: if the agent can't even start, that's THE problem
    preflight_section = ""
    if preflight_error:
        preflight_section = (
            f"\n## CRITICAL: Agent Startup Failure\n"
            f"The agent CANNOT START due to: `{preflight_error}`\n"
            f"**This MUST be fixed before any other improvement can take effect.**\n"
            f"Override your focus area and propose a fix for this startup error.\n"
        )

    prompt = (
        f"You are reviewing a repository containing an autonomous coding agent "
        f"(at `agent/`). Your focus area is: **{focus}** — {desc}.\n\n"
        f"{preflight_section}"
        f"## Previous Iterations (what has already been tried)\n"
        f"{history_text}\n\n"
        f"## Blacklisted Proposals (do NOT propose these again)\n"
        f"{blacklist_text}\n\n"
        f"## Current Benchmark Results (with error details)\n"
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
        "7. Addresses REAL problems visible in the benchmark errors above "
        "(not hypothetical issues)\n"
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
    changed_files: list[str] | None = None,
    benchmark_before: dict | None = None,
    benchmark_after: dict | None = None,
) -> str:
    """Build critique prompt to evaluate the implementation diff.

    Improvements over original:
    - Shows up to 12000 chars of diff (was 4000) for better context
    - Lists changed files so the critic can read them in full (agent mode)
    - Includes efficiency/performance criteria in the rubric
    - Includes before/after benchmark data when available
    """
    max_diff_chars = 12000
    diff_display = diff_content[-max_diff_chars:] if len(diff_content) > max_diff_chars else diff_content
    if len(diff_content) > max_diff_chars:
        diff_note = f"(showing last {max_diff_chars} of {len(diff_content)} chars)"
    else:
        diff_note = "(complete diff)"

    files_section = ""
    if changed_files:
        files_list = "\n".join(f"- `{f}`" for f in sorted(changed_files))
        files_section = (
            f"\n## Changed Files\n{files_list}\n"
            "You can read these files to see the full context of the changes.\n"
        )

    bench_section = ""
    if benchmark_before or benchmark_after:
        bench_section = "\n## Benchmark Impact\n"
        if benchmark_before:
            bench_section += f"- Before: {benchmark_before.get('score', '?')}\n"
        if benchmark_after:
            bench_section += f"- After: {benchmark_after.get('score', '?')}\n"
        if benchmark_before and benchmark_after:
            before_p = benchmark_before.get("passed", 0)
            after_p = benchmark_after.get("passed", 0)
            if after_p > before_p:
                bench_section += f"- Delta: +{after_p - before_p} benchmarks passing\n"
            elif after_p < before_p:
                bench_section += f"- Delta: -{before_p - after_p} REGRESSION\n"
            else:
                bench_section += "- Delta: no change\n"

    return (
        "You are an expert code reviewer evaluating a change to an autonomous "
        "coding agent. Your goal is to ensure only HIGH-QUALITY improvements "
        "land. Be strict — mediocre changes degrade the codebase over time.\n\n"
        f"## Task\n"
        f"**Title**: {title}\n"
        f"**Instruction**: {instruction[:1200]}\n"
        f"{files_section}"
        f"{bench_section}\n"
        f"## Changes Made\n```\n{diff_stat}\n```\n\n"
        f"## Diff {diff_note}\n```\n{diff_display}\n```\n\n"
        "## Evaluation Criteria (score each 0-10, then give overall)\n\n"
        "1. **Correctness** (weight: 3x) — Does the change implement what was "
        "requested? Are there bugs, edge cases, or logical errors?\n"
        "2. **Quality** (weight: 2x) — Is the code clean, safe, idiomatic, and "
        "well-integrated with the existing codebase?\n"
        "3. **Impact** (weight: 3x) — Will this measurably improve the agent? "
        "Does it address a real bottleneck? Would it help pass more benchmarks, "
        "reduce token usage, or complete tasks in fewer turns?\n"
        "4. **Efficiency** (weight: 1x) — Does the change make the agent faster "
        "or more token-efficient? Or does it add unnecessary overhead?\n"
        "5. **Risk** (weight: 1x) — Could this break existing functionality? "
        "How defensive is the implementation?\n\n"
        "A score of 6/10 or higher means the change is GOOD ENOUGH to land.\n"
        "A score below 6/10 means it should be REVERTED.\n"
        "Be honest — a 5/10 change that barely works should not land.\n\n"
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
    agent_healthy: bool = True,
    preflight_error: str = "",
) -> dict:
    """Run one full improvement iteration.

    Flow:
        1. Select focus (adaptive + curriculum tier) + build discovery prompt
        2. Discovery via Cursor CLI → TASK_TITLE + TASK_INSTRUCTION
        3. Implementation via Cursor CLI (agent mode)
        4. Verification: syntax check + import/compile check (with retries)
        5. Benchmark: run suite, compare to baseline (optional)
        6. Critique: evaluate the diff, extract score (agent mode, strict)
        7. Decision: accept (commit) or revert based on quality gates (6/10)
        8. Update memory + record benchmark trend
    """
    data: dict = {
        "iteration": iteration,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # ── Adaptive parameters ───────────────────────────────────────────────
    adaptive = get_adaptive_params(memory)
    if adaptive.get("increase_timeout"):
        agent_timeout = max(agent_timeout, 2400)

    # ── Focus selection (curriculum-aware) ─────────────────────────────────
    focus = select_focus(
        memory,
        benchmark_scores=baseline_benchmark,
        agent_healthy=agent_healthy,
    )
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
        preflight_error=preflight_error,
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

    # Snapshot file contents so we can detect changes to pre-dirty files
    pre_contents: dict[str, str] = {}
    for f in pre_modified:
        full = REPO_ROOT / f
        if full.exists() and full.is_file():
            try:
                pre_contents[f] = full.read_text(encoding="utf-8")
            except Exception:
                pass

    impl_prompt = build_impl_prompt(title, instruction, adaptive)
    impl_result = cursor_cli(impl_prompt, mode="agent", model=model)

    data["implementation"] = impl_result[:2000]

    # ── Detect changes using content comparison (not just file names) ─────
    post_modified = git_diff_name_only()

    # Truly new files (not in pre_modified at all)
    new_files = post_modified - pre_modified

    # Pre-dirty files whose content actually changed
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
    log(f"Files changed by agent: {sorted(impl_files)}")
    if changed_predirty:
        log(f"  (includes pre-dirty files with new edits: {sorted(changed_predirty)})")

    if not impl_files:
        log("Implementation produced no changes at all", "WARN")
        data["outcome"] = "no_changes"
        update_memory(
            memory, iteration=iteration, focus=focus,
            task_title=title, score=0, outcome="crash",
            summary="Implementation produced no file changes",
        )
        return data

    # Compute agent-specific diff (only the agent's changes, not pre-existing)
    agent_diff = _compute_agent_diff(pre_contents, impl_files)
    agent_diff_stat = _summarize_diff(agent_diff, impl_files)

    data["implementation_diff"] = agent_diff_stat
    log(f"Agent diff stat:\n{agent_diff_stat}")

    # ── Step 3: Verification (with retry loop) ──────────────────────────────
    banner(f"ITERATION {iteration} — STEP 3: Verification")

    max_fix_attempts = 2
    verification_passed = False

    for fix_attempt in range(max_fix_attempts + 1):
        # Filter to only agent source files for verification
        agent_files = {f for f in impl_files if f.startswith("agent/")}

        # 3a: Syntax check
        syntax_ok, syntax_errors = verify_syntax(agent_files)
        data["syntax_ok"] = syntax_ok
        data["syntax_errors"] = syntax_errors

        if not syntax_ok:
            if fix_attempt < max_fix_attempts:
                log(f"SYNTAX ERRORS (attempt {fix_attempt + 1}/{max_fix_attempts + 1}) "
                    f"— asking implementer to fix:\n  " + "\n  ".join(syntax_errors), "WARN")
                fix_prompt = (
                    f"Your previous implementation has syntax errors. Fix them:\n\n"
                    + "\n".join(f"- {e}" for e in syntax_errors) + "\n\n"
                    "Fix ONLY the syntax errors. Do not make other changes."
                )
                cursor_cli(fix_prompt, mode="agent", model=model)
                # Re-detect changed files after fix
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
                    task_title=title, score=0, outcome="syntax_error",
                    files_changed=sorted(impl_files),
                    summary=f"Syntax errors after {max_fix_attempts} fix attempts: "
                            f"{'; '.join(syntax_errors[:3])}",
                )
                return data

        # 3b: Import/compile check
        import_ok, import_errors = verify_imports(agent_files)
        data["import_ok"] = import_ok
        data["import_errors"] = import_errors

        if not import_ok:
            if fix_attempt < max_fix_attempts:
                log(f"IMPORT ERRORS (attempt {fix_attempt + 1}/{max_fix_attempts + 1}) "
                    f"— asking implementer to fix:\n  " + "\n  ".join(import_errors), "WARN")
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
                    task_title=title, score=0, outcome="import_error",
                    files_changed=sorted(impl_files),
                    summary=f"Import errors after {max_fix_attempts} fix attempts: "
                            f"{'; '.join(import_errors[:3])}",
                )
                return data

        # Both checks passed
        verification_passed = True
        break

    if not verification_passed:
        # Should not reach here, but defensive
        git_revert_to_patch(pre_patch)
        data["outcome"] = "verification_failed"
        update_memory(
            memory, iteration=iteration, focus=focus,
            task_title=title, score=0, outcome="crash",
            summary="Verification loop exited without passing",
        )
        return data

    # Recompute diff after possible fix attempts
    agent_diff = _compute_agent_diff(pre_contents, impl_files)
    agent_diff_stat = _summarize_diff(agent_diff, impl_files)
    data["implementation_diff"] = agent_diff_stat

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

    critique_prompt = build_critique_prompt(
        title, instruction, agent_diff_stat, agent_diff,
        changed_files=sorted(impl_files),
        benchmark_before=baseline_benchmark,
        benchmark_after=benchmark_result,
    )
    # Run critic in agent mode so it can read the changed files in full context
    critique_result = cursor_cli(critique_prompt, mode="agent", model=model)

    data["critique"] = critique_result
    score = extract_score(critique_result)
    data["score"] = score
    log(f"Critique score: {score}/10" if score is not None else "Critique score: N/A")

    # ── Step 6: Decision ──────────────────────────────────────────────────
    banner(f"ITERATION {iteration} — STEP 6: Decision")

    # Quality gate: require score >= 6 (or accept if score couldn't be parsed)
    # A 6/10 bar ensures only genuinely useful changes land.
    min_score = 6
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

    # ── Pre-flight health check ───────────────────────────────────────────
    banner("PRE-FLIGHT HEALTH CHECK")
    agent_healthy, preflight_error = preflight_check()
    if agent_healthy:
        log("Agent health check: PASSED")
    else:
        log(f"Agent health check: FAILED — {preflight_error}", "WARN")
        log("The agent cannot start. Self-improvement will focus on fixing this.")

    # ── Benchmark trend check ─────────────────────────────────────────────
    trend_desc, trend_declining = check_benchmark_trend(memory)
    log(f"Benchmark trend: {trend_desc}")
    if trend_declining:
        log("WARNING: Benchmark performance is declining. "
            "Will continue but watch closely.", "WARN")

    # ── Baseline benchmarks ───────────────────────────────────────────────
    baseline_benchmark = None
    if not args.skip_benchmarks and not args.skip_impl:
        banner("BASELINE BENCHMARK")
        baseline_benchmark = run_benchmarks_safely()
        if baseline_benchmark:
            log(f"Baseline score: {baseline_benchmark['score']}")
            memory["last_benchmark"] = baseline_benchmark
            # Record in trend
            record_benchmark_trend(
                memory,
                iteration=memory.get("iterations_total", 0),
                benchmark=baseline_benchmark,
            )
            save_memory(memory)

            # Log curriculum tier
            tier = get_curriculum_tier(baseline_benchmark, agent_healthy)
            tier_info = CURRICULUM_TIERS.get(tier, CURRICULUM_TIERS[1])
            log(f"Curriculum tier: {tier} ({tier_info['name']}) — "
                f"{tier_info['description']}")
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
                agent_healthy=agent_healthy,
                preflight_error=preflight_error,
            )
            results.append(result)
            save_iteration_log(iteration, result)

            # Update baseline if improvement landed and we have new benchmark data
            if result.get("outcome") == "landed" and result.get("benchmark"):
                baseline_benchmark = result["benchmark"]
                record_benchmark_trend(memory, iteration, baseline_benchmark)
                log("Updated baseline benchmark after successful improvement")

                # Re-check agent health after a landed improvement
                agent_healthy, preflight_error = preflight_check()
                if agent_healthy:
                    log("Agent health: PASSED (post-improvement)")

            # Check benchmark trend periodically
            if i % 3 == 0:
                trend_desc, trend_declining = check_benchmark_trend(memory)
                log(f"Trend check: {trend_desc}")
                if trend_declining:
                    log("ALERT: Benchmark trend is declining. "
                        "Consider reviewing recent commits.", "WARN")

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
