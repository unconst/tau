#!/usr/bin/env python3
"""
Benchmark suite for the agent self-improvement loop.

Defines deterministic tasks with setup, instruction, and verification.
Used to ensure improvements don't regress agent capability.

Can be run standalone:
    python3 agent/scripts/benchmark_suite.py

Or imported by self_improve.py.
"""

from __future__ import annotations

import ast
import json
import shutil
import time
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = REPO_ROOT / "agent"
SANDBOX_BASE = AGENT_DIR / ".agent" / "benchmark_sandbox"


# ---------------------------------------------------------------------------
# Sandbox management
# ---------------------------------------------------------------------------
def _setup_sandbox(name: str) -> Path:
    """Create a clean sandbox directory for a benchmark task."""
    sandbox = SANDBOX_BASE / name
    if sandbox.exists():
        shutil.rmtree(sandbox)
    sandbox.mkdir(parents=True)
    return sandbox


def _cleanup_sandbox(name: str):
    """Remove a sandbox directory."""
    sandbox = SANDBOX_BASE / name
    if sandbox.exists():
        shutil.rmtree(sandbox)


# ---------------------------------------------------------------------------
# Setup functions (prepare fixtures in sandbox)
# ---------------------------------------------------------------------------
def _setup_noop(sandbox: Path):
    pass


def _setup_find_replace(sandbox: Path):
    (sandbox / "data.txt").write_text(
        "The quick brown fox jumps over the lazy dog.\n"
        "The fox is clever and fast.\n"
        "Everyone admires the fox.\n"
    )


def _setup_fix_syntax(sandbox: Path):
    (sandbox / "broken.py").write_text(
        "def greet(name):\n"
        '    msg = f"Hello, {name}!"\n'
        "    print(msg\n"  # Missing closing paren
        "    return msg\n"
    )


def _setup_read_count(sandbox: Path):
    lines = [str(i) for i in range(1, 26)]
    (sandbox / "numbers.txt").write_text("\n".join(lines) + "\n")


def _setup_grep_collect(sandbox: Path):
    src = sandbox / "src"
    src.mkdir()
    (src / "main.py").write_text(
        "# TODO: implement login\n"
        "def login():\n"
        "    pass\n"
        "\n"
        "# TODO: implement logout\n"
        "def logout():\n"
        "    pass\n"
    )
    (src / "utils.py").write_text(
        "# TODO: add validation\n"
        "def validate():\n"
        "    pass\n"
    )
    (src / "config.py").write_text(
        "DEBUG = True\n"
        "PORT = 8080\n"
    )


# ---------------------------------------------------------------------------
# Verify functions (check results after agent runs)
# ---------------------------------------------------------------------------
def _verify_create_file(sandbox: Path) -> bool:
    f = sandbox / "hello.txt"
    if not f.exists():
        return False
    return f.read_text().strip() == "Hello, World!"


def _verify_find_replace(sandbox: Path) -> bool:
    f = sandbox / "data.txt"
    if not f.exists():
        return False
    content = f.read_text()
    return "cat" in content and "fox" not in content


def _verify_fix_syntax(sandbox: Path) -> bool:
    f = sandbox / "broken.py"
    if not f.exists():
        return False
    try:
        ast.parse(f.read_text())
        return True
    except SyntaxError:
        return False


def _verify_multi_step(sandbox: Path) -> bool:
    output_dir = sandbox / "output"
    if not output_dir.is_dir():
        return False
    data_file = output_dir / "data.json"
    readme_file = output_dir / "README.md"
    if not data_file.exists() or not readme_file.exists():
        return False
    try:
        data = json.loads(data_file.read_text())
        return data.get("status") == "ok"
    except (json.JSONDecodeError, KeyError):
        return False


def _verify_read_count(sandbox: Path) -> bool:
    f = sandbox / "count.txt"
    if not f.exists():
        return False
    return "25" in f.read_text().strip()


def _verify_grep_collect(sandbox: Path) -> bool:
    f = sandbox / "todos.txt"
    if not f.exists():
        return False
    content = f.read_text().lower()
    return "login" in content and "logout" in content and "validation" in content


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------
BENCHMARKS = [
    {
        "name": "create_file",
        "description": "Create a file with specific content",
        "setup": _setup_noop,
        "instruction": (
            "Create a file at {sandbox}/hello.txt containing exactly the text "
            "'Hello, World!' (without quotes, no trailing newline). "
            "Do not create any other files."
        ),
        "verify": _verify_create_file,
        "timeout": 120,
        "max_iterations": 5,
    },
    {
        "name": "find_replace",
        "description": "Find and replace text in a file",
        "setup": _setup_find_replace,
        "instruction": (
            "In the file {sandbox}/data.txt, replace ALL occurrences of the "
            "word 'fox' with 'cat'. Do not change anything else in the file."
        ),
        "verify": _verify_find_replace,
        "timeout": 120,
        "max_iterations": 5,
    },
    {
        "name": "fix_syntax",
        "description": "Fix a Python syntax error",
        "setup": _setup_fix_syntax,
        "instruction": (
            "The file {sandbox}/broken.py has a syntax error. Find and fix "
            "the syntax error so the file is valid Python. Only fix the "
            "syntax; do not change any logic."
        ),
        "verify": _verify_fix_syntax,
        "timeout": 120,
        "max_iterations": 5,
    },
    {
        "name": "multi_step",
        "description": "Create directory structure with multiple files",
        "setup": _setup_noop,
        "instruction": (
            'Create a directory at {sandbox}/output/. Inside it, create two files: '
            '(1) data.json containing exactly {{"status": "ok", "count": 42}}, and '
            '(2) README.md with a brief description of the data file.'
        ),
        "verify": _verify_multi_step,
        "timeout": 150,
        "max_iterations": 8,
    },
    {
        "name": "read_and_count",
        "description": "Read a file and count its lines",
        "setup": _setup_read_count,
        "instruction": (
            "Read the file {sandbox}/numbers.txt, count the total number of "
            "lines, and write ONLY the count as a plain number to "
            "{sandbox}/count.txt. Nothing else in the file."
        ),
        "verify": _verify_read_count,
        "timeout": 120,
        "max_iterations": 5,
    },
    {
        "name": "grep_and_collect",
        "description": "Search files and collect TODO comments",
        "setup": _setup_grep_collect,
        "instruction": (
            "Search all Python files under {sandbox}/src/ for lines containing "
            "'TODO'. Write each matching line to {sandbox}/todos.txt "
            "(one per line, preserving the original text)."
        ),
        "verify": _verify_grep_collect,
        "timeout": 120,
        "max_iterations": 8,
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_single_benchmark(
    benchmark: dict,
    run_agent_fn: Callable,
) -> dict:
    """Run a single benchmark task. Returns result dict."""
    name = benchmark["name"]
    sandbox = _setup_sandbox(name)

    try:
        # Setup fixtures
        benchmark["setup"](sandbox)

        # Format instruction with sandbox path
        instruction = benchmark["instruction"].format(sandbox=str(sandbox))

        # Run the agent
        started = time.time()
        agent_result = run_agent_fn(
            instruction=instruction,
            timeout=benchmark.get("timeout", 120),
            max_iterations=benchmark.get("max_iterations", 5),
        )
        elapsed = time.time() - started

        # Verify result
        passed = benchmark["verify"](sandbox)

        return {
            "name": name,
            "description": benchmark["description"],
            "passed": passed,
            "elapsed": round(elapsed, 1),
            "timed_out": agent_result.get("timed_out", False),
            "exit_code": agent_result.get("exit_code", -1),
            "turns": agent_result.get("turns", 0),
        }
    except Exception as e:
        return {
            "name": name,
            "description": benchmark["description"],
            "passed": False,
            "elapsed": 0,
            "error": str(e),
        }
    finally:
        _cleanup_sandbox(name)


def run_all_benchmarks(
    run_agent_fn: Callable,
    task_names: list[str] | None = None,
) -> dict:
    """Run all (or selected) benchmarks. Returns aggregate results."""
    tasks = BENCHMARKS
    if task_names:
        tasks = [b for b in BENCHMARKS if b["name"] in task_names]

    results = {}
    passed_count = 0
    total = len(tasks)

    for benchmark in tasks:
        print(f"\n{'─' * 50}", flush=True)
        print(f"  Benchmark: {benchmark['name']} — {benchmark['description']}", flush=True)
        print(f"{'─' * 50}", flush=True)

        result = run_single_benchmark(benchmark, run_agent_fn)
        results[benchmark["name"]] = result

        if result.get("passed"):
            passed_count += 1

        status = "PASS" if result.get("passed") else "FAIL"
        elapsed = result.get("elapsed", 0)
        print(f"  Result: {status} ({elapsed:.1f}s)\n", flush=True)

    return {
        "results": results,
        "passed": passed_count,
        "total": total,
        "score": f"{passed_count}/{total}",
    }
