#!/usr/bin/env python3
"""
SuperAgent for Term Challenge - Entry Point (SDK 3.0 Compatible).

This agent accepts --instruction from the validator and runs autonomously.
Uses Chutes API for LLM calls instead of term_sdk.

Installation:
    pip install .                    # via pyproject.toml
    pip install -r requirements.txt  # via requirements.txt

Usage:
    python agent.py --instruction "Your task description here..."
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Venv bootstrap: re-exec inside .venv if we're running with system Python.
# This avoids PEP 668 "externally-managed-environment" failures and ensures
# all project dependencies are importable.
# ---------------------------------------------------------------------------
def _reexec_in_venv() -> None:
    """If running outside the project venv, re-exec with .venv/bin/python."""
    if sys.prefix != sys.base_prefix:
        return  # already in a venv

    agent_dir = Path(__file__).resolve().parent
    # Walk up to find project root (where .venv lives)
    for candidate in (agent_dir.parent, agent_dir):
        venv_python = candidate / ".venv" / "bin" / "python"
        if venv_python.is_file():
            print(
                f"[setup] Re-launching with venv Python: {venv_python}",
                file=sys.stderr,
                flush=True,
            )
            os.execv(str(venv_python), [str(venv_python)] + sys.argv)
            # execv replaces the process — this line is never reached


_reexec_in_venv()

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))


# Auto-install dependencies if missing
def ensure_dependencies():
    """Install dependencies if not present.

    Tries multiple installation strategies so the agent can bootstrap in any
    Python environment (uv-managed venvs, plain pip, system Python, etc.).
    Never raises on failure — if all installers fail we still attempt to
    proceed since the deps may already be importable via other means.
    """
    try:
        import httpx  # noqa: F401
        import pydantic  # noqa: F401
        return  # Already available, nothing to do
    except ImportError:
        pass

    agent_dir = Path(__file__).parent
    req_file = agent_dir / "requirements.txt"
    target = ["-r", str(req_file)] if req_file.exists() else [str(agent_dir)]

    # Ordered list of installation strategies to try
    strategies = [
        # 1. uv pip install (works in uv-managed environments / .venv)
        (["uv", "pip", "install"] + target + ["-q"], "uv pip"),
        # 2. Current interpreter's pip module (traditional venvs / system Python)
        ([sys.executable, "-m", "pip", "install"] + target + ["-q"], "pip"),
    ]

    for cmd, label in strategies:
        try:
            print(f"[setup] Trying {label} install...", file=sys.stderr)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                print(f"[setup] Dependencies installed via {label}", file=sys.stderr)
                return
            # Non-zero exit — log and try next strategy
            stderr_snippet = (result.stderr or "").strip()[:200]
            print(
                f"[setup] {label} failed (exit {result.returncode}): {stderr_snippet}",
                file=sys.stderr,
            )
        except FileNotFoundError:
            # Installer binary not found (e.g. uv not installed) — skip
            print(f"[setup] {label} not available, skipping", file=sys.stderr)
        except subprocess.TimeoutExpired:
            print(f"[setup] {label} timed out, skipping", file=sys.stderr)
        except Exception as e:
            print(f"[setup] {label} error: {e}", file=sys.stderr)

    # All strategies exhausted — warn but don't crash.  The subsequent
    # imports will raise a clear ImportError if deps are truly missing.
    print(
        "[setup] WARNING: Could not install dependencies via any method. "
        "Proceeding anyway — imports may fail.",
        file=sys.stderr,
    )


ensure_dependencies()

from src.config.defaults import CONFIG
from src.core.loop import run_agent_loop
from src.llm.client import CostLimitExceeded, LLMClient
from src.output.jsonl import ErrorEvent, emit
from src.tools.registry import ToolRegistry


class AgentContext:
    """Minimal context for agent execution (replaces term_sdk.AgentContext)."""

    def __init__(self, instruction: str, cwd: str = None):
        self.instruction = instruction
        self.cwd = cwd or os.getcwd()
        self.step = 0
        self.is_done = False
        self.history = []
        self._start_time = time.time()

    @property
    def elapsed_secs(self) -> float:
        return time.time() - self._start_time

    def shell(self, cmd: str, timeout: int = 120) -> "ShellResult":
        """Execute a shell command."""
        self.step += 1
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.cwd,
            )
            output = result.stdout + result.stderr
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            output = "[TIMEOUT]"
            exit_code = -1
        except Exception as e:
            output = f"[ERROR] {e}"
            exit_code = -1

        shell_result = ShellResult(output=output, exit_code=exit_code)
        self.history.append(
            {
                "step": self.step,
                "command": cmd,
                "output": output[:1000],
                "exit_code": exit_code,
            }
        )
        return shell_result

    def done(self):
        """Mark task as complete."""
        self.is_done = True

    def log(self, msg: str):
        """Log a message."""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [ctx] {msg}", file=sys.stderr, flush=True)


class ShellResult:
    """Result from shell command."""

    def __init__(self, output: str, exit_code: int):
        self.output = output
        self.stdout = output
        self.stderr = ""
        self.exit_code = exit_code

    def has(self, text: str) -> bool:
        return text in self.output


def _log(msg: str):
    """Log to stderr."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [superagent] {msg}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description="SuperAgent for Term Challenge SDK 3.0")
    parser.add_argument("--instruction", required=True, help="Task instruction from validator")
    parser.add_argument("--resume", help="Resume from a specific saved session ID")
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume from the latest saved session",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="Maximum agent loop iterations (0 = use config default)",
    )
    args = parser.parse_args()

    _log("=" * 60)
    _log("SuperAgent Starting (SDK 3.0 - Chutes API)")
    _log("=" * 60)
    _log(f"Model: {CONFIG['model']}")
    _log(f"Reasoning effort: {CONFIG.get('reasoning_effort', 'default')}")
    _log(f"Instruction: {args.instruction[:200]}...")
    _log("-" * 60)

    # Initialize components
    start_time = time.time()

    llm = LLMClient(
        model=CONFIG["model"],
        temperature=CONFIG.get("temperature"),
        max_tokens=CONFIG.get("max_tokens", 16384),
    )

    tools = ToolRegistry()
    ctx = AgentContext(instruction=args.instruction)

    _log("Components initialized")

    # Build config, allowing CLI overrides
    run_config = {
        **CONFIG,
        "resume_session_id": args.resume,
        "resume_latest": args.resume_latest,
    }
    if args.max_iterations > 0:
        run_config["max_iterations"] = args.max_iterations

    try:
        run_agent_loop(
            llm=llm,
            tools=tools,
            ctx=ctx,
            config=run_config,
        )
    except CostLimitExceeded as e:
        _log(f"Cost limit exceeded: {e}")
        emit(ErrorEvent(message=f"Cost limit exceeded: {e}"))
    except Exception as e:
        _log(f"Fatal error: {e}")
        emit(ErrorEvent(message=str(e)))
        raise
    finally:
        elapsed = time.time() - start_time
        try:
            stats = llm.get_stats()
            _log(f"Total tokens: {stats.get('total_tokens', 0)}")
            _log(f"Total cost: ${stats.get('total_cost', 0):.4f}")
            _log(f"Requests: {stats.get('request_count', 0)}")
        except Exception as e:
            _log(f"Stats error: {e}")
        _log(f"Elapsed: {elapsed:.1f}s")
        _log("Agent finished")
        _log("=" * 60)


if __name__ == "__main__":
    main()
