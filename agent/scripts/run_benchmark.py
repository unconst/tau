#!/usr/bin/env python3
"""Run a reproducible local benchmark suite against tau-agent."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TaskResult:
    name: str
    success: bool
    duration_s: float
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    llm_retries: int
    compactions: int
    parallel_batches: int
    completion_reason: str


def _load_tasks(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Benchmark file must contain a non-empty `tasks` array")
    return tasks


def _run_task(agent_dir: Path, task: dict[str, str], max_iterations: int) -> TaskResult:
    prompt = task.get("prompt", "").strip()
    name = task.get("name", prompt[:40] or "unnamed-task")
    if not prompt:
        raise ValueError(f"Task `{name}` is missing prompt")

    cmd = [
        sys.executable,
        "-m",
        "src.main",
        "exec",
        prompt,
        "--json",
        "--workdir",
        str(agent_dir.parent),
        "--max-iterations",
        str(max_iterations),
    ]
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(agent_dir),
        capture_output=True,
        text=True,
    )
    duration = time.perf_counter() - started

    usage = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}
    metrics: dict[str, Any] = {}
    success = False
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_type = event.get("type")
        if event_type == "turn.completed":
            usage = event.get("usage", usage)
            success = True
        elif event_type == "turn.failed":
            success = False
        elif event_type == "turn.metrics":
            metrics = event

    # Non-zero exit means failed even if we missed final events.
    if proc.returncode != 0:
        success = False

    return TaskResult(
        name=name,
        success=success,
        duration_s=duration,
        input_tokens=int(usage.get("input_tokens", 0) or 0),
        output_tokens=int(usage.get("output_tokens", 0) or 0),
        cached_tokens=int(usage.get("cached_input_tokens", 0) or 0),
        llm_retries=int(metrics.get("llm_retries", 0) or 0),
        compactions=int(metrics.get("compactions", 0) or 0),
        parallel_batches=int(metrics.get("parallel_batches", 0) or 0),
        completion_reason=str(metrics.get("completion_reason", "")),
    )


def _summarize(results: list[TaskResult]) -> dict[str, Any]:
    durations = [r.duration_s for r in results]
    completed = [r for r in results if r.success]
    return {
        "tasks": len(results),
        "success_rate": round((len(completed) / len(results)) * 100, 2),
        "mean_duration_s": round(statistics.mean(durations), 2),
        "p95_duration_s": round(sorted(durations)[max(0, int(len(durations) * 0.95) - 1)], 2),
        "total_input_tokens": sum(r.input_tokens for r in results),
        "total_output_tokens": sum(r.output_tokens for r in results),
        "total_cached_tokens": sum(r.cached_tokens for r in results),
        "total_llm_retries": sum(r.llm_retries for r in results),
        "total_compactions": sum(r.compactions for r in results),
        "total_parallel_batches": sum(r.parallel_batches for r in results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run tau-agent benchmark suite")
    parser.add_argument(
        "--suite",
        default="benchmarks/default_tasks.json",
        help="Path to benchmark suite JSON file (relative to agent/)",
    )
    parser.add_argument("--max-iterations", type=int, default=80, help="Per-task max iterations")
    parser.add_argument(
        "--output",
        default="benchmarks/last_run.json",
        help="Output JSON report path (relative to agent/)",
    )
    args = parser.parse_args()

    agent_dir = Path(__file__).resolve().parents[1]
    suite_path = (agent_dir / args.suite).resolve()
    output_path = (agent_dir / args.output).resolve()
    tasks = _load_tasks(suite_path)

    results: list[TaskResult] = []
    for index, task in enumerate(tasks, 1):
        print(f"[{index}/{len(tasks)}] {task.get('name', 'task')}", flush=True)
        results.append(_run_task(agent_dir, task, max_iterations=args.max_iterations))

    summary = _summarize(results)
    payload = {
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "suite_path": str(suite_path),
        "summary": summary,
        "results": [r.__dict__ for r in results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
