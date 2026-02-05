#!/usr/bin/env python3
"""Create a task for the agent to process later.

Usage:
    python -m tau.tools.create_task "Task Title" "Task description and context"
    python -m tau.tools.create_task --delay 2h "Check on X" "Follow up on..."
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

WORKSPACE = Path(__file__).parent.parent.parent
TASKS_DIR = WORKSPACE / "context" / "tasks"


def get_next_task_id() -> int:
    """Find the next available task ID."""
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    existing = [d.name for d in TASKS_DIR.iterdir() if d.is_dir() and d.name.startswith("task-")]
    if not existing:
        return 1
    ids = [int(d.split("-")[1]) for d in existing if d.split("-")[1].isdigit()]
    return max(ids) + 1 if ids else 1


def create_task(title: str, body: str, delay: str = None) -> str:
    """Create a new task. Returns the task ID."""
    task_id = get_next_task_id()
    task_dir = TASKS_DIR / f"task-{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    content = f"## {title}\n\n{body}\n\n---\nCreated: {timestamp}\n"
    
    if delay:
        content += f"Scheduled delay: {delay}\n"
    
    (task_dir / "task.md").write_text(content)
    (task_dir / "memory.md").write_text("# Task Memory\n\n")
    
    return f"task-{task_id}"


def main():
    parser = argparse.ArgumentParser(description="Create a task for the agent")
    parser.add_argument("title", help="Task title")
    parser.add_argument("body", nargs="?", default="", help="Task description")
    parser.add_argument("--delay", help="Schedule delay (e.g., 2h, 30m, 1d)")
    args = parser.parse_args()
    
    task_id = create_task(args.title, args.body, args.delay)
    print(f"Created {task_id}")


if __name__ == "__main__":
    main()
