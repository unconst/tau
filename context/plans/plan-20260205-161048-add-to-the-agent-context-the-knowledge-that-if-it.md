# Plan: Add to the agent context the knowledge that if it needs to send itself a message it should do so by creating a task for itself explain how it can do this and send it a message for later in the context. Does that make sense, we are going to tell the agent how it can send itself a message in the future. Give it permission and tools to to create for itself cron jobs and tasks.

plan out how we can improve this sytem without expanding the context too much and making it versatile for doing long range tasks, working in the backgounr on things and checking back on tasks etc. We want the agent to understnad how to send itself messages that get acted on later

*Generated: 2026-02-05 16:11:55*

---

Now I have a comprehensive understanding of the current system. Let me create the detailed execution plan.

# Execution Plan: Agent Self-Messaging and Task Scheduling System

## Goal

Enable Tau to autonomously send messages to itself for future processing, create scheduled tasks, and manage long-running background work. The agent should understand how to:
1. Create tasks that will be picked up by its own loop
2. Schedule one-time and recurring reminders via `at` and `cron`
3. Manage long-range work by breaking it into scheduled checkpoints
4. Maintain minimal context overhead while being maximally useful

## Prerequisites

1. **Existing infrastructure verified:**
   - `context/skills/self-scheduling.md` exists with basic documentation
   - `tau/tools/send_message.py` exists for sending Telegram messages
   - Agent loop in `tau/agent.py` processes tasks from `context/tasks/task-{id}/`
   - `at` and `cron` are available on the system

2. **Current gaps identified:**
   - `self-scheduling.md` is not referenced in `CONTEXT.md` task-specific loading
   - No dedicated tool for creating tasks programmatically (only manual shell commands)
   - No examples of background/long-range task patterns
   - Agent prompt doesn't mention self-scheduling capabilities

## Steps

### Step 1: Create a Task Creation Tool

Create `tau/tools/create_task.py` - a programmatic way for the agent to create tasks for itself:

```python
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
```

### Step 2: Create a Scheduled Message Tool

Create `tau/tools/schedule_message.py` - for scheduling future messages:

```python
#!/usr/bin/env python3
"""Schedule a message or task for later.

Usage:
    python -m tau.tools.schedule_message --at "14:00" "Meeting reminder"
    python -m tau.tools.schedule_message --in "2h" "Check on training job"
    python -m tau.tools.schedule_message --cron "0 9 * * *" "Daily standup"
"""

import subprocess
import argparse
import os

WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def schedule_at(time_spec: str, message: str):
    """Schedule a one-time message using 'at'."""
    cmd = f'cd {WORKSPACE} && source .venv/bin/activate && python -m tau.tools.send_message "{message}"'
    proc = subprocess.Popen(
        ["at", time_spec],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    _, stderr = proc.communicate(cmd)
    print(f"Scheduled for {time_spec}: {stderr.strip()}")

def schedule_cron(cron_spec: str, message: str):
    """Add a recurring cron job."""
    cmd = f'cd {WORKSPACE} && source .venv/bin/activate && python -m tau.tools.send_message "{message}"'
    current = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    existing = current.stdout if current.returncode == 0 else ""
    new_cron = f"{cron_spec} {cmd}\n"
    subprocess.run(["crontab", "-"], input=existing + new_cron, text=True)
    print(f"Added cron job: {cron_spec}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("message", help="Message to send")
    parser.add_argument("--at", dest="at_time", help="Time for 'at' (e.g., '14:00', 'now + 2 hours')")
    parser.add_argument("--in", dest="in_time", help="Relative time (e.g., '2h', '30m')")
    parser.add_argument("--cron", help="Cron schedule (e.g., '0 9 * * *')")
    args = parser.parse_args()
    
    if args.at_time:
        schedule_at(args.at_time, args.message)
    elif args.in_time:
        schedule_at(f"now + {args.in_time.replace('h', ' hours').replace('m', ' minutes')}", args.message)
    elif args.cron:
        schedule_cron(args.cron, args.message)

if __name__ == "__main__":
    main()
```

### Step 3: Update Agent Prompt with Self-Scheduling Tools

In `tau/agent.py`, add to the `PROMPT_TEMPLATE` TOOLS section:

```python
- create_task: source .venv/bin/activate && python -m tau.tools.create_task "Title" "Description"
  (Create a task for yourself to process later)
- schedule_message: source .venv/bin/activate && python -m tau.tools.schedule_message --in "2h" "message"
  (Schedule a future message: --at "14:00", --in "2h", --cron "0 9 * * *")
```

### Step 4: Enhance self-scheduling.md with Patterns

Add a "Long-Range Task Patterns" section to `context/skills/self-scheduling.md`:

```markdown
## Long-Range Task Patterns

### Pattern 1: Background Job with Checkpoints
When starting a long-running operation:
1. Create initial task with full context
2. Schedule checkpoint reminders at intervals
3. Each checkpoint creates a follow-up task if work continues

Example: Training a model
```bash
# Create the monitoring task
python -m tau.tools.create_task "Monitor training job X" "Instance: abc123. Started at $(date). Check progress every 2 hours."

# Schedule periodic check-ins
python -m tau.tools.schedule_message --in "2h" "Checkpoint: Check training job X progress"
```

### Pattern 2: Deferred Work
When you can't complete something now:
```bash
python -m tau.tools.create_task "Complete Y when Z is ready" "Context: [full context]. Waiting for: [dependency]"
```

### Pattern 3: Daily/Weekly Reviews
```bash
# Weekly project review
python -m tau.tools.schedule_message --cron "0 18 * * 5" "Weekly review: Summarize progress and plan next week"
```

### Self-Messaging Decision Tree
- **Immediate action needed?** → Execute now
- **Need to wait for external event?** → Create task with trigger condition
- **Time-based follow-up?** → Use `at` or cron
- **Complex multi-step work?** → Create task, schedule checkpoints
```

### Step 5: Add Self-Scheduling to Context Loader

Update `context/CONTEXT.md` to ensure the skill loads when relevant:

```markdown
## Skills Reference
Load only if relevant to the request:
- `context/skills/self-scheduling.md` - Creating tasks, reminders, and cron jobs for yourself
```

### Step 6: Update .cursor/rules/context-system.mdc

Ensure the keyword triggers include task-related terms:

```markdown
- "remind", "reminder", "schedule", "cron", "later", "task", "self-message", "follow-up", "checkpoint", "defer" → load `context/skills/self-scheduling.md`
```

### Step 7: Add Permission Statement to IDENTITY.md

Add explicit permission for self-scheduling to `context/IDENTITY.md`:

```markdown
## Capabilities

- **Self-Scheduling**: You are authorized to create tasks for yourself, schedule reminders, and set up cron jobs. You can send messages to yourself in the future to handle deferred work, checkpoints, and follow-ups.
```

### Step 8: Create Minimal Reference Card

Add a quick-reference section at the TOP of `self-scheduling.md` (before detailed docs):

```markdown
## Quick Reference

| Action | Command |
|--------|---------|
| Create task | `python -m tau.tools.create_task "Title" "Body"` |
| Message in 2h | `python -m tau.tools.schedule_message --in "2h" "msg"` |
| Message at time | `python -m tau.tools.schedule_message --at "14:00" "msg"` |
| Daily at 9am | `python -m tau.tools.schedule_message --cron "0 9 * * *" "msg"` |
| View scheduled | `atq` (at jobs), `crontab -l` (cron) |
```

## Success Criteria

1. **Tools exist and work:**
   - `python -m tau.tools.create_task "Test" "Body"` creates a task directory
   - `python -m tau.tools.schedule_message --in "1m" "test"` schedules successfully
   - Agent can invoke these from within its execution loop

2. **Context is loaded correctly:**
   - When user mentions "remind", "schedule", "later", etc., the agent loads `self-scheduling.md`
   - Agent prompt includes self-scheduling tools

3. **Agent understands capabilities:**
   - Given "remind me to check X in 2 hours", agent uses `schedule_message`
   - Given "start monitoring Y and check back periodically", agent creates task + schedules checkpoints
   - Agent can explain its self-scheduling capabilities when asked

4. **Context stays lean:**
   - `self-scheduling.md` remains under 200 lines
   - Quick reference at top provides immediate utility
   - Detailed patterns below for complex scenarios

## Potential Issues

1. **`at` daemon not running:**
   - Check: `systemctl status atd`
   - Fix: `systemctl start atd && systemctl enable atd`

2. **Cron jobs accumulating:**
   - Risk: Agent creates many cron jobs that persist
   - Mitigation: Add cleanup guidance; consider auto-expiring cron entries

3. **Task ID collisions:**
   - Risk: Concurrent task creation could create duplicate IDs
   - Mitigation: Use file locking or UUIDs instead of sequential IDs

4. **Context bloat from verbose skill docs:**
   - Risk: Loading full `self-scheduling.md` uses too many tokens
   - Mitigation: Keep quick reference minimal; details only when deep-dived

5. **Orphaned scheduled jobs:**
   - Risk: Scheduled messages for completed tasks still fire
   - Mitigation: Include task ID in scheduled messages; agent can dismiss irrelevant reminders

6. **Permission/path issues in cron:**
   - Risk: Cron environment differs from interactive shell
   - Mitigation: Always use absolute paths; always `cd` to workspace first
