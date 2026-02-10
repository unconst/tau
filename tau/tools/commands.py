#!/usr/bin/env python3
"""Execute tau bot commands programmatically.

This module allows the agent to call any telegram command directly,
enabling self-directed behavior like creating tasks for itself,
managing crons, adapting its own code, etc.

Usage:
    python -m tau.tools.commands task "Description of the task"
    python -m tau.tools.commands plan "Create a plan for X"
    python -m tau.tools.commands status
    python -m tau.tools.commands adapt "Add feature X"
    python -m tau.tools.commands cron "5min" "Check on training job"
    python -m tau.tools.commands crons
    python -m tau.tools.commands uncron 1
    python -m tau.tools.commands clear
    python -m tau.tools.commands debug
"""

import sys
import os
import json
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime

WORKSPACE = Path(__file__).parent.parent.parent
TASKS_DIR = WORKSPACE / "context" / "tasks"
PLANS_DIR = WORKSPACE / "context" / "plans"
CRON_FILE = WORKSPACE / "cron_jobs.json"


def get_next_task_id() -> int:
    """Find the next available task ID."""
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    existing = [d.name for d in TASKS_DIR.iterdir() if d.is_dir() and d.name.startswith("task-")]
    if not existing:
        return 1
    ids = [int(d.split("-")[1]) for d in existing if d.split("-")[1].isdigit()]
    return max(ids) + 1 if ids else 1


def cmd_task(description: str) -> str:
    """Add a task to the task queue.
    
    Args:
        description: The task description
        
    Returns:
        Confirmation message with task ID
    """
    if not description:
        return "Error: Task description is required"
    
    task_id = get_next_task_id()
    task_dir = TASKS_DIR / f"task-{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    content = f"## {description}\n- created: {timestamp}\n- source: agent\n"
    
    (task_dir / "task.md").write_text(content)
    (task_dir / "memory.md").write_text("# Task Memory\n\n<!-- Detailed memory for this task -->\n")
    
    return f"Task added (task-{task_id}): {description}"


def cmd_plan(description: str) -> str:
    """Create an execution plan for a task.
    
    Uses the agent to generate a comprehensive plan based on 
    the task description and workspace context.
    
    Args:
        description: What to create a plan for
        
    Returns:
        Plan content and filename
    """
    if not description:
        return "Error: Plan description is required"
    
    import re
    
    # Generate a filename-safe slug
    slug = re.sub(r'[^a-z0-9]+', '-', description.lower())[:50].strip('-')
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    plan_filename = f"plan-{timestamp}-{slug}.md"
    
    # Read context files to provide to the agent
    context_dir = WORKSPACE / "context"
    identity_content = ""
    memory_system_content = ""
    
    identity_file = context_dir / "IDENTITY.md"
    if identity_file.exists():
        identity_content = identity_file.read_text()
    
    memory_system_file = context_dir / "MEMORY-SYSTEM.md"
    if memory_system_file.exists():
        memory_system_content = memory_system_file.read_text()
    
    # Build context section
    context_section = ""
    if identity_content:
        context_section += f"\n## Agent Identity\n{identity_content}\n"
    if memory_system_content:
        context_section += f"\n## Memory System\n{memory_system_content}\n"
    
    # Build the planning prompt with workspace context
    plan_prompt = f"""You are Tau, creating an execution plan for a task.

WORKSPACE: {WORKSPACE}
{context_section}
---

TASK: {description}

Generate a comprehensive execution plan that includes:

1. **Goal**: Clear statement of what needs to be accomplished
2. **Current State**: What exists now (check relevant files/directories)
3. **Prerequisites**: What needs to be in place before starting
4. **Steps**: Numbered action items with specific shell commands or file operations
   - Each step should be atomic and verifiable
   - Include exact file paths relative to workspace
   - Include exact commands to run (with `source .venv/bin/activate` where needed)
5. **Success Criteria**: How to verify the task is complete
6. **Potential Issues**: Risks or blockers to watch for

Format as clean markdown. Be specific and actionable.
Use the workspace context above to inform your plan.

Output ONLY the plan content, no preamble."""

    from tau.codex import llm_chat
    
    try:
        plan_content = llm_chat(plan_prompt, timeout=600.0)
        
        if not plan_content:
            return "Error: Failed to generate plan content"
        
        # Save the plan
        PLANS_DIR.mkdir(parents=True, exist_ok=True)
        plan_path = PLANS_DIR / plan_filename
        
        full_plan = f"""# Plan: {description}

*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

---

{plan_content}
"""
        plan_path.write_text(full_plan)
        
        return f"Plan saved to {plan_filename}\n\n{plan_content[:500]}..."
        
    except Exception as e:
        return f"Error generating plan: {str(e)}"


def cmd_status() -> str:
    """Show recent activity and task status.
    
    Returns:
        Status information about tasks and memory
    """
    memory_file = TASKS_DIR / "memory.md"
    
    # Get high-level memory
    high_level = ""
    if memory_file.exists():
        content = memory_file.read_text()
        lines = content.strip().split("\n")
        high_level = "\n".join(lines[-20:])
    
    # Get task status
    tasks = []
    if TASKS_DIR.exists():
        for task_dir in sorted(TASKS_DIR.iterdir()):
            if task_dir.is_dir() and task_dir.name.startswith("task-"):
                task_file = task_dir / "task.md"
                if task_file.exists():
                    content = task_file.read_text()
                    # Extract title from first ## heading
                    for line in content.split("\n"):
                        if line.startswith("## "):
                            title = line[3:].strip()
                            # Check if complete
                            is_complete = "status: complete" in content.lower()
                            tasks.append({
                                "id": task_dir.name,
                                "title": title,
                                "complete": is_complete
                            })
                            break
    
    incomplete = [t for t in tasks if not t["complete"]]
    
    status_msg = "Status\n\n"
    
    if tasks:
        status_msg += f"Total tasks: {len(tasks)}\n"
        status_msg += f"Incomplete: {len(incomplete)}\n\n"
        
        if incomplete:
            status_msg += "Active tasks:\n"
            for task in incomplete[:5]:
                status_msg += f"  - {task['title'][:50]}\n"
    
    if high_level.strip():
        status_msg += f"\nRecent activity:\n\n{high_level[:2000]}"
    
    return status_msg if status_msg.strip() else "No tasks or memory yet."


def cmd_adapt(prompt: str) -> str:
    """Self-modify the bot using the agent.
    
    WARNING: This modifies the bot's own code and triggers a restart.
    Use carefully.
    
    Args:
        prompt: Description of what to change
        
    Returns:
        Result of the adaptation
    """
    if not prompt:
        return "Error: Adaptation prompt is required"
    
    from tau.codex import run_baseagent
    
    try:
        output = run_baseagent(prompt)
        
        # Commit changes
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, cwd=str(WORKSPACE)
            )
            if status.stdout.strip():
                subprocess.run(["git", "add", "-A"], cwd=str(WORKSPACE))
                subprocess.run(
                    ["git", "commit", "-m", f"[tau] adapt: {prompt[:100]}"],
                    cwd=str(WORKSPACE)
                )
        except Exception:
            pass
        
        return f"Adaptation complete: {output[:500]}"
        
    except Exception as e:
        return f"Error during adaptation: {str(e)}"


def _load_crons() -> tuple[list, int]:
    """Load cron jobs from disk."""
    if CRON_FILE.exists():
        try:
            data = json.loads(CRON_FILE.read_text())
            return data.get("jobs", []), data.get("next_id", 1)
        except Exception:
            pass
    return [], 1


def _save_crons(jobs: list, next_id: int):
    """Save cron jobs to disk."""
    CRON_FILE.write_text(json.dumps({"jobs": jobs, "next_id": next_id}, indent=2))


def _parse_interval(interval_str: str) -> int:
    """Parse interval string like '5min', '30sec', '2h' into seconds."""
    import re
    interval_str = interval_str.lower().strip()
    
    match = re.match(r'^(\d+)\s*([a-z]*)$', interval_str)
    if not match:
        try:
            return int(interval_str) * 60
        except ValueError:
            return 5 * 60
    
    number = int(match.group(1))
    unit = match.group(2)
    
    if unit in ('s', 'sec', 'second', 'seconds'):
        return number
    elif unit in ('m', 'min', 'minute', 'minutes'):
        return number * 60
    elif unit in ('h', 'hr', 'hour', 'hours'):
        return number * 3600
    else:
        return number * 60


def _get_chat_id() -> int | None:
    """Get stored chat ID from disk."""
    chat_id_file = WORKSPACE / "chat_id.txt"
    if chat_id_file.exists():
        try:
            return int(chat_id_file.read_text().strip())
        except (ValueError, IOError):
            return None
    return None


def cmd_cron(interval: str, prompt: str) -> str:
    """Schedule a recurring prompt.
    
    Args:
        interval: How often to run (e.g., '5min', '1h', '30sec')
        prompt: The prompt to run periodically
        
    Returns:
        Confirmation message with cron ID
    """
    if not interval or not prompt:
        return "Error: Both interval and prompt are required"
    
    # Get saved chat_id - required for cron jobs to run
    chat_id = _get_chat_id()
    if chat_id is None:
        return "Error: No chat_id found. Send /start to the bot first via Telegram."
    
    jobs, next_id = _load_crons()
    interval_seconds = _parse_interval(interval)
    
    job = {
        "id": next_id,
        "interval_seconds": interval_seconds,
        "prompt": prompt,
        "next_run": time.time() + interval_seconds,
        "chat_id": chat_id  # Use saved chat_id so cron jobs can send messages
    }
    
    jobs.append(job)
    _save_crons(jobs, next_id + 1)
    
    # Format interval for display
    if interval_seconds < 60:
        interval_display = f"{interval_seconds}sec"
    elif interval_seconds < 3600:
        interval_display = f"{interval_seconds // 60}min"
    else:
        interval_display = f"{interval_seconds // 3600}h"
    
    return f"Cron #{next_id} created: every {interval_display}\nPrompt: {prompt[:100]}"


def cmd_crons() -> str:
    """List all active cron jobs.
    
    Returns:
        List of active cron jobs
    """
    jobs, _ = _load_crons()
    
    if not jobs:
        return "No active cron jobs."
    
    lines = ["Active cron jobs:"]
    for job in jobs:
        interval = job["interval_seconds"]
        if interval < 60:
            interval_display = f"{interval}sec"
        elif interval < 3600:
            interval_display = f"{interval // 60}min"
        else:
            interval_display = f"{interval // 3600}h"
        lines.append(f"  #{job['id']}: every {interval_display} - {job['prompt'][:50]}...")
    
    return "\n".join(lines)


def cmd_uncron(cron_id: str) -> str:
    """Remove a cron job by ID.
    
    Args:
        cron_id: The cron job ID to remove
        
    Returns:
        Confirmation message
    """
    try:
        target_id = int(cron_id.replace("#", ""))
    except ValueError:
        return f"Error: Invalid cron ID '{cron_id}'"
    
    jobs, next_id = _load_crons()
    
    for i, job in enumerate(jobs):
        if job["id"] == target_id:
            jobs.pop(i)
            _save_crons(jobs, next_id)
            return f"Cron #{target_id} removed."
    
    return f"Cron #{target_id} not found. Use 'crons' to see active jobs."


def cmd_clear() -> str:
    """Stop any active agent processes.
    
    Returns:
        Number of processes stopped
    """
    # Import here to avoid circular imports
    try:
        sys.path.insert(0, str(WORKSPACE))
        from tau import processes
        stopped = processes.terminate_all(label_prefix="agent:", timeout_seconds=2.0)
        if stopped:
            return f"Stopped {len(stopped)} active agent process(es)."
        return "No active agent processes."
    except ImportError:
        return "Error: Could not import processes module"
    except Exception as e:
        return f"Error clearing processes: {str(e)}"


def cmd_debug() -> str:
    """Toggle debug mode.
    
    Note: This only works for the current session. The bot needs to
    be accessed via Telegram to persist debug mode across restarts.
    
    Returns:
        Current debug mode status
    """
    # Debug mode is managed by the main bot process, we can't toggle it from here
    # But we can report what it does
    return "Debug mode toggles verbose notifications. Use /debug in Telegram to toggle."


def cmd_restart() -> str:
    """Restart the bot process via supervisor or exec.
    
    WARNING: This will restart the bot process. Use carefully.
    
    Returns:
        Confirmation message (though process will restart before returning)
    """
    tauctl = WORKSPACE / "tauctl"
    
    # Try supervisor restart first
    if tauctl.exists():
        try:
            result = subprocess.run(
                [str(tauctl), "_agent_restart"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=str(WORKSPACE)
            )
            if result.returncode == 0:
                return "Restarting via supervisor..."
        except Exception:
            pass
    
    # Fallback: direct exec restart
    try:
        import sys
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        return f"Error restarting: {str(e)}"
    
    return "Restarting..."


def cmd_kill() -> str:
    """Fully stop the bot process.
    
    WARNING: This will stop the bot process. Use carefully.
    
    Returns:
        Confirmation message (though process will exit before returning)
    """
    # Stop any active agent processes first
    try:
        sys.path.insert(0, str(WORKSPACE))
        from tau import processes
        processes.terminate_all(label_prefix="agent:", timeout_seconds=2.0)
    except Exception:
        pass
    
    # Try supervisor stop
    tauctl = WORKSPACE / "tauctl"
    if tauctl.exists():
        try:
            result = subprocess.run(
                [str(tauctl), "_agent_stop"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=str(WORKSPACE)
            )
            if result.returncode == 0:
                os._exit(0)
        except Exception:
            pass
    
    # Fallback: hard exit
    os._exit(0)
    
    return "Stopping bot..."  # This won't be reached, but satisfies type checker


# Command registry for programmatic access
COMMANDS = {
    "task": {
        "func": cmd_task,
        "description": "Add a task to the queue",
        "usage": "commands task <description>",
        "args": ["description"],
    },
    "plan": {
        "func": cmd_plan,
        "description": "Create an execution plan",
        "usage": "commands plan <description>",
        "args": ["description"],
    },
    "status": {
        "func": cmd_status,
        "description": "See recent activity and task status",
        "usage": "commands status",
        "args": [],
    },
    "adapt": {
        "func": cmd_adapt,
        "description": "Self-modify the bot code",
        "usage": "commands adapt <prompt>",
        "args": ["prompt"],
    },
    "cron": {
        "func": cmd_cron,
        "description": "Schedule a recurring prompt",
        "usage": "commands cron <interval> <prompt>",
        "args": ["interval", "prompt"],
    },
    "crons": {
        "func": cmd_crons,
        "description": "List active cron jobs",
        "usage": "commands crons",
        "args": [],
    },
    "uncron": {
        "func": cmd_uncron,
        "description": "Remove a cron job",
        "usage": "commands uncron <id>",
        "args": ["cron_id"],
    },
    "clear": {
        "func": cmd_clear,
        "description": "Stop active agent processes",
        "usage": "commands clear",
        "args": [],
    },
    "debug": {
        "func": cmd_debug,
        "description": "Toggle debug mode",
        "usage": "commands debug",
        "args": [],
    },
    "restart": {
        "func": cmd_restart,
        "description": "Restart the bot process",
        "usage": "commands restart",
        "args": [],
    },
    "kill": {
        "func": cmd_kill,
        "description": "Stop the bot process",
        "usage": "commands kill",
        "args": [],
    },
}


def run_command(command: str, *args) -> str:
    """Execute a command by name.
    
    Args:
        command: The command name (task, plan, status, etc.)
        *args: Arguments for the command
        
    Returns:
        Command result
    """
    if command not in COMMANDS:
        available = ", ".join(COMMANDS.keys())
        return f"Unknown command: {command}\nAvailable commands: {available}"
    
    cmd_info = COMMANDS[command]
    func = cmd_info["func"]
    expected_args = cmd_info["args"]
    
    if len(args) < len(expected_args):
        return f"Error: {command} requires {len(expected_args)} argument(s): {', '.join(expected_args)}"
    
    try:
        return func(*args[:len(expected_args)]) if expected_args else func()
    except Exception as e:
        return f"Error executing {command}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Execute tau bot commands programmatically",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  task <description>         Add a task to the queue
  plan <description>         Create an execution plan  
  status                     See recent activity
  adapt <prompt>             Self-modify the bot
  cron <interval> <prompt>   Schedule recurring prompt
  crons                      List active crons
  uncron <id>                Remove a cron
  clear                      Stop active agent processes
  restart                    Restart the bot process
  kill                       Stop the bot process
  debug                      Toggle debug mode

Examples:
  python -m tau.tools.commands task "Research GPU providers"
  python -m tau.tools.commands cron 1h "Check training status"
  python -m tau.tools.commands status
"""
    )
    parser.add_argument("command", help="Command to execute")
    parser.add_argument("args", nargs="*", help="Command arguments")
    
    args = parser.parse_args()
    
    result = run_command(args.command, *args.args)
    print(result)


if __name__ == "__main__":
    main()
