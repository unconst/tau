"""Minimal two-buffer agent loop with task-specific memory."""

import os
import re
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

from .telegram import think, notify, get_chat_history

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS_DIR = os.path.join(WORKSPACE, "context", "tasks")
MEMORY_FILE = os.path.join(WORKSPACE, "context", "tasks", "memory.md")
IDENTITY_FILE = os.path.join(WORKSPACE, "context", "IDENTITY.md")
MEMORY_SYSTEM_FILE = os.path.join(WORKSPACE, "context", "MEMORY-SYSTEM.md")
STORY_FILE = os.path.join(WORKSPACE, "context", "STORY.md")
CHAT_HISTORY_FILE = os.path.join(WORKSPACE, "context", "CHAT.md")
LOG_FILE = os.path.join(WORKSPACE, "logs", "tau.log")

# Track last story update for detecting new activity
_last_story_chat_hash = None
_last_story_log_hash = None

# Ensure tasks directory exists
os.makedirs(TASKS_DIR, exist_ok=True)

PROMPT_TEMPLATE = """You are Tau, a single-threaded autonomous agent.

{identity}

{memory_rules}

---

NARRATIVE (your story so far):
{narrative}

---

TELEGRAM CHAT (recent):
{chat_history}

INCOMPLETE TASKS:
{tasks}

HIGH-LEVEL MEMORY (recent):
{high_level_memory}

CURRENT TASK MEMORY (recent):
{task_memory}

---

Execution Loop:
1. Identify incomplete tasks (skip completed ones entirely)
2. Select the single most important next action
3. Perform exactly one action
4. Record detailed factual output in task memory (context/tasks/task-{{id}}/memory.md)
5. Mark task complete explicitly if finished ("Task complete")

Memory Rules:
- Write facts only, never invent results
- Do not repeat prior actions
- Every action must advance a task
- Task-specific files go in context/tasks/task-{{id}}/
- Shared/system files go at project root

Invariant:
If no task can be safely advanced, ask the user instead of guessing.

TOOLS:
- send_message: source .venv/bin/activate && python -m tau.tools.send_message "message"
- send_voice: source .venv/bin/activate && python -m tau.tools.send_voice "message"
"""

STORY_PROMPT_TEMPLATE = """You are summarizing Tau's recent activity for STORY.md.

{identity}

---

CURRENT STORY (context/STORY.md):
{current_story}

---

RECENT CHAT ACTIVITY (last 5 minutes):
{recent_chat}

---

RECENT AGENT LOGS (last 5 minutes):
{recent_logs}

---

Write a brief summary of what happened in the last 5 minutes. Be concise and factual.
- What did the user ask for?
- What did Tau do?
- What was the outcome?

Use as few sentences as needed. 1-3 sentences is fine if that covers it.
Write in third person. No flowery language or storytelling - just a clear summary.

Output ONLY the summary, nothing else."""


def read_file(path: str) -> str:
    """Read file contents, return empty string if missing."""
    if os.path.exists(path):
        return open(path).read()
    return ""


def tail(content: str, n: int = 50) -> str:
    """Get last N lines of content."""
    lines = content.strip().split("\n")
    return "\n".join(lines[-n:])


def get_task_directories() -> list[Path]:
    """Get all task directories, sorted by creation time."""
    tasks_path = Path(TASKS_DIR)
    if not tasks_path.exists():
        return []
    
    task_dirs = [d for d in tasks_path.iterdir() if d.is_dir() and d.name.startswith("task-")]
    # Sort by directory name (which includes task ID)
    return sorted(task_dirs, key=lambda x: x.name)


def parse_task_from_dir(task_dir: Path) -> dict:
    """Parse a task from its directory. Returns task dict with id, title, body, and path."""
    task_file = task_dir / "task.md"
    if not task_file.exists():
        return None
    
    content = task_file.read_text()
    lines = content.split("\n")
    
    # First ## heading is the title
    title = ""
    body_lines = []
    found_title = False
    
    for line in lines:
        if line.startswith("## ") and not found_title:
            title = line[3:].strip()
            found_title = True
        elif found_title:
            body_lines.append(line)
    
    task_id = task_dir.name  # e.g., "task-1"
    
    return {
        "id": task_id,
        "title": title,
        "body": "\n".join(body_lines).strip(),
        "dir": task_dir,
        "memory_file": task_dir / "memory.md"
    }


def get_all_tasks() -> list[dict]:
    """Get all tasks from task directories."""
    task_dirs = get_task_directories()
    tasks = []
    
    for task_dir in task_dirs:
        task = parse_task_from_dir(task_dir)
        if task:
            tasks.append(task)
    
    return tasks


def parse_tasks(content: str) -> list[dict]:
    """Legacy: Parse tasks from markdown. Each ## heading is a task.
    This is kept for backward compatibility during migration."""
    tasks = []
    current = None
    
    for line in content.split("\n"):
        if line.startswith("## "):
            if current:
                tasks.append(current)
            current = {"title": line[3:].strip(), "body": ""}
        elif current:
            current["body"] += line + "\n"
    
    if current:
        tasks.append(current)
    
    return tasks


def is_task_complete(task: dict) -> bool:
    """Check if task memory indicates task is complete.
    Checks both memory.md (if it exists) and task.md (which persists after cleanup)."""
    # First check task.md for completion marker (persists after cleanup)
    task_file = task["dir"] / "task.md"
    if task_file.exists():
        task_content = task_file.read_text().lower()
        # Check for completion marker in task.md
        if '- status: complete' in task_content or '- complete' in task_content:
            return True
    
    # Also check memory.md if it exists (before cleanup)
    if "memory_file" in task:
        memory_file = task["memory_file"]
        if memory_file.exists():
            memory_content = memory_file.read_text().lower()
            title = task["title"].lower()
            
            # Look for explicit completion markers
            patterns = [
                f'task "{title}" is complete',
                f'task "{title}" complete',
                f'completed: {title}',
                f'{title} is complete',
                f'finished {title}',
                'task is complete',
                'task complete',
                'completed',
            ]
            
            for pattern in patterns:
                if pattern in memory_content:
                    return True
    
    return False


def find_incomplete(tasks: list[dict]) -> list[dict]:
    """Return tasks not marked complete."""
    return [t for t in tasks if not is_task_complete(t)]


def summarize(text: str, chars: int = 50) -> str:
    """Truncate text to N chars."""
    text = text.strip().replace("\n", " ")
    if len(text) > chars:
        return text[:chars-3] + "..."
    return text


def create_high_level_summary(detailed_content: str, task_title: str) -> str:
    """Create a high-level summary from detailed content.
    Focus on what was accomplished, not technical details."""
    content = detailed_content.strip()
    
    # Try to extract first sentence
    sentences = content.split(".")
    if len(sentences) > 1:
        first_sentence = sentences[0].strip()
        if len(first_sentence) > 20 and len(first_sentence) < 200:
            return first_sentence + "."
    
    # If no good sentence, create summary from first part
    # Remove technical details like file paths, function names in quotes
    summary = content.replace("\n", " ").strip()
    
    # Limit length
    if len(summary) > 200:
        # Try to cut at a word boundary
        cut_point = summary[:200].rfind(" ")
        if cut_point > 100:
            summary = summary[:cut_point] + "..."
        else:
            summary = summary[:197] + "..."
    
    return summary


def get_recent_content(filepath: str, minutes: int = 5) -> str:
    """Get content from the last N minutes based on timestamps in the file."""
    if not os.path.exists(filepath):
        return ""
    
    content = read_file(filepath)
    if not content:
        return ""
    
    # For log files, filter by timestamp
    if filepath.endswith('.log'):
        cutoff = datetime.now() - timedelta(minutes=minutes)
        recent_lines = []
        for line in content.split('\n'):
            # Log format: 2024-01-15 10:30:45 [INFO] ...
            try:
                if len(line) >= 19:
                    ts_str = line[:19]
                    ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                    if ts >= cutoff:
                        recent_lines.append(line)
            except (ValueError, IndexError):
                # Include lines that don't have timestamps if we're in a recent section
                if recent_lines:
                    recent_lines.append(line)
        return '\n'.join(recent_lines[-100:])  # Limit to last 100 lines
    
    # For CHAT.md, look for ## headers with timestamps
    if 'CHAT' in filepath.upper():
        cutoff = datetime.now() - timedelta(minutes=minutes)
        lines = content.split('\n')
        recent_lines = []
        include_section = False
        
        for line in lines:
            # Chat format: ## USER - 2024-01-15 10:30:45
            if line.startswith('## '):
                try:
                    # Extract timestamp from header
                    parts = line.split(' - ')
                    if len(parts) >= 2:
                        ts_str = parts[-1].strip()
                        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                        include_section = ts >= cutoff
                except (ValueError, IndexError):
                    include_section = False
            
            if include_section:
                recent_lines.append(line)
        
        return '\n'.join(recent_lines[-100:])
    
    # Fallback: return last 50 lines
    return tail(content, 50)


def get_content_hash(content: str) -> str:
    """Get a simple hash of content for change detection."""
    import hashlib
    return hashlib.md5(content.encode()).hexdigest()


def should_update_story() -> bool:
    """Check if there's new activity since last story update."""
    global _last_story_chat_hash, _last_story_log_hash
    
    recent_chat = get_recent_content(CHAT_HISTORY_FILE, minutes=5)
    recent_logs = get_recent_content(LOG_FILE, minutes=5)
    
    chat_hash = get_content_hash(recent_chat) if recent_chat else ""
    log_hash = get_content_hash(recent_logs) if recent_logs else ""
    
    # If no content at all, don't update
    if not recent_chat and not recent_logs:
        return False
    
    # Check if anything changed
    changed = (chat_hash != _last_story_chat_hash) or (log_hash != _last_story_log_hash)
    
    # Update stored hashes
    _last_story_chat_hash = chat_hash
    _last_story_log_hash = log_hash
    
    return changed


def append_story(narrative: str):
    """Append a narrative entry to STORY.md."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Ensure file exists with header
    if not os.path.exists(STORY_FILE):
        with open(STORY_FILE, "w") as f:
            f.write("# Tau's Story\n\n")
            f.write("<!-- Summarized event log, updated every 5 minutes -->\n\n")
    
    with open(STORY_FILE, "a") as f:
        f.write(f"### {timestamp}\n\n")
        f.write(f"{narrative.strip()}\n\n")
        f.write("---\n\n")


def run_story_update():
    """Run the story agent to update STORY.md with recent activity."""
    global _last_story_chat_hash, _last_story_log_hash
    
    # Check if there's new activity
    if not should_update_story():
        return
    
    # Get recent content
    recent_chat = get_recent_content(CHAT_HISTORY_FILE, minutes=5)
    recent_logs = get_recent_content(LOG_FILE, minutes=5)
    
    # Get current story for context
    current_story = ""
    if os.path.exists(STORY_FILE):
        current_story = tail(read_file(STORY_FILE), 30)
    
    # Get identity for context
    identity_content = read_file(IDENTITY_FILE)
    
    # Build prompt
    prompt = STORY_PROMPT_TEMPLATE.format(
        identity=identity_content.strip() if identity_content else "",
        current_story=current_story if current_story else "No story yet.",
        recent_chat=recent_chat if recent_chat else "No recent chat.",
        recent_logs=recent_logs if recent_logs else "No recent logs."
    )
    
    # Run agent to generate narrative
    try:
        result = subprocess.run(
            [
                "agent",
                "--force",
                "--model", "gemini-3-flash",
                "--mode=ask",
                "--output-format=text",
                "--print",
                prompt
            ],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=60,
            cwd=WORKSPACE
        )
        
        narrative = result.stdout.strip() if result.stdout else ""
        
        if narrative and len(narrative) > 20:
            append_story(narrative)
            
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        pass  # Silently handle story errors


def run_story_loop(stop_event=None):
    """Story update loop. Runs every 5 minutes."""
    # Wait 30 seconds before first run to let things settle
    time.sleep(30)
    
    STORY_INTERVAL = 300  # 5 minutes
    
    while True:
        if stop_event and stop_event.is_set():
            break
        
        try:
            run_story_update()
        except Exception:
            pass  # Silently handle story loop errors
        
        # Wait for next interval
        for _ in range(STORY_INTERVAL):
            if stop_event and stop_event.is_set():
                break
            time.sleep(1)


def run_cursor(prompt: str) -> str:
    """Run Cursor agent with prompt, return output."""
    try:
        result = subprocess.run(
            [
                "agent",
                "--force",
                "--model", "composer-1",
                "--output-format=text",
                "--print",
                prompt
            ],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=WORKSPACE
        )
        return result.stdout.strip() if result.stdout else result.stderr.strip()
    except subprocess.TimeoutExpired:
        return "Error: Agent timed out after 5 minutes"
    except Exception as e:
        return f"Error: {str(e)}"


def git_commit_changes(description: str):
    """Commit any changes made by the agent with a description based on the task.
    This allows reverting to previous states if needed."""
    try:
        # Check if there are any changes to commit
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=WORKSPACE
        )
        
        if not status_result.stdout.strip():
            # No changes to commit
            return
        
        # Stage all changes
        subprocess.run(
            ["git", "add", "-A"],
            capture_output=True,
            text=True,
            cwd=WORKSPACE
        )
        
        # Create commit message from description
        # Truncate if too long, clean up for commit message
        commit_msg = description.replace("\n", " ").strip()
        if len(commit_msg) > 200:
            commit_msg = commit_msg[:197] + "..."
        
        # Prefix with [tau] to identify agent commits
        commit_msg = f"[tau] {commit_msg}"
        
        # Commit
        subprocess.run(
            ["git", "commit", "-m", commit_msg],
            capture_output=True,
            text=True,
            cwd=WORKSPACE
        )
        
    except Exception:
        pass  # Silently handle git errors


def mark_task_complete_in_taskmd(task: dict):
    """Mark task as complete in task.md by adding completion marker."""
    task_file = task["dir"] / "task.md"
    if not task_file.exists():
        return
    
    task_content = task_file.read_text()
    # Check if already marked complete
    if '- status: complete' in task_content.lower() or '- complete' in task_content.lower():
        return
    
    # Add completion marker
    with open(task_file, "a") as f:
        f.write("\n- status: complete\n")


def append_task_memory(task: dict, detailed_content: str):
    """Append detailed timestamped entry to task's memory.md.
    If the content indicates task completion, also mark task.md as complete."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"\n### {timestamp}\n{detailed_content}\n"
    
    memory_file = task["memory_file"]
    # Ensure parent directory exists
    memory_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(memory_file, "a") as f:
        f.write(entry)
    
    # Check if this entry indicates completion and mark task.md accordingly
    content_lower = detailed_content.lower()
    # Look for explicit completion statements
    completion_patterns = [
        'task complete',
        'task is complete', 
        'completed',
        'finished',
    ]
    
    # Only mark complete if there's an explicit completion statement
    # This avoids false positives from verification messages
    if any(pattern in content_lower for pattern in completion_patterns):
        mark_task_complete_in_taskmd(task)


def append_high_level_memory(high_level_summary: str):
    """Append high-level summary to context/tasks/memory.md."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"\n### {timestamp}\n{high_level_summary}\n"
    
    # Ensure memory file exists with header
    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "w") as f:
            f.write("# Memory\n\n<!-- High-level summaries only. Detailed memory is in context/tasks/*/memory.md -->\n")
    
    with open(MEMORY_FILE, "a") as f:
        f.write(entry)


def cleanup_task_context(task: dict):
    """Clean up unneeded context files for a completed task.
    Keeps only files required for scripts to continue running (scripts, configs, dependencies).
    Removes memory.md and other documentation/output files.
    Marks task as complete in task.md so completion status persists after cleanup."""
    task_dir = task["dir"]
    if not task_dir.exists():
        return
    
    # Ensure task is marked complete in task.md (persists after cleanup)
    mark_task_complete_in_taskmd(task)
    
    # Files/extensions to keep (runtime files needed for scripts)
    keep_extensions = {
        '.py',  # Python scripts
        '.sh',  # Shell scripts
        '.js', '.ts', '.jsx', '.tsx',  # JavaScript/TypeScript
        '.json',  # Config files
        '.yaml', '.yml',  # Config files
        '.toml',  # Config files (pyproject.toml, etc.)
        '.txt',  # Requirements.txt, etc.
        '.env',  # Environment files
        '.sql',  # Database scripts
        '.csv', '.tsv',  # Data files
        '.db', '.sqlite', '.sqlite3',  # Database files
        '.pem', '.key', '.crt',  # Certificates/keys
    }
    
    # Files to always keep (by name)
    keep_files = {
        'task.md',  # Task description
        'requirements.txt',
        'pyproject.toml',
        'package.json',
        '.env',
        '.env.local',
    }
    
    # Files to always remove
    remove_files = {
        'memory.md',  # Detailed memory
    }
    
    removed_count = 0
    
    # Iterate through all files in task directory
    for item in task_dir.iterdir():
        if item.is_file():
            # Check if it's a file we should always remove
            if item.name in remove_files:
                try:
                    item.unlink()
                    removed_count += 1
                    think(f"removed {item.name} from {task['id']}")
                except Exception as e:
                    think(f"error removing {item.name}: {str(e)[:40]}")
            # Check if it's a file we should always keep
            elif item.name in keep_files:
                continue  # Keep it
            # Check if it's a runtime file by extension
            elif item.suffix.lower() in keep_extensions:
                continue  # Keep it
            # Otherwise, if it's a markdown file (except task.md), remove it
            elif item.suffix.lower() == '.md':
                try:
                    item.unlink()
                    removed_count += 1
                    think(f"removed {item.name} from {task['id']}")
                except Exception as e:
                    think(f"error removing {item.name}: {str(e)[:40]}")
            # For other files, be conservative and keep them (they might be needed)
    
    if removed_count > 0:
        think(f"cleaned {removed_count} file(s) from {task['id']}")


def run_loop(stop_event=None):
    """Main agent loop. Runs every 5 minutes, sends status ping, and processes tasks."""
    # Target interval: 300 seconds (5 minutes)
    LOOP_INTERVAL = 300
    
    while True:
        # Check if we should stop
        if stop_event and stop_event.is_set():
            break
        
        # Track start time for this iteration
        iteration_start = time.time()
        
        try:
            # Get all tasks from task directories
            tasks = get_all_tasks()
            
            # If no tasks in new structure, try to migrate from old structure
            if not tasks:
                migrate_legacy_tasks()
                tasks = get_all_tasks()
            
            # Clean up any tasks that are already complete (in case they weren't cleaned before)
            completed_tasks = [t for t in tasks if is_task_complete(t)]
            for completed_task in completed_tasks:
                # Check if memory.md still exists (if not, already cleaned)
                if completed_task["memory_file"].exists():
                    think(f"cleaning up completed task {completed_task['id']}...")
                    cleanup_task_context(completed_task)
            
            incomplete = find_incomplete(tasks)
            
            # Send status ping every minute (only if there are tasks)
            if incomplete:
                status_msg = f"📋 Status: {len(incomplete)} task(s) pending"
                if len(incomplete) <= 3:
                    # Show task titles if 3 or fewer
                    task_list = "\n".join([f"  • {t['title'][:50]}" for t in incomplete])
                    status_msg += f"\n{task_list}"
                notify(status_msg)
            
            # Process tasks if there are any
            if incomplete:
                # Double-check that the task we're about to work on is actually incomplete
                # This prevents working on tasks that were just marked complete
                task = incomplete[0]
                if is_task_complete(task):
                    think(f"task {task['id']} is complete, skipping...")
                else:
                    # Report status
                    think(f"working: {task['title'][:40]}")
                    
                    # Read task-specific memory
                    task_memory_content = ""
                    if task["memory_file"].exists():
                        task_memory_content = read_file(str(task["memory_file"]))
                    
                    # Read high-level memory
                    high_level_memory_content = read_file(MEMORY_FILE)
                    
                    # Read chat history
                    chat_history_content = get_chat_history()
                    # Show last 100 lines of chat history to keep prompt manageable
                    chat_history_tail = tail(chat_history_content, 100) if chat_history_content else "No chat history yet."
                    
                    # Read context files for injection
                    identity_content = read_file(IDENTITY_FILE)
                    memory_system_content = read_file(MEMORY_SYSTEM_FILE)
                    
                    # Read narrative/story context
                    story_content = read_file(STORY_FILE)
                    narrative_tail = tail(story_content, 40) if story_content else "No story yet."
                    
                    # Build prompt
                    tasks_text = "\n\n".join(
                        f"## {t['title']} ({t['id']})\n{t['body']}" for t in incomplete
                    )
                    prompt = PROMPT_TEMPLATE.format(
                        identity=identity_content.strip() if identity_content else "",
                        memory_rules=memory_system_content.strip() if memory_system_content else "",
                        narrative=narrative_tail,
                        chat_history=chat_history_tail,
                        tasks=tasks_text,
                        high_level_memory=tail(high_level_memory_content, 30),
                        task_memory=tail(task_memory_content, 50)
                    )
                    
                    # Run agent
                    think("executing...")
                    output = run_cursor(prompt)
                    
                    # Commit any changes made by the agent
                    # Use task title + first part of output as commit description
                    # For /adapt, the prompt itself is a good description
                    commit_desc = f"{task['title']}: {output[:100] if output else 'agent action'}"
                    if "adapt" in task['title'].lower():
                        # Try to extract the original prompt if it's an adapt task
                        commit_desc = f"adapt: {task['title'].replace('/adapt', '').strip()}"
                    
                    git_commit_changes(commit_desc)
                    
                    # The agent's output should contain the detailed action description
                    # We'll use it for task memory and create a summary for high-level memory
                    detailed_content = output.strip()
                    
                    # Create a high-level summary (first sentence or first 150 chars)
                    # Focus on what was accomplished, not how
                    high_level_summary = create_high_level_summary(detailed_content, task["title"])
                    
                    # Append to task memory (detailed)
                    append_task_memory(task, detailed_content)
                    
                    # Append to high-level memory (summary)
                    append_high_level_memory(high_level_summary)
                    
                    think(f"done: {summarize(output)}")
                    
                    # Check if task just became complete and cleanup if so
                    if is_task_complete(task):
                        think(f"task {task['id']} complete, cleaning up context...")
                        cleanup_task_context(task)
            
            # Calculate elapsed time and sleep for remainder of minute
            elapsed = time.time() - iteration_start
            sleep_time = max(0, LOOP_INTERVAL - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
        except Exception as e:
            think(f"error: {str(e)[:40]}")
            # On error, still try to maintain 1-minute interval
            elapsed = time.time() - iteration_start
            sleep_time = max(0, LOOP_INTERVAL - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)


def migrate_legacy_tasks():
    """Migrate tasks from old tasks.md format to new directory structure."""
    legacy_tasks_file = os.path.join(WORKSPACE, "tasks.md")
    if not os.path.exists(legacy_tasks_file):
        return
    
    try:
        content = read_file(legacy_tasks_file)
        if not content.strip():
            return
        
        tasks = parse_tasks(content)
        if not tasks:
            return
        
        think("migrating legacy tasks to new structure...")
        
        for idx, task in enumerate(tasks, start=1):
            task_id = f"task-{idx}"
            task_dir = Path(TASKS_DIR) / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            
            # Write task.md
            task_file = task_dir / "task.md"
            task_content = f"## {task['title']}\n{task['body']}"
            task_file.write_text(task_content)
            
            # Initialize memory.md if it doesn't exist
            memory_file = task_dir / "memory.md"
            if not memory_file.exists():
                memory_file.write_text("# Task Memory\n\n<!-- Detailed memory for this task -->\n")
        
        think(f"migrated {len(tasks)} tasks")
        
    except Exception as e:
        think(f"migration error: {str(e)[:40]}")
