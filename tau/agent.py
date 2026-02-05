"""Minimal two-buffer agent loop with task-specific memory."""

import os
import re
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

from .telegram import think as _think_impl, notify, get_chat_history
from . import processes

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Memory tier paths
ARCHIVE_DIR = os.path.join(WORKSPACE, "context", "archive")
CORE_MEMORY_FILE = os.path.join(WORKSPACE, "context", "memory", "CORE_MEMORY.md")
MID_TERM_FILE = os.path.join(WORKSPACE, "context", "memory", "MID_TERM.md")
SUMMARY_VERSIONS_DIR = os.path.join(WORKSPACE, "context", "summaries")

# Memory configuration
MAX_ACTIVE_MEMORY_ENTRIES = 50
ARCHIVE_AGE_DAYS = 90
SHORT_TERM_DAYS = 7
MID_TERM_DAYS = 30

# Debug mode flag - controlled by __init__.py
_debug_mode = False

def set_debug_mode(enabled: bool):
    """Enable or disable debug mode for verbose notifications."""
    global _debug_mode
    _debug_mode = enabled

def think(msg: str):
    """Send thinking message only if debug mode is enabled."""
    if _debug_mode:
        _think_impl(msg)
TASKS_DIR = os.path.join(WORKSPACE, "context", "tasks")
MEMORY_FILE = os.path.join(WORKSPACE, "context", "tasks", "memory.md")
IDENTITY_FILE = os.path.join(WORKSPACE, "context", "IDENTITY.md")
MEMORY_SYSTEM_FILE = os.path.join(WORKSPACE, "context", "MEMORY-SYSTEM.md")
CHAT_HISTORY_FILE = os.path.join(WORKSPACE, "context", "CHAT.md")
CHAT_SUMMARY_FILE = os.path.join(WORKSPACE, "context", "CHAT_SUMMARY.md")

# Ensure tasks directory exists
os.makedirs(TASKS_DIR, exist_ok=True)

PROMPT_TEMPLATE = """You are Tau, a single-threaded autonomous agent.

{identity}

{memory_rules}

---

CORE MEMORY (persistent facts - rarely changes):
{core_memory}

MID-TERM MEMORY (recent weeks - compressed summaries):
{mid_term_memory}

CONVERSATION SUMMARY (auto-updated hourly - short-term):
{chat_summary}

TELEGRAM CHAT (recent):
{chat_history}

INCOMPLETE TASKS:
{tasks}

HIGH-LEVEL MEMORY (recent activity):
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
- search_skills: source .venv/bin/activate && python -m tau.tools.search_skills "query"
  (Search for creative AI skills - image/video/audio generation, social media posting, etc.)
  Examples:
    - python -m tau.tools.search_skills                    # List all skills
    - python -m tau.tools.search_skills "image"            # Search for image skills
    - python -m tau.tools.search_skills --category video   # Filter by category (image, video, audio, social, utility)
    - python -m tau.tools.search_skills --details flux     # Get detailed info about a specific skill
- create_task: source .venv/bin/activate && python -m tau.tools.create_task "Title" "Description"
  (Create a task for yourself to process later)
- schedule_message: source .venv/bin/activate && python -m tau.tools.schedule_message --in "2h" "message"
  (Schedule a future message: --at "14:00", --in "2h", --cron "0 9 * * *")
- commands: source .venv/bin/activate && python -m tau.tools.commands COMMAND [ARGS]
  (Execute any tau bot command directly - for self-directed behavior)
  Available commands:
    - python -m tau.tools.commands task "description"       # Add a task
    - python -m tau.tools.commands plan "description"       # Create an execution plan
    - python -m tau.tools.commands status                   # See recent activity
    - python -m tau.tools.commands adapt "prompt"           # Self-modify code (triggers restart!)
    - python -m tau.tools.commands cron "5min" "prompt"     # Schedule recurring prompt
    - python -m tau.tools.commands crons                    # List active crons
    - python -m tau.tools.commands uncron 1                 # Remove cron #1
    - python -m tau.tools.commands clear                    # Stop active agent processes
    - python -m tau.tools.commands restart                  # Restart bot process
    - python -m tau.tools.commands kill                     # Stop bot process
    - python -m tau.tools.commands debug                    # Toggle debug mode
"""

def read_file(path: str) -> str:
    """Read file contents, return empty string if missing."""
    if os.path.exists(path):
        return open(path).read()
    return ""


def compress_high_level_memory():
    """Archive old entries and keep only recent active memory.
    
    - Archives entries older than ARCHIVE_AGE_DAYS
    - Keeps only MAX_ACTIVE_MEMORY_ENTRIES most recent entries
    """
    if not os.path.exists(MEMORY_FILE):
        return
    
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    content = read_file(MEMORY_FILE)
    
    if not content.strip():
        return
    
    cutoff_date = datetime.now() - timedelta(days=ARCHIVE_AGE_DAYS)
    archive_entries = []
    active_entries = []
    
    # Parse entries (format: ### YYYY-MM-DD HH:MM\ncontent)
    current_entry = []
    entry_date = None
    
    for line in content.split("\n"):
        if line.startswith("### "):
            if current_entry and entry_date:
                if entry_date < cutoff_date:
                    archive_entries.append("\n".join(current_entry))
                else:
                    active_entries.append("\n".join(current_entry))
            current_entry = [line]
            try:
                date_str = line[4:].strip()[:16]  # "YYYY-MM-DD HH:MM"
                entry_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
            except (ValueError, IndexError):
                entry_date = datetime.now()
        elif current_entry:
            current_entry.append(line)
        # Skip header lines before first entry
    
    # Handle last entry
    if current_entry and entry_date:
        if entry_date < cutoff_date:
            archive_entries.append("\n".join(current_entry))
        else:
            active_entries.append("\n".join(current_entry))
    
    # Archive old entries
    if archive_entries:
        archive_file = os.path.join(ARCHIVE_DIR, f"memory_{datetime.now().strftime('%Y%m')}.md")
        with open(archive_file, "a") as f:
            f.write("\n\n".join(archive_entries) + "\n")
        think(f"archived {len(archive_entries)} old memory entries")
    
    # Keep only last MAX_ACTIVE_MEMORY_ENTRIES
    if len(active_entries) > MAX_ACTIVE_MEMORY_ENTRIES:
        excess = len(active_entries) - MAX_ACTIVE_MEMORY_ENTRIES
        # Archive excess entries (oldest first)
        excess_entries = active_entries[:excess]
        active_entries = active_entries[excess:]
        
        archive_file = os.path.join(ARCHIVE_DIR, f"memory_{datetime.now().strftime('%Y%m')}.md")
        with open(archive_file, "a") as f:
            f.write("\n\n".join(excess_entries) + "\n")
        think(f"archived {excess} excess memory entries (keeping {MAX_ACTIVE_MEMORY_ENTRIES})")
    
    # Rewrite active memory
    with open(MEMORY_FILE, "w") as f:
        f.write("# Memory\n\n<!-- High-level summaries only. Detailed memory is in context/tasks/*/memory.md -->\n")
        if active_entries:
            f.write("\n" + "\n\n".join(active_entries))


def detect_and_archive_stale_entries():
    """Detect memory entries that haven't been referenced recently and archive them.
    
    Entries older than 30 days that aren't referenced in recent chat are considered stale.
    """
    if not os.path.exists(MEMORY_FILE):
        return
    
    content = read_file(MEMORY_FILE)
    if not content.strip():
        return
    
    # Get recent chat for reference checking
    recent_chat = get_chat_history(max_lines=500).lower()
    
    stale_cutoff = datetime.now() - timedelta(days=30)
    stale_entries = []
    active_entries = []
    
    current_entry = []
    entry_date = None
    entry_content = ""
    
    for line in content.split("\n"):
        if line.startswith("### "):
            if current_entry:
                # Check if entry is stale
                age_days = (datetime.now() - entry_date).days if entry_date else 0
                
                # Extract keywords from entry content for reference checking
                words = entry_content.lower().split()[:15]  # First 15 words
                is_referenced = any(
                    len(word) > 4 and word in recent_chat
                    for word in words
                    if word.isalnum()
                )
                
                if age_days > 30 and not is_referenced:
                    stale_entries.append("\n".join(current_entry))
                else:
                    active_entries.append("\n".join(current_entry))
            
            current_entry = [line]
            try:
                date_str = line[4:].strip()[:16]
                entry_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
            except (ValueError, IndexError):
                entry_date = datetime.now()
            entry_content = ""
        elif current_entry:
            current_entry.append(line)
            entry_content += line + " "
    
    # Handle last entry
    if current_entry:
        active_entries.append("\n".join(current_entry))
    
    # Archive stale entries
    if stale_entries:
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        archive_file = os.path.join(ARCHIVE_DIR, f"stale_{datetime.now().strftime('%Y%m%d')}.md")
        with open(archive_file, "w") as f:
            f.write("# Archived Stale Entries\n\n")
            f.write("\n\n".join(stale_entries))
        
        # Rewrite active memory
        with open(MEMORY_FILE, "w") as f:
            f.write("# Memory\n\n<!-- High-level summaries only. Detailed memory is in context/tasks/*/memory.md -->\n")
            if active_entries:
                f.write("\n" + "\n\n".join(active_entries))
        
        think(f"archived {len(stale_entries)} stale memory entries")


def compress_summary_content(content: str, target_ratio: float = 0.5) -> str:
    """Compress a summary to target ratio of original size using the agent."""
    target_chars = int(len(content) * target_ratio)
    
    compress_prompt = f"""Compress this summary to approximately {target_chars} characters.
Keep only the most important information.

Original:
{content}

Output ONLY the compressed summary, no preamble."""

    cmd = [
        "agent",
        "--force",
        "--model",
        "composer-1",
        "--mode=ask",
        "--output-format=text",
        "--print",
        compress_prompt,
    ]

    try:
        from . import processes
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            stdin=subprocess.DEVNULL,
            cwd=WORKSPACE,
            start_new_session=True,
        )
        processes.track(proc, label="agent:compress", cmd=cmd, own_process_group=True)
        stdout, stderr = proc.communicate(timeout=120)
        processes.untrack(proc)
        result = stdout.strip() if stdout.strip() else stderr.strip()
        return result[:target_chars + 100]  # Allow slight overage
    except Exception:
        return content[:target_chars]  # Fallback to truncation


def extract_core_facts(summary_content: str) -> list[str]:
    """Extract important facts that should be stored in core memory."""
    extract_prompt = f"""Analyze this summary and extract ONLY persistent facts that are:
1. User preferences or requirements
2. Important decisions made
3. Technical constraints discovered
4. Recurring patterns

Summary:
{summary_content}

Return a bullet list of core facts (each starting with "- "), or "NONE" if no new core facts.
Be very selective - only truly persistent information that would be useful months later."""

    cmd = [
        "agent",
        "--force",
        "--model",
        "composer-1",
        "--mode=ask",
        "--output-format=text",
        "--print",
        extract_prompt,
    ]

    try:
        from . import processes
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            stdin=subprocess.DEVNULL,
            cwd=WORKSPACE,
            start_new_session=True,
        )
        processes.track(proc, label="agent:extract_facts", cmd=cmd, own_process_group=True)
        stdout, stderr = proc.communicate(timeout=120)
        processes.untrack(proc)
        
        facts = stdout.strip() if stdout.strip() else stderr.strip()
        if "NONE" in facts.upper() or not facts:
            return []
        
        return [f.strip() for f in facts.split("\n") if f.strip().startswith("-")]
    except Exception:
        return []


def migrate_summaries_to_tiers():
    """Migrate old summaries through memory tiers.
    
    - Summaries older than MID_TERM_DAYS: Extract core facts, then archive
    - Summaries older than SHORT_TERM_DAYS: Compress and move to mid-term
    """
    if not os.path.exists(SUMMARY_VERSIONS_DIR):
        return
    
    versions = sorted(Path(SUMMARY_VERSIONS_DIR).glob("summary_*.md"))
    
    for version_file in versions:
        # Parse timestamp from filename (format: summary_YYYYMMDD_HHMMSS.md)
        timestamp_str = version_file.stem.replace("summary_", "")
        try:
            file_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        
        age_days = (datetime.now() - file_date).days
        
        if age_days > MID_TERM_DAYS:
            # Extract core facts before archiving
            content = version_file.read_text()
            
            # Extract just the summary part
            if "## Summary" in content:
                summary_part = content.split("## Summary", 1)[1].split("---")[0].strip()
            else:
                summary_part = content
            
            core_facts = extract_core_facts(summary_part)
            
            if core_facts:
                os.makedirs(os.path.dirname(CORE_MEMORY_FILE), exist_ok=True)
                with open(CORE_MEMORY_FILE, "a") as f:
                    f.write(f"\n### Extracted {datetime.now().strftime('%Y-%m-%d')}\n")
                    f.write("\n".join(core_facts) + "\n")
                think(f"extracted {len(core_facts)} core facts from old summary")
            
            # Archive the summary
            os.makedirs(ARCHIVE_DIR, exist_ok=True)
            archive_file = os.path.join(ARCHIVE_DIR, f"summaries_{file_date.strftime('%Y%m')}.md")
            with open(archive_file, "a") as f:
                f.write(f"\n---\n{content}\n")
            
            version_file.unlink()
            think(f"archived summary from {file_date.strftime('%Y-%m-%d')}")
        
        elif age_days > SHORT_TERM_DAYS:
            # Compress and move to mid-term
            content = version_file.read_text()
            
            # Extract just the summary part
            if "## Summary" in content:
                summary_part = content.split("## Summary", 1)[1].split("---")[0].strip()
            else:
                summary_part = content
            
            if len(summary_part) > 200:
                compressed = compress_summary_content(summary_part)
            else:
                compressed = summary_part
            
            os.makedirs(os.path.dirname(MID_TERM_FILE), exist_ok=True)
            with open(MID_TERM_FILE, "a") as f:
                f.write(f"\n### {file_date.strftime('%Y-%m-%d')}\n{compressed}\n")
            
            think(f"migrated summary from {file_date.strftime('%Y-%m-%d')} to mid-term")


def run_memory_maintenance():
    """Run all memory maintenance tasks.
    
    Called once at startup and then daily by the maintenance loop.
    """
    try:
        think("running memory maintenance...")
        
        # 1. Compress high-level memory
        compress_high_level_memory()
        
        # 2. Migrate summaries through tiers
        migrate_summaries_to_tiers()
        
        # 3. Detect and archive stale entries
        detect_and_archive_stale_entries()
        
        think("memory maintenance complete")
    except Exception as e:
        think(f"memory maintenance error: {str(e)[:40]}")


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


def run_cursor(prompt: str) -> str:
    """Run Cursor agent with prompt, return output."""
    cmd = [
        "agent",
        "--force",
        "--model",
        "composer-1",
        "--output-format=text",
        "--print",
        prompt,
    ]

    proc = None
    stdout = ""
    stderr = ""
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            stdin=subprocess.DEVNULL,
            cwd=WORKSPACE,
            start_new_session=True,
        )
        processes.track(proc, label="agent:loop", cmd=cmd, own_process_group=True)
        stdout, stderr = proc.communicate(timeout=86400)  # 24 hours
    except subprocess.TimeoutExpired:
        # Kill the whole process group (agent can spawn children).
        try:
            import signal as _signal
            if proc is not None:
                os.killpg(proc.pid, _signal.SIGTERM)
        except Exception:
            try:
                if proc is not None:
                    proc.terminate()
            except Exception:
                pass
        try:
            if proc is not None:
                proc.wait(timeout=2)
        except Exception:
            try:
                import signal as _signal
                if proc is not None:
                    os.killpg(proc.pid, _signal.SIGKILL)
            except Exception:
                try:
                    if proc is not None:
                        proc.kill()
                except Exception:
                    pass
        return "Error: Agent timed out after 24 hours"
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        processes.untrack(proc)

    if proc is not None and processes.pop_cancelled(proc.pid):
        return "Cancelled."

    out = stdout.strip() if stdout and stdout.strip() else (stderr or "").strip()
    return out


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
                    
                    # Read tiered memory
                    core_memory_content = read_file(CORE_MEMORY_FILE)
                    mid_term_memory_content = read_file(MID_TERM_FILE)
                    
                    # Read chat history
                    chat_history_content = get_chat_history()
                    # Show last 100 lines of chat history to keep prompt manageable
                    chat_history_tail = tail(chat_history_content, 100) if chat_history_content else "No chat history yet."
                    
                    # Read chat summary (hourly updated)
                    chat_summary_content = read_file(CHAT_SUMMARY_FILE)
                    chat_summary = chat_summary_content.strip() if chat_summary_content else "No summary available yet."
                    
                    # Read context files for injection
                    identity_content = read_file(IDENTITY_FILE)
                    memory_system_content = read_file(MEMORY_SYSTEM_FILE)
                    
                    # Build prompt
                    tasks_text = "\n\n".join(
                        f"## {t['title']} ({t['id']})\n{t['body']}" for t in incomplete
                    )
                    prompt = PROMPT_TEMPLATE.format(
                        identity=identity_content.strip() if identity_content else "",
                        memory_rules=memory_system_content.strip() if memory_system_content else "",
                        core_memory=tail(core_memory_content, 30) if core_memory_content.strip() else "No core memory yet.",
                        mid_term_memory=tail(mid_term_memory_content, 20) if mid_term_memory_content.strip() else "No mid-term memory yet.",
                        chat_summary=chat_summary,
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
