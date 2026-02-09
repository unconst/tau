import os
import json
import queue
import subprocess
import sys
import threading
import tempfile
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

from openai import OpenAI
from .telegram import (
    bot, save_chat_id, notify, WORKSPACE, append_chat_history,
    TelegramStreamingMessage, authorize, is_owner, is_private_chat,
    is_group_chat, save_chat_metadata, list_chats, get_chat_history_for,
    send_to_chat,
)
from .agent import run_loop, TASKS_DIR, get_all_tasks, git_commit_changes, set_debug_mode, read_file, run_memory_maintenance
from .tools.commands import run_command
from . import processes
import re


# Intent patterns for command routing
# Each pattern maps to (command_name, arg_extractor_function_or_None)
INTENT_PATTERNS = {
    # Adapt/modify patterns
    "adapt": [
        r"(?:update|modify|change|edit|adapt|improve|fix|add|implement|create)\s+(?:your|the|my|tau'?s?)?\s*(?:code|yourself|bot|implementation)",
        r"(?:add|implement|create)\s+(?:a\s+)?(?:new\s+)?(?:feature|functionality|capability)",
        r"can you (?:update|modify|change|add|implement)",
        r"i want (?:you )?to (?:update|modify|change|add|implement)",
        r"(?:make|do) (?:a )?(?:change|modification|update) to",
        r"self[- ]?modify",
    ],
    # Task patterns
    "task": [
        r"(?:add|create|make|new)\s+(?:a\s+)?task",
        r"(?:add|put)\s+(?:this\s+)?(?:to|on)\s+(?:my\s+)?(?:todo|task)",
        r"todo[:\s]+",
        r"i need (?:you )?to (?:work on|do|complete)",
        r"work on[:\s]",
        r"remember to",
    ],
    # Plan patterns
    "plan": [
        r"(?:create|make|generate)\s+(?:a\s+)?(?:plan|roadmap|strategy)",
        r"(?:how should (?:we|i|you) (?:approach|implement|do))",
        r"(?:plan|outline) (?:for|how to)",
        r"what(?:'s| is) the (?:plan|approach|strategy) for",
    ],
    # Status patterns
    "status": [
        r"(?:what(?:'s| is|'re| are) (?:you )?(?:working on|doing|up to))",
        r"(?:show|get|check)\s+(?:me\s+)?(?:the\s+)?(?:my\s+)?(?:status|progress|tasks?)",
        r"(?:what(?:'s| is)) (?:the\s+)?(?:my\s+)?(?:status|progress)",
        r"(?:what|any) (?:tasks?|work|progress)",
        r"recent activity",
    ],
    # Cron patterns
    "cron": [
        r"(?:remind me|set (?:a )?reminder|schedule)\s+(?:to\s+)?(?:in\s+)?(\d+\s*(?:min(?:ute)?s?|h(?:our)?s?|sec(?:ond)?s?))",
        r"(?:every|each)\s+(\d+\s*(?:min(?:ute)?s?|h(?:our)?s?|sec(?:ond)?s?))\s+(?:run|do|check|send)",
        r"(?:set up|create|add)\s+(?:a\s+)?(?:recurring|scheduled|cron)",
        r"(?:in|after)\s+(\d+\s*(?:min(?:ute)?s?|h(?:our)?s?|sec(?:ond)?s?))\s+(?:remind|tell|notify|send)",
    ],
    # Crons list patterns
    "crons": [
        r"(?:list|show|what(?:'s| is|'re| are))\s+(?:my\s+)?(?:scheduled|recurring|cron|reminder)",
        r"(?:active|current)\s+(?:cron|reminder|schedule)",
        r"what(?:'s| is) scheduled",
    ],
    # Uncron patterns
    "uncron": [
        r"(?:remove|delete|cancel|stop)\s+(?:cron|reminder|schedule)\s*#?(\d+)",
        r"(?:uncron|remove cron)\s*#?(\d+)",
        r"stop (?:cron|reminder)\s*#?(\d+)",
    ],
    # Clear patterns
    "clear": [
        r"(?:stop|cancel|clear|kill)\s+(?:all\s+)?(?:active\s+)?(?:agent|process)",
        r"(?:stop|cancel) what(?:'s| is| you(?:'re| are)) (?:running|doing)",
    ],
    # Restart patterns
    "restart": [
        r"(?:restart|reboot)\s+(?:yourself|the bot|tau)",
        r"(?:can you )?restart",
    ],
    # Kill/stop patterns
    "kill": [
        r"(?:shutdown|stop|kill)\s+(?:yourself|the bot|tau|completely)",
        r"turn (?:yourself )?off",
    ],
    # Debug patterns
    "debug": [
        r"(?:toggle|turn (?:on|off)|enable|disable)\s+debug",
        r"debug mode",
    ],
}


def classify_intent(message: str) -> dict | None:
    """Classify user intent and return command info if applicable.
    
    Uses pattern matching for fast detection. Falls back to None for
    ambiguous cases (let normal chat handle it).
    
    Args:
        message: The user's message text
        
    Returns:
        dict with 'command', 'args', 'confirmation_message', 'needs_confirmation'
        or None if this should be handled as normal chat
    """
    message_lower = message.lower().strip()
    
    # Skip very short messages or questions about commands
    if len(message_lower) < 5:
        return None
    
    # Skip if it looks like a question about how things work (not a command)
    question_words = ["what is", "what's", "how does", "how do", "can you explain", "tell me about"]
    if any(message_lower.startswith(q) for q in question_words):
        # Exception: "what's the status" should trigger status
        if not any(p in message_lower for p in ["status", "working on", "scheduled"]):
            return None
    
    # Check each command's patterns
    for command, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, message_lower, re.IGNORECASE)
            if match:
                return _build_intent_result(command, message, match)
    
    return None


def _build_intent_result(command: str, original_message: str, match: re.Match) -> dict:
    """Build the intent result dict with appropriate args and confirmations."""
    
    # Commands that need confirmation
    needs_confirmation = command in ("adapt", "kill", "restart")
    
    # Extract arguments based on command type
    args = []
    confirmation_msg = ""
    
    if command == "adapt":
        # The full message (minus pattern match prefix) is the adaptation prompt
        args = [original_message]
        confirmation_msg = f"I'll modify my code based on your request. This will restart the bot. Proceed? (yes/no)"
        
    elif command == "task":
        # Extract the task description from the message
        # Try to find everything after the trigger phrase
        task_text = original_message
        # Remove common prefixes
        for prefix in ["add a task", "add task", "create a task", "create task", "new task", 
                       "todo:", "todo ", "add to my todo", "work on", "i need you to", 
                       "i need to", "remember to"]:
            lower = task_text.lower()
            if lower.startswith(prefix):
                task_text = task_text[len(prefix):].strip()
                break
        # Remove leading punctuation/spaces
        task_text = task_text.lstrip(":- ").strip()
        if task_text:
            args = [task_text]
        confirmation_msg = f"Adding task: {task_text[:50]}..."
        
    elif command == "plan":
        # Extract what to plan
        plan_text = original_message
        for prefix in ["create a plan for", "create plan for", "make a plan for", 
                       "plan for", "plan:", "how should we approach", "how should i approach"]:
            lower = plan_text.lower()
            if lower.startswith(prefix):
                plan_text = plan_text[len(prefix):].strip()
                break
        plan_text = plan_text.lstrip(":- ").strip()
        if plan_text:
            args = [plan_text]
        confirmation_msg = f"Creating plan for: {plan_text[:50]}..."
        
    elif command == "cron":
        # Try to extract interval and prompt
        # Look for time patterns
        time_match = re.search(r'(\d+)\s*(min(?:ute)?s?|h(?:our)?s?|sec(?:ond)?s?)', original_message.lower())
        if time_match:
            interval = f"{time_match.group(1)}{time_match.group(2)[0]}"  # e.g., "5m" or "1h"
            # The prompt is everything else, cleaned up
            prompt = original_message
            # Remove time portion and common prefixes
            for prefix in ["remind me", "set a reminder", "schedule", "every", "in", "after"]:
                lower = prompt.lower()
                idx = lower.find(prefix)
                if idx != -1:
                    prompt = prompt[idx + len(prefix):].strip()
            # Remove the time expression
            prompt = re.sub(r'\d+\s*(?:min(?:ute)?s?|h(?:our)?s?|sec(?:ond)?s?)', '', prompt, flags=re.IGNORECASE).strip()
            prompt = prompt.lstrip(":,- ").strip()
            if prompt:
                args = [interval, prompt]
            confirmation_msg = f"Scheduling reminder every {interval}: {prompt[:40]}..."
        else:
            return None  # Can't parse cron without interval
            
    elif command == "uncron":
        # Extract cron ID
        id_match = re.search(r'#?(\d+)', original_message)
        if id_match:
            args = [id_match.group(1)]
            confirmation_msg = f"Removing cron #{id_match.group(1)}..."
        else:
            return None
            
    elif command == "status":
        confirmation_msg = "Checking status..."
        
    elif command == "crons":
        confirmation_msg = "Listing scheduled jobs..."
        
    elif command == "clear":
        confirmation_msg = "Stopping active processes..."
        
    elif command == "restart":
        confirmation_msg = "I'll restart now. This will briefly disconnect me."
        needs_confirmation = True
        
    elif command == "kill":
        confirmation_msg = "This will completely stop me. Are you sure? (yes/no)"
        needs_confirmation = True
        
    elif command == "debug":
        confirmation_msg = "Toggling debug mode..."
    
    return {
        "command": command,
        "args": args,
        "confirmation_message": confirmation_msg,
        "needs_confirmation": needs_confirmation,
        "original_message": original_message,
    }


def execute_intent(intent: dict, chat_id: int, message_id: int) -> str:
    """Execute a detected intent by calling the appropriate command.
    
    Args:
        intent: The intent dict from classify_intent()
        chat_id: Telegram chat ID
        message_id: Message ID to reply to
        
    Returns:
        Result message to send to user
    """
    command = intent["command"]
    args = intent["args"]
    
    # For adapt, we need special handling with streaming
    if command == "adapt" and args:
        # Use the streaming adapt function
        result = run_adapt_streaming(
            args[0],
            chat_id=chat_id,
            reply_to_message_id=message_id,
            timeout_seconds=3600,
        )
        # Commit changes and restart
        git_commit_changes(args[0][:100] if args else "self-modification")
        bot.stop_polling()
        if restart_via_supervisor():
            sys.exit(0)
        else:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        return result
        
    elif command == "plan" and args:
        # Use the streaming plan function
        plan_content, plan_filename = run_plan_generation(
            args[0],
            chat_id=chat_id,
            reply_to_message_id=message_id,
            timeout_seconds=600,
        )
        return plan_content if plan_filename else "Failed to generate plan."
        
    elif command == "restart":
        bot.send_message(chat_id, "Restarting...")
        append_chat_history("assistant", "Restarting...")
        bot.stop_polling()
        if restart_via_supervisor():
            sys.exit(0)
        else:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        return "Restarting..."
        
    elif command == "kill":
        bot.send_message(chat_id, "🛑 Shutting down...")
        append_chat_history("assistant", "🛑 Shutting down...")
        try:
            processes.terminate_all(label_prefix="agent:", timeout_seconds=2.0)
        except Exception:
            pass
        try:
            _stop_event.set()
        except Exception:
            pass
        try:
            bot.stop_polling()
        except Exception:
            pass
        if stop_via_supervisor():
            os._exit(0)
        os._exit(0)
        
    else:
        # Use the commands module for other commands
        result = run_command(command, *args)
        return result


# Track pending confirmations: chat_id -> intent dict
_pending_confirmations: dict[int, dict] = {}

# Debug mode flag - controls verbose notifications
DEBUG_MODE = False

# Memory tier paths
SUMMARY_VERSIONS_DIR = os.path.join(WORKSPACE, "context", "summaries")
ARCHIVE_DIR = os.path.join(WORKSPACE, "context", "archive")
CORE_MEMORY_FILE = os.path.join(WORKSPACE, "context", "memory", "CORE_MEMORY.md")
MID_TERM_FILE = os.path.join(WORKSPACE, "context", "memory", "MID_TERM.md")
LAST_CHAT_POSITION_FILE = os.path.join(WORKSPACE, "context", ".chat_position")

# Memory configuration
MAX_SUMMARY_VERSIONS = 5
MAX_ACTIVE_MEMORY_ENTRIES = 50
ARCHIVE_AGE_DAYS = 90
SHORT_TERM_DAYS = 7
MID_TERM_DAYS = 30

# Storage for active cron jobs: list of {id, interval_seconds, prompt, next_run, chat_id}
_cron_jobs = []
_cron_lock = threading.Lock()
_next_cron_id = 1
CRON_FILE = os.path.join(WORKSPACE, "cron_jobs.json")


def save_crons():
    """Persist cron jobs to disk. Must be called with _cron_lock held."""
    try:
        with open(CRON_FILE, 'w', encoding='utf-8') as f:
            json.dump({'jobs': _cron_jobs, 'next_id': _next_cron_id}, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save cron jobs: {e}")


def load_crons():
    """Load cron jobs from disk at startup."""
    global _cron_jobs, _next_cron_id
    if os.path.exists(CRON_FILE):
        try:
            with open(CRON_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            with _cron_lock:
                _cron_jobs = data.get('jobs', [])
                _next_cron_id = data.get('next_id', 1)
                # Filter out jobs with missing chat_id (invalid jobs)
                valid_jobs = []
                for job in _cron_jobs:
                    if job.get('chat_id') is None:
                        logger.warning(f"Skipping cron job #{job.get('id')} with missing chat_id")
                        continue
                    valid_jobs.append(job)
                _cron_jobs = valid_jobs
                # Reset next_run times to now + interval (so they don't all fire immediately)
                now = time.time()
                for job in _cron_jobs:
                    job['next_run'] = now + job['interval_seconds']
                # Save updated state (with reset next_run and filtered invalid jobs)
                save_crons()
            logger.info(f"Loaded {len(_cron_jobs)} cron job(s) from disk")
        except Exception as e:
            logger.error(f"Failed to load cron jobs: {e}")

# Configure logging
LOG_FILE = os.path.join(WORKSPACE, "context", "logs", "tau.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MEMORY_FILE = os.path.join(WORKSPACE, "context", "tasks", "memory.md")
CHAT_SUMMARY_FILE = os.path.join(WORKSPACE, "context", "CHAT_SUMMARY.md")

# Event to signal agent loop to stop
_stop_event = threading.Event()

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    notify("⚠️ Warning: OPENAI_API_KEY not set. Voice transcription will not work.")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def download_voice_file(file_id: str) -> str:
    """Download a voice file from Telegram and return the local file path."""
    try:
        file_info = bot.get_file(file_id)
        file_path = file_info.file_path
        
        # Create temporary file to store the voice
        with tempfile.NamedTemporaryFile(delete=False, suffix='.ogg') as tmp_file:
            temp_path = tmp_file.name
        
        # Download the file
        downloaded_file = bot.download_file(file_path)
        with open(temp_path, 'wb') as f:
            f.write(downloaded_file)
        
        return temp_path
    except Exception as e:
        raise Exception(f"Failed to download voice file: {str(e)}")


def transcribe_voice(voice_path: str) -> str:
    """Transcribe a voice file using OpenAI Whisper API."""
    if not openai_client:
        raise Exception("OpenAI API key not configured")
    
    try:
        with open(voice_path, 'rb') as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        raise Exception(f"Failed to transcribe voice: {str(e)}")


def parse_interval(interval_str: str) -> int:
    """Parse interval string like '5min', '30sec', '2h' into seconds.
    
    Supported formats:
    - Xsec, Xs, Xsecond, Xseconds
    - Xmin, Xm, Xminute, Xminutes  
    - Xh, Xhr, Xhour, Xhours
    
    If parsing fails, assumes minutes.
    Returns interval in seconds.
    """
    interval_str = interval_str.lower().strip()
    
    # Try to extract number and unit
    match = re.match(r'^(\d+)\s*([a-z]*)$', interval_str)
    if not match:
        # Try to parse as just a number (assume minutes)
        try:
            return int(interval_str) * 60
        except ValueError:
            return 5 * 60  # Default to 5 minutes
    
    number = int(match.group(1))
    unit = match.group(2)
    
    if unit in ('s', 'sec', 'second', 'seconds'):
        return number
    elif unit in ('m', 'min', 'minute', 'minutes'):
        return number * 60
    elif unit in ('h', 'hr', 'hour', 'hours'):
        return number * 3600
    else:
        # Default to minutes if unit not recognized
        return number * 60


def save_summary_with_version(summary_content: str):
    """Save summary with versioning for rollback capability."""
    os.makedirs(SUMMARY_VERSIONS_DIR, exist_ok=True)
    
    # Save new version with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_file = os.path.join(SUMMARY_VERSIONS_DIR, f"summary_{timestamp}.md")
    with open(version_file, "w") as f:
        f.write(summary_content)
    
    # Update main summary file
    with open(CHAT_SUMMARY_FILE, "w") as f:
        f.write(summary_content)
    
    # Rotate old versions (keep only MAX_SUMMARY_VERSIONS)
    versions = sorted(Path(SUMMARY_VERSIONS_DIR).glob("summary_*.md"))
    for old_version in versions[:-MAX_SUMMARY_VERSIONS]:
        old_version.unlink()
    
    logger.info(f"Summary version saved: {version_file}")


def get_new_chat_since_last_summary() -> tuple[str, int]:
    """Get chat history added since last summary generation.
    
    Returns:
        Tuple of (new_chat_content, new_position)
    """
    from .telegram import CHAT_HISTORY_FILE
    
    last_pos = 0
    if os.path.exists(LAST_CHAT_POSITION_FILE):
        try:
            with open(LAST_CHAT_POSITION_FILE) as f:
                last_pos = int(f.read().strip())
        except (ValueError, IOError):
            pass
    
    if not os.path.exists(CHAT_HISTORY_FILE):
        return "", 0
    
    with open(CHAT_HISTORY_FILE) as f:
        full_chat = f.read()
    
    new_chat = full_chat[last_pos:]
    return new_chat, len(full_chat)


def save_chat_position(position: int):
    """Save the current chat position for incremental updates."""
    os.makedirs(os.path.dirname(LAST_CHAT_POSITION_FILE), exist_ok=True)
    with open(LAST_CHAT_POSITION_FILE, "w") as f:
        f.write(str(position))


def generate_incremental_summary(old_summary: str, new_chat_delta: str) -> str:
    """Generate summary incrementally, updating only changed parts."""
    prompt = f"""Compare the existing summary with new chat history.
Update the summary to incorporate new information.

EXISTING SUMMARY:
{old_summary}

NEW CHAT HISTORY (since last update):
{new_chat_delta}

Rules:
1. Preserve unchanged information
2. Add new important information  
3. Update information that has changed
4. Remove outdated information
5. Keep under 2000 characters
6. Use bullet points for easy scanning

Output ONLY the updated summary, no preamble or explanation."""

    from .codex import build_cmd, CHAT_MODEL
    cmd = build_cmd(prompt, model=CHAT_MODEL, readonly=True)

    proc = None
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
        processes.track(proc, label="agent:summary:incremental", cmd=cmd, own_process_group=True)
        stdout, stderr = proc.communicate(timeout=300)
        from .codex import strip_think_tags
        raw = stdout.strip() if stdout.strip() else stderr.strip()
        return strip_think_tags(raw)
    except subprocess.TimeoutExpired:
        try:
            processes.terminate_all(label_prefix="agent:summary:incremental", timeout_seconds=1.0)
        except Exception:
            pass
        return ""
    finally:
        if proc:
            processes.untrack(proc)


def run_summary_loop(stop_event=None):
    """Background loop that summarizes chat history every hour.
    
    Uses incremental updates when possible to avoid full regeneration.
    Saves versioned summaries for rollback capability.
    """
    SUMMARY_INTERVAL = 3600  # 1 hour
    
    while True:
        if stop_event and stop_event.is_set():
            break
        
        try:
            # Get new chat since last summary
            from .telegram import CHAT_HISTORY_FILE
            if not os.path.exists(CHAT_HISTORY_FILE):
                time.sleep(SUMMARY_INTERVAL)
                continue
            
            new_chat, new_position = get_new_chat_since_last_summary()
            
            # Also get full chat for fallback
            with open(CHAT_HISTORY_FILE) as f:
                full_chat = f.read()
            
            if not full_chat.strip() or len(full_chat) < 500:
                # Not enough history to summarize
                time.sleep(SUMMARY_INTERVAL)
                continue
            
            # Check if we have enough new content for incremental update
            use_incremental = False
            old_summary = ""
            
            if os.path.exists(CHAT_SUMMARY_FILE) and new_chat.strip() and len(new_chat) > 100:
                # We have a previous summary and meaningful new content - use incremental
                with open(CHAT_SUMMARY_FILE) as f:
                    old_summary_file = f.read()
                # Extract just the summary content (after "## Summary" header)
                if "## Summary" in old_summary_file:
                    old_summary = old_summary_file.split("## Summary", 1)[1].split("---")[0].strip()
                    if old_summary:
                        use_incremental = True
                        logger.info(f"Using incremental summary update ({len(new_chat)} new chars)")
            
            if use_incremental:
                # Incremental update
                summary = generate_incremental_summary(old_summary, new_chat)
            else:
                # Full regeneration
                logger.info("Running full chat summary regeneration...")
                
                summary_prompt = f"""You are Tau's memory system. Your job is to extract and summarize the important information from the conversation history below.

FULL CHAT HISTORY:
{full_chat}

---

Please create a concise summary that captures:
1. Key topics discussed
2. Important decisions made
3. User preferences or requirements mentioned
4. Ongoing tasks or projects
5. Any commitments or promises made
6. Technical context (e.g., what the user is working on)

The summary should be:
- Short enough to fit in an agent's context window (under 2000 characters)
- Focused on actionable information that would help future interactions
- Written in bullet points for easy scanning
- Updated to reflect the current state (not historical play-by-play)

Output ONLY the summary content, no preamble or explanation."""

                from .codex import build_cmd, CHAT_MODEL
                cmd = build_cmd(summary_prompt, model=CHAT_MODEL, readonly=True)

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
                    processes.track(
                        proc,
                        label="agent:summary",
                        cmd=cmd,
                        own_process_group=True,
                    )
                    stdout, stderr = proc.communicate(timeout=300)  # 5 minutes
                except subprocess.TimeoutExpired:
                    try:
                        processes.terminate_all(label_prefix="agent:summary", timeout_seconds=1.0)
                    except Exception:
                        pass
                    logger.error("Chat summary timed out")
                    time.sleep(SUMMARY_INTERVAL)
                    continue
                finally:
                    if proc:
                        processes.untrack(proc)

                if proc is not None and processes.pop_cancelled(proc.pid):
                    logger.info("Chat summary cancelled")
                    time.sleep(SUMMARY_INTERVAL)
                    continue
                
                from .codex import strip_think_tags
                summary = strip_think_tags(stdout.strip() if stdout.strip() else stderr.strip())
            
            if summary and len(summary) > 50:
                # Build and save summary with versioning
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                summary_content = f"""# Chat Summary

<!-- This file is automatically updated hourly by Tau to summarize the full conversation history. -->
<!-- It provides context to all agent calls so they understand the conversation without reading the full history. -->

## Summary

{summary}

---
*Last updated: {timestamp}*
"""
                save_summary_with_version(summary_content)
                save_chat_position(new_position)
                logger.info(f"Chat summary updated ({len(summary)} chars)")
            else:
                logger.warning("Chat summary produced no useful output")

        except Exception as e:
            logger.error(f"Chat summary error: {e}")
        
        # Sleep for 1 hour
        time.sleep(SUMMARY_INTERVAL)


def run_memory_maintenance_loop(stop_event=None):
    """Background loop for memory maintenance (runs daily).
    
    Handles:
    - Compressing high-level memory (archiving old entries)
    - Migrating summaries through tiers (short-term → mid-term → archive)
    - Detecting and archiving stale entries
    """
    MAINTENANCE_INTERVAL = 86400  # 24 hours
    
    # Run maintenance once at startup
    try:
        logger.info("Running initial memory maintenance...")
        run_memory_maintenance()
        logger.info("Initial memory maintenance complete")
    except Exception as e:
        logger.error(f"Initial memory maintenance error: {e}")
    
    while True:
        if stop_event and stop_event.is_set():
            break
        
        # Sleep first, then run maintenance (since we ran at startup)
        time.sleep(MAINTENANCE_INTERVAL)
        
        if stop_event and stop_event.is_set():
            break
        
        try:
            logger.info("Running daily memory maintenance...")
            run_memory_maintenance()
            logger.info("Memory maintenance complete")
        except Exception as e:
            logger.error(f"Memory maintenance error: {e}")


def run_cron_loop(stop_event=None):
    """Background loop that runs cron jobs at their scheduled times."""
    import time
    
    while True:
        if stop_event and stop_event.is_set():
            break
        
        now = time.time()
        jobs_to_run = []
        
        with _cron_lock:
            for job in _cron_jobs:
                if now >= job['next_run']:
                    jobs_to_run.append(job.copy())
                    # Schedule next run
                    job['next_run'] = now + job['interval_seconds']
        
        # Execute jobs outside the lock
        for job in jobs_to_run:
            try:
                logger.info(f"Running cron job {job['id']}: {job['prompt'][:50]}...")
                
                # Build prompt with context
                from .telegram import get_chat_history
                # Get recent chat history (last 50 lines is enough for cron context)
                chat_history = get_chat_history(max_lines=50)
                
                # Get chat summary for broader context
                chat_summary = ""
                if os.path.exists(CHAT_SUMMARY_FILE):
                    with open(CHAT_SUMMARY_FILE) as f:
                        chat_summary = f.read().strip()
                
                prompt_with_context = f"""CONVERSATION SUMMARY (auto-updated hourly):
{chat_summary if chat_summary else "No summary available yet."}

TELEGRAM CHAT HISTORY (recent):
{chat_history}

CRON JOB PROMPT (runs every {job['interval_seconds']}s):
{job['prompt']}

Please execute this scheduled task. Provide a fresh update based on the current state or by using your tools. Do not simply repeat previous messages."""

                from .codex import build_cmd, AGENT_MODEL
                cmd = build_cmd(prompt_with_context, model=AGENT_MODEL)

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
                    processes.track(
                        proc,
                        label=f"agent:cron:{job['id']}",
                        cmd=cmd,
                        own_process_group=True,
                    )
                    stdout, stderr = proc.communicate(timeout=600)  # 10 minutes
                except subprocess.TimeoutExpired:
                    # Ensure we clean up the process group before propagating.
                    try:
                        processes.terminate_all(label_prefix=f"agent:cron:{job['id']}", timeout_seconds=1.0)
                    except Exception:
                        pass
                    raise
                finally:
                    processes.untrack(proc)

                if proc is not None and processes.pop_cancelled(proc.pid):
                    logger.info(f"Cron job {job['id']} cancelled")
                    continue

                from .codex import strip_think_tags
                response = strip_think_tags(stdout.strip() if stdout.strip() else stderr.strip())
                if response:
                    # Send result to user
                    try:
                        bot.send_message(job['chat_id'], f"⏰ Cron #{job['id']}:\n{response[:4000]}")
                        append_chat_history("assistant", f"[cron #{job['id']}]: {response}", chat_id=job['chat_id'])
                    except Exception as e:
                        logger.error(f"Failed to send cron result: {e}")
                        
            except subprocess.TimeoutExpired:
                logger.error(f"Cron job {job['id']} timed out")
            except Exception as e:
                logger.error(f"Cron job {job['id']} error: {e}")
        
        # Sleep for 1 second between checks
        time.sleep(1)


@bot.message_handler(commands=['cron'])
def add_cron(message):
    """Add a cron job that runs a prompt at specified intervals."""
    global _next_cron_id
    
    save_chat_metadata(message)
    if not authorize(message):
        return
    save_chat_id(message.chat.id)
    
    # Parse: /cron <interval> <prompt>
    text = message.text.replace('/cron', '', 1).strip()
    append_chat_history("user", f"/cron {text}", chat_id=message.chat.id)
    
    if not text:
        response = "Usage: /cron <interval> <prompt>\nExamples:\n  /cron 5min check the weather\n  /cron 30sec ping\n  /cron 1h summarize news"
        bot.reply_to(message, response)
        append_chat_history("assistant", response, chat_id=message.chat.id)
        return
    
    # Split into interval and prompt
    parts = text.split(None, 1)
    if len(parts) < 2:
        response = "Usage: /cron <interval> <prompt>\nNeed both an interval and a prompt."
        bot.reply_to(message, response)
        append_chat_history("assistant", response, chat_id=message.chat.id)
        return
    
    interval_str, prompt = parts
    interval_seconds = parse_interval(interval_str)
    
    # Create cron job
    import time
    with _cron_lock:
        cron_id = _next_cron_id
        _next_cron_id += 1
        
        job = {
            'id': cron_id,
            'interval_seconds': interval_seconds,
            'prompt': prompt,
            'next_run': time.time() + interval_seconds,
            'chat_id': message.chat.id
        }
        _cron_jobs.append(job)
        save_crons()
    
    # Format interval for display
    if interval_seconds < 60:
        interval_display = f"{interval_seconds}sec"
    elif interval_seconds < 3600:
        interval_display = f"{interval_seconds // 60}min"
    else:
        interval_display = f"{interval_seconds // 3600}h"
    
    response = f"✅ Cron #{cron_id} created: every {interval_display}\nPrompt: {prompt[:100]}"
    bot.reply_to(message, response)
    append_chat_history("assistant", response, chat_id=message.chat.id)


@bot.message_handler(commands=['crons'])
def list_crons(message):
    """List all active cron jobs."""
    save_chat_metadata(message)
    if not authorize(message):
        return
    save_chat_id(message.chat.id)
    append_chat_history("user", "/crons", chat_id=message.chat.id)
    
    with _cron_lock:
        if not _cron_jobs:
            response = "No active cron jobs."
        else:
            lines = ["Active cron jobs:"]
            for job in _cron_jobs:
                interval = job['interval_seconds']
                if interval < 60:
                    interval_display = f"{interval}sec"
                elif interval < 3600:
                    interval_display = f"{interval // 60}min"
                else:
                    interval_display = f"{interval // 3600}h"
                lines.append(f"  #{job['id']}: every {interval_display} - {job['prompt'][:50]}...")
            response = "\n".join(lines)
    
    bot.reply_to(message, response)
    append_chat_history("assistant", response, chat_id=message.chat.id)


@bot.message_handler(commands=['uncron'])
def remove_cron(message):
    """Remove a cron job by ID."""
    save_chat_metadata(message)
    if not authorize(message):
        return
    save_chat_id(message.chat.id)
    
    text = message.text.replace('/uncron', '', 1).strip()
    append_chat_history("user", f"/uncron {text}", chat_id=message.chat.id)
    
    if not text:
        response = "Usage: /uncron <id>\nUse /crons to see active jobs."
        bot.reply_to(message, response)
        append_chat_history("assistant", response, chat_id=message.chat.id)
        return
    
    try:
        cron_id = int(text.replace('#', ''))
    except ValueError:
        response = "Invalid cron ID. Use /crons to see active jobs."
        bot.reply_to(message, response)
        append_chat_history("assistant", response, chat_id=message.chat.id)
        return
    
    with _cron_lock:
        for i, job in enumerate(_cron_jobs):
            if job['id'] == cron_id:
                _cron_jobs.pop(i)
                save_crons()
                response = f"✅ Cron #{cron_id} removed."
                bot.reply_to(message, response)
                append_chat_history("assistant", response, chat_id=message.chat.id)
                return
    
    response = f"Cron #{cron_id} not found. Use /crons to see active jobs."
    bot.reply_to(message, response)
    append_chat_history("assistant", response, chat_id=message.chat.id)


@bot.message_handler(commands=['start'])
def send_welcome(message):
    save_chat_metadata(message)
    if not authorize(message):
        return
    save_chat_id(message.chat.id)
    append_chat_history("user", f"/start")
    
    if DEBUG_MODE:
        # In debug mode, show the full command list for testing
        response = "Hello! I'm Tau. Commands:\n/task <description> - Add a task\n/plan <description> - Create an execution plan\n/status - See recent activity\n/adapt <prompt> - Self-modify\n/cron <interval> <prompt> - Schedule recurring prompt\n/crons - List active crons\n/uncron <id> - Remove a cron\n/clear - Stop active agent processes\n/restart - Restart bot\n/kill - Stop the bot\n/debug - Toggle debug mode"
    else:
        # Conversational welcome - no command list
        response = "Hey! I'm Tau. Just chat with me like you would a friend who happens to be good with computers.\n\nTry things like:\n• \"remind me to call mom at 5pm\"\n• \"what's the weather in Tokyo?\"\n• \"every morning, send me a motivational quote\"\n\nWhat's on your mind?"
    
    bot.reply_to(message, response)
    append_chat_history("assistant", response, chat_id=message.chat.id)


@bot.message_handler(commands=['debug'])
def toggle_debug(message):
    """Toggle debug mode on/off."""
    global DEBUG_MODE
    save_chat_metadata(message)
    if not authorize(message):
        return
    save_chat_id(message.chat.id)
    append_chat_history("user", "/debug", chat_id=message.chat.id)
    
    DEBUG_MODE = not DEBUG_MODE
    set_debug_mode(DEBUG_MODE)
    
    status = "on" if DEBUG_MODE else "off"
    response = f"Debug mode: {status}"
    bot.reply_to(message, response)
    append_chat_history("assistant", response, chat_id=message.chat.id)

def restart_via_supervisor():
    """Restart tau via supervisorctl. Returns True if supervisor handled it."""
    tauctl = os.path.join(WORKSPACE, "tauctl")
    if os.path.exists(tauctl):
        try:
            result = subprocess.run(
                [tauctl, "_agent_restart"],
                capture_output=True,
                text=True,
                timeout=5
            )
            # Exit code 0 means supervisor is handling restart
            # Exit code 2 means fallback to direct restart
            return result.returncode == 0
        except Exception:
            pass
    return False


def stop_via_supervisor():
    """Stop tau via supervisorctl. Returns True if supervisor handled it."""
    tauctl = os.path.join(WORKSPACE, "tauctl")
    if os.path.exists(tauctl):
        try:
            result = subprocess.run(
                [tauctl, "_agent_stop"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # Exit code 0 means supervisor is handling stop.
            return result.returncode == 0
        except Exception:
            pass
    return False


@bot.message_handler(commands=['clear'])
def clear_active_agent_processes(message):
    """Stop any in-flight agent subprocesses (e.g. stuck /ask, cron, agent loop)."""
    save_chat_metadata(message)
    if not authorize(message):
        return
    save_chat_id(message.chat.id)
    append_chat_history("user", "/clear", chat_id=message.chat.id)

    stopped = processes.terminate_all(label_prefix="agent:", timeout_seconds=2.0)
    if stopped:
        response = f"🧹 Stopped {len(stopped)} active agent process(es)."
    else:
        response = "No active agent processes."

    bot.reply_to(message, response)
    append_chat_history("assistant", response, chat_id=message.chat.id)


@bot.message_handler(commands=['kill'])
def kill_bot(message):
    """Fully stop the bot process (and attempt to stop supervisord-managed service)."""
    save_chat_metadata(message)
    if not authorize(message):
        return
    save_chat_id(message.chat.id)
    append_chat_history("user", "/kill", chat_id=message.chat.id)

    response = "🛑 Killing bot..."
    bot.reply_to(message, response)
    append_chat_history("assistant", response, chat_id=message.chat.id)

    # Stop any in-flight agent subprocesses first.
    try:
        processes.terminate_all(label_prefix="agent:", timeout_seconds=2.0)
    except Exception:
        pass

    # Stop background loops, then stop polling.
    try:
        _stop_event.set()
    except Exception:
        pass
    try:
        bot.stop_polling()
    except Exception:
        pass

    # If supervised, ask supervisor to stop the program so it won't auto-restart.
    if stop_via_supervisor():
        os._exit(0)

    # Fallback: hard-exit the current process.
    os._exit(0)


@bot.message_handler(commands=['restart'])
def restart_bot(message):
    """Restart the bot process via supervisor."""
    save_chat_metadata(message)
    if not authorize(message):
        return
    append_chat_history("user", f"/restart", chat_id=message.chat.id)
    response = "Restarting..."
    bot.reply_to(message, response)
    append_chat_history("assistant", response, chat_id=message.chat.id)
    bot.stop_polling()
    
    # Try supervisor restart first
    if restart_via_supervisor():
        # Supervisor will restart us - just exit cleanly
        sys.exit(0)
    else:
        # Fallback to direct exec restart
        os.execv(sys.executable, [sys.executable] + sys.argv)

def run_adapt_streaming(
    prompt: str,
    *,
    chat_id: int,
    reply_to_message_id: int | None = None,
    timeout_seconds: int = 3600,
) -> str:
    """Run the agent CLI for adaptation and stream thinking/progress to Telegram.
    
    Streams minor updates during the process, then sends a final summary of what was done.
    """
    stream = TelegramStreamingMessage(
        chat_id,
        reply_to_message_id=reply_to_message_id,
        initial_text="🫡 Starting...",
        min_edit_interval_seconds=1.5,  # Faster updates to show thinking
        min_chars_delta=15,  # More responsive updates
    )

    from .codex import build_cmd, parse_event, format_tool_update, AGENT_MODEL
    cmd = build_cmd(prompt, model=AGENT_MODEL, json_mode=True)

    start_time = time.time()
    thinking_updates: list[str] = []
    thinking_text_buffer: str = ""  # Buffer for streaming assistant text
    final_result_text: str | None = None
    stderr_lines: list[str] = []
    current_tool: str | None = None
    last_thinking_snippet: str = ""  # Track last shown thinking snippet

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            stdin=subprocess.DEVNULL,
            cwd=WORKSPACE,
            bufsize=1,
            start_new_session=True,
        )
        processes.track(
            proc,
            label=f"agent:adapt:{chat_id}",
            cmd=cmd,
            own_process_group=True,
        )
    except Exception as e:
        err = f"Error: {str(e)}"
        stream.set_text(err)
        return err

    sentinel = object()
    q: queue.Queue[object] = queue.Queue()

    def _stdout_reader():
        try:
            if proc.stdout is None:
                return
            for line in proc.stdout:
                q.put(line)
        finally:
            q.put(sentinel)

    reader_thread = threading.Thread(
        target=_stdout_reader,
        daemon=True,
        name="AgentAdaptStreamReader",
    )
    reader_thread.start()

    def _stderr_reader():
        try:
            if proc.stderr is None:
                return
            for line in proc.stderr:
                stderr_lines.append(line)
        except Exception:
            pass

    stderr_thread = threading.Thread(
        target=_stderr_reader,
        daemon=True,
        name="AgentAdaptStderrReader",
    )
    stderr_thread.start()

    timed_out = False
    
    # Cycling status messages for interactive feel
    adapting_phases = [
        "🫡 Adapting...",
        "📝 Planning changes...",
        "🔍 Analyzing code...",
        "🧠 Reasoning...",
        "⚙️ Working...",
        "✨ Making progress...",
    ]
    
    def build_display():
        """Build the display message showing current thinking and actions."""
        # Cycle through phases every 2 seconds
        elapsed = time.time() - start_time
        phase_idx = int(elapsed / 2) % len(adapting_phases)
        lines = [adapting_phases[phase_idx] + "\n"]
        
        # Show current thinking or the last complete snippet
        display_thinking = ""
        if thinking_text_buffer.strip():
            cleaned = thinking_text_buffer.strip().replace("\n", " ")
            # If we have a last complete snippet, use it as the base
            if last_thinking_snippet:
                display_thinking = last_thinking_snippet
                # If there's more text after the snippet, show a bit of it to show progress
                if len(cleaned) > len(last_thinking_snippet) + 5:
                    remaining = cleaned[len(last_thinking_snippet):].strip()
                    display_thinking += " " + remaining
            else:
                # No complete sentence yet, just show what we have
                display_thinking = cleaned
            
            # Truncate for display
            if len(display_thinking) > 150:
                display_thinking = "..." + display_thinking[-147:]
            
            lines.append(f"💭 {display_thinking}\n")
        
        # Show recent tool actions
        if thinking_updates:
            lines.append("")
            lines.extend(thinking_updates[-6:])
        
        return "\n".join(lines)

    # Track when we last updated the display (for periodic refresh)
    last_display_update = start_time
    
    try:
        while True:
            if time.time() - start_time > timeout_seconds:
                timed_out = True
                try:
                    import signal as _signal
                    os.killpg(proc.pid, _signal.SIGTERM)
                except Exception:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                try:
                    proc.wait(timeout=2)
                except Exception:
                    try:
                        import signal as _signal
                        os.killpg(proc.pid, _signal.SIGKILL)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                break

            try:
                item = q.get(timeout=0.25)
            except queue.Empty:
                item = None
                # Periodically refresh display even when no data (cycles adapting phases)
                now = time.time()
                if now - last_display_update >= 1.5:
                    stream.set_text(build_display())
                    last_display_update = now

            if item is sentinel:
                break

            if isinstance(item, str) and item:
                line = item.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                
                parsed = parse_event(event)
                if parsed is None:
                    continue

                kind = parsed["kind"]

                # Handle tool start events - show what the agent is doing
                if kind == "tool_start":
                    current_tool = parsed.get("tool_name")
                    update = format_tool_update(parsed)
                    if update:
                        thinking_updates.append(update)
                        stream.set_text(build_display())
                        last_display_update = time.time()

                # Handle tool done events
                elif kind == "tool_done":
                    if current_tool:
                        if thinking_updates:
                            thinking_updates[-1] = thinking_updates[-1].replace("...", " ✓")
                            stream.set_text(build_display())
                            last_display_update = time.time()
                        current_tool = None

                # Handle message text (agent reasoning / response)
                elif kind in ("message", "message_done"):
                    text = parsed.get("text", "")
                    if text:
                        # For message_done, text is the full message; replace buffer
                        if kind == "message_done":
                            thinking_text_buffer = text
                        else:
                            # For incremental updates, only append if text is longer
                            if len(text) > len(thinking_text_buffer):
                                thinking_text_buffer = text

                        cleaned = thinking_text_buffer.strip().replace("\n", " ")

                        last_sentence_end = -1
                        for i in range(len(cleaned) - 1, -1, -1):
                            if cleaned[i] in '.!?' and (i == len(cleaned) - 1 or cleaned[i + 1] in ' \n\t'):
                                last_sentence_end = i
                                break

                        if last_sentence_end > 0:
                            complete_text = cleaned[:last_sentence_end + 1]
                            if len(complete_text) > 100:
                                snippet_start = len(complete_text) - 100
                                for i in range(snippet_start, len(complete_text)):
                                    if complete_text[i] in '.!?' and i + 1 < len(complete_text):
                                        snippet_start = i + 2
                                        break
                                snippet = complete_text[snippet_start:].strip()
                            else:
                                snippet = complete_text

                            if snippet and snippet != last_thinking_snippet:
                                last_thinking_snippet = snippet
                                stream.set_text(build_display())
                                last_display_update = time.time()

                        if len(thinking_text_buffer) > 50:
                            final_result_text = thinking_text_buffer

                elif kind == "turn_done":
                    # Turn finished — final_result_text should already be set
                    pass

                elif kind == "error":
                    err_text = parsed.get("text", "Agent error")
                    if not final_result_text:
                        final_result_text = f"Error: {err_text}"

            if proc.poll() is not None and q.empty():
                continue
    finally:
        stream.finalize()
        processes.untrack(proc)

    if processes.pop_cancelled(proc.pid):
        msg = "❌ Cancelled."
        stream.set_text(msg)
        return msg

    if timed_out:
        msg = "⏰ Adaptation timed out."
        stream.set_text(msg)
        return msg

    # Build final summary describing what was done
    # Prefer the agent's own explanation if available
    if final_result_text:
        # Clean up and truncate the result
        summary = final_result_text.strip()
        # Remove markdown code blocks for cleaner display
        summary = re.sub(r'```[\s\S]*?```', '[code]', summary)
        if len(summary) > 600:
            # Find a good cut point at sentence boundary
            cut = summary[:600].rfind('. ')
            if cut > 300:
                summary = summary[:cut+1]
            else:
                cut = summary[:600].rfind('\n')
                if cut > 300:
                    summary = summary[:cut]
                else:
                    summary = summary[:597] + "..."
        final_msg = f"✅ Done\n\n{summary}"
    else:
        # Fallback: Use the tool operations as summary
        ops = [u.replace("...", "").replace(" ✓", "") for u in thinking_updates]
        if ops:
            final_msg = "✅ Done\n\nChanges made:\n" + "\n".join(ops[-10:])
        else:
            final_msg = "✅ Done"
    
    stream.set_text(final_msg)
    return final_result_text or "Adaptation complete"


@bot.message_handler(commands=['adapt'])
def adapt_bot(message):
    """Self-modify the bot using Codex agent, then restart."""
    save_chat_metadata(message)
    if not authorize(message):
        return
    # Extract prompt after /adapt
    prompt = message.text.replace('/adapt', '', 1).strip()
    if not prompt:
        response = "Usage: /adapt <prompt>"
        bot.reply_to(message, response)
        append_chat_history("user", f"/adapt", chat_id=message.chat.id)
        append_chat_history("assistant", response, chat_id=message.chat.id)
        return
    
    append_chat_history("user", f"/adapt {prompt}", chat_id=message.chat.id)
    save_chat_id(message.chat.id)
    
    try:
        # Run adaptation with streaming progress
        result = run_adapt_streaming(
            prompt,
            chat_id=message.chat.id,
            reply_to_message_id=message.message_id,
            timeout_seconds=3600,  # 1 hour timeout
        )
        
        append_chat_history("assistant", f"✅ Adaptation complete: {result[:200]}", chat_id=message.chat.id)
        
        # Commit any changes made by the agent
        git_commit_changes(prompt)
        
        bot.stop_polling()
        
        # Try supervisor restart first
        if restart_via_supervisor():
            sys.exit(0)
        else:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        
    except Exception as e:
        error_msg = f"Adaptation error: {str(e)}"
        bot.reply_to(message, error_msg)
        append_chat_history("assistant", error_msg, chat_id=message.chat.id)


PLANS_DIR = os.path.join(WORKSPACE, "context", "plans")


def run_plan_generation(
    task_description: str,
    *,
    chat_id: int,
    reply_to_message_id: int | None = None,
    timeout_seconds: int = 600,
) -> tuple[str, str]:
    """Generate a plan for a task using the agent.
    
    Returns:
        Tuple of (plan_content, plan_filename)
    """
    stream = TelegramStreamingMessage(
        chat_id,
        reply_to_message_id=reply_to_message_id,
        initial_text="📋 Creating plan...",
        min_edit_interval_seconds=1.5,
        min_chars_delta=15,
    )

    # Generate a filename-safe slug from the task description
    import re as _re
    slug = _re.sub(r'[^a-z0-9]+', '-', task_description.lower())[:50].strip('-')
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    plan_filename = f"plan-{timestamp}-{slug}.md"
    
    # Build the planning prompt
    plan_prompt = f"""Create a detailed execution plan for the following task:

TASK: {task_description}

Generate a comprehensive plan that includes:
1. **Goal**: Clear statement of what needs to be accomplished
2. **Prerequisites**: What needs to be in place before starting
3. **Steps**: Numbered action items with specific, actionable instructions
4. **Success Criteria**: How to verify the task is complete
5. **Potential Issues**: Risks or blockers to watch for

Format the plan as a clean markdown document. Be specific and actionable.
The plan should be self-contained so someone can follow it without additional context.

Output ONLY the plan content, no preamble or meta-commentary."""

    from .codex import build_cmd, parse_event, CHAT_MODEL
    cmd = build_cmd(plan_prompt, model=CHAT_MODEL, readonly=True, json_mode=True)

    start_time = time.time()
    thinking_updates: list[str] = []
    thinking_text_buffer: str = ""
    final_result_text: str | None = None
    stderr_lines: list[str] = []
    last_thinking_snippet: str = ""

    planning_phases = [
        "📋 Creating plan...",
        "🎯 Defining goals...",
        "📝 Outlining steps...",
        "🔍 Adding details...",
        "✨ Finalizing...",
    ]

    def build_display():
        elapsed = time.time() - start_time
        phase_idx = int(elapsed / 2) % len(planning_phases)
        lines = [planning_phases[phase_idx] + "\n"]
        
        if thinking_text_buffer.strip():
            cleaned = thinking_text_buffer.strip().replace("\n", " ")
            display_thinking = last_thinking_snippet if last_thinking_snippet else cleaned
            if len(display_thinking) > 150:
                display_thinking = "..." + display_thinking[-147:]
            lines.append(f"💭 {display_thinking}\n")
        
        if thinking_updates:
            lines.append("")
            lines.extend(thinking_updates[-6:])
        
        return "\n".join(lines)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            stdin=subprocess.DEVNULL,
            cwd=WORKSPACE,
            bufsize=1,
            start_new_session=True,
        )
        processes.track(
            proc,
            label=f"agent:plan:{chat_id}",
            cmd=cmd,
            own_process_group=True,
        )
    except Exception as e:
        err = f"Error: {str(e)}"
        stream.set_text(err)
        return err, ""

    sentinel = object()
    q: queue.Queue[object] = queue.Queue()

    def _stdout_reader():
        try:
            if proc.stdout is None:
                return
            for line in proc.stdout:
                q.put(line)
        finally:
            q.put(sentinel)

    reader_thread = threading.Thread(
        target=_stdout_reader,
        daemon=True,
        name="AgentPlanStreamReader",
    )
    reader_thread.start()

    def _stderr_reader():
        try:
            if proc.stderr is None:
                return
            for line in proc.stderr:
                stderr_lines.append(line)
        except Exception:
            pass

    stderr_thread = threading.Thread(
        target=_stderr_reader,
        daemon=True,
        name="AgentPlanStderrReader",
    )
    stderr_thread.start()

    timed_out = False
    last_display_update = start_time

    try:
        while True:
            if time.time() - start_time > timeout_seconds:
                timed_out = True
                try:
                    import signal as _signal
                    os.killpg(proc.pid, _signal.SIGTERM)
                except Exception:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                try:
                    proc.wait(timeout=2)
                except Exception:
                    try:
                        import signal as _signal
                        os.killpg(proc.pid, _signal.SIGKILL)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                break

            try:
                item = q.get(timeout=0.25)
            except queue.Empty:
                item = None
                now = time.time()
                if now - last_display_update >= 1.5:
                    stream.set_text(build_display())
                    last_display_update = now

            if item is sentinel:
                break

            if isinstance(item, str) and item:
                line = item.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    continue

                parsed = parse_event(event)
                if parsed is None:
                    continue

                kind = parsed["kind"]

                if kind in ("message", "message_done"):
                    text = parsed.get("text", "")
                    if text:
                        if kind == "message_done":
                            thinking_text_buffer = text
                        elif len(text) > len(thinking_text_buffer):
                            thinking_text_buffer = text

                        cleaned = thinking_text_buffer.strip().replace("\n", " ")
                        last_sentence_end = -1
                        for i in range(len(cleaned) - 1, -1, -1):
                            if cleaned[i] in '.!?' and (i == len(cleaned) - 1 or cleaned[i + 1] in ' \n\t'):
                                last_sentence_end = i
                                break

                        if last_sentence_end > 0:
                            complete_text = cleaned[:last_sentence_end + 1]
                            if len(complete_text) > 100:
                                snippet_start = len(complete_text) - 100
                                for i in range(snippet_start, len(complete_text)):
                                    if complete_text[i] in '.!?' and i + 1 < len(complete_text):
                                        snippet_start = i + 2
                                        break
                                snippet = complete_text[snippet_start:].strip()
                            else:
                                snippet = complete_text

                            if snippet and snippet != last_thinking_snippet:
                                last_thinking_snippet = snippet
                                stream.set_text(build_display())
                                last_display_update = time.time()

                        if len(thinking_text_buffer) > 50:
                            final_result_text = thinking_text_buffer

                elif kind == "error":
                    err_text = parsed.get("text", "Plan generation error")
                    if not final_result_text:
                        final_result_text = f"Error: {err_text}"

            if proc.poll() is not None and q.empty():
                continue
    finally:
        stream.finalize()
        processes.untrack(proc)

    if processes.pop_cancelled(proc.pid):
        msg = "❌ Cancelled."
        stream.set_text(msg)
        return msg, ""

    if timed_out:
        msg = "⏰ Plan generation timed out."
        stream.set_text(msg)
        return msg, ""

    # Get the plan content
    plan_content = (final_result_text or thinking_text_buffer or "").strip()
    
    if not plan_content:
        stderr_text = "".join(stderr_lines).strip()
        msg = stderr_text if stderr_text else "Failed to generate plan."
        stream.set_text(msg)
        return msg, ""

    # Save the plan to a file
    os.makedirs(PLANS_DIR, exist_ok=True)
    plan_path = os.path.join(PLANS_DIR, plan_filename)
    
    # Add header to the plan
    full_plan = f"""# Plan: {task_description}

*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

---

{plan_content}
"""
    
    with open(plan_path, "w") as f:
        f.write(full_plan)
    
    # Show success message with plan preview
    preview = plan_content[:500] + "..." if len(plan_content) > 500 else plan_content
    success_msg = f"✅ Plan saved to `{plan_filename}`\n\n{preview}"
    stream.set_text(success_msg)
    
    return full_plan, plan_filename


@bot.message_handler(commands=['plan'])
def create_plan(message):
    """Create a plan for a task and save it to a file."""
    save_chat_metadata(message)
    if not authorize(message):
        return
    save_chat_id(message.chat.id)
    
    task_text = message.text.replace('/plan', '', 1).strip()
    append_chat_history("user", f"/plan {task_text}", chat_id=message.chat.id)
    
    if not task_text:
        response = "Usage: /plan <task description>\n\nExample: /plan implement user authentication with OAuth2"
        bot.reply_to(message, response)
        append_chat_history("assistant", response, chat_id=message.chat.id)
        return
    
    try:
        plan_content, plan_filename = run_plan_generation(
            task_text,
            chat_id=message.chat.id,
            reply_to_message_id=message.message_id,
            timeout_seconds=600,
        )
        
        if plan_filename:
            append_chat_history("assistant", f"✅ Plan created: {plan_filename}", chat_id=message.chat.id)
        else:
            append_chat_history("assistant", f"Plan generation failed: {plan_content[:200]}", chat_id=message.chat.id)
            
    except Exception as e:
        error_msg = f"Error creating plan: {str(e)}"
        bot.reply_to(message, error_msg)
        append_chat_history("assistant", error_msg, chat_id=message.chat.id)


@bot.message_handler(commands=['task'])
def add_task(message):
    """Add a task to its own directory."""
    save_chat_metadata(message)
    if not authorize(message):
        return
    save_chat_id(message.chat.id)
    
    task_text = message.text.replace('/task', '', 1).strip()
    append_chat_history("user", f"/task {task_text}", chat_id=message.chat.id)
    
    if not task_text:
        response = "Usage: /task <description>"
        bot.reply_to(message, response)
        append_chat_history("assistant", response, chat_id=message.chat.id)
        return
    
    # Ensure tasks directory exists
    os.makedirs(TASKS_DIR, exist_ok=True)
    
    # Find next available task ID
    existing_tasks = get_all_tasks()
    next_id = len(existing_tasks) + 1
    task_id = f"task-{next_id}"
    
    # Create task directory
    task_dir = Path(TASKS_DIR) / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Create task.md
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    task_content = f"## {task_text}\n- created: {timestamp}\n- source: telegram\n"
    task_file = task_dir / "task.md"
    task_file.write_text(task_content)
    
    # Initialize memory.md
    memory_file = task_dir / "memory.md"
    if not memory_file.exists():
        memory_file.write_text("# Task Memory\n\n<!-- Detailed memory for this task -->\n")
    
    response = f"✅ Task added ({task_id}): {task_text}"
    bot.reply_to(message, response)
    append_chat_history("assistant", response, chat_id=message.chat.id)


@bot.message_handler(commands=['status'])
def get_status(message):
    """Show recent high-level memory and task status."""
    save_chat_metadata(message)
    if not authorize(message):
        return
    save_chat_id(message.chat.id)
    append_chat_history("user", "/status", chat_id=message.chat.id)
    
    # Get high-level memory
    high_level = ""
    if os.path.exists(MEMORY_FILE):
        content = open(MEMORY_FILE).read()
        lines = content.strip().split("\n")
        high_level = "\n".join(lines[-20:])  # Last 20 lines
    
    # Get task status
    tasks = get_all_tasks()
    incomplete = [t for t in tasks if not os.path.exists(t["memory_file"]) or 
                  "complete" not in t["memory_file"].read_text().lower()[-500:]]
    
    status_msg = "📊 Status\n\n"
    
    if tasks:
        status_msg += f"Total tasks: {len(tasks)}\n"
        status_msg += f"Incomplete: {len(incomplete)}\n\n"
        
        if incomplete:
            status_msg += "Active tasks:\n"
            for task in incomplete[:5]:  # Show first 5
                status_msg += f"  • {task['title'][:50]}\n"
    
    if high_level.strip():
        status_msg += f"\n📝 Recent activity:\n\n{high_level[:3000]}"
    
    if not status_msg.strip() or status_msg == "📊 Status\n\n":
        response = "No tasks or memory yet."
        bot.reply_to(message, response)
        append_chat_history("assistant", response, chat_id=message.chat.id)
    else:
        response = status_msg[:4000]
        bot.reply_to(message, response)
        append_chat_history("assistant", response, chat_id=message.chat.id)


@bot.message_handler(content_types=['voice'])
def handle_voice_message(message):
    """Handle voice messages by transcribing and processing as text."""
    save_chat_metadata(message)
    if not authorize(message):
        return
    save_chat_id(message.chat.id)
    
    # Notify user that we're processing the voice message (we'll edit this message with the answer)
    processing_msg = bot.reply_to(message, "🎤 Processing voice message...")
    
    voice_path = None
    try:
        # Download the voice file
        voice_path = download_voice_file(message.voice.file_id)
        
        # Transcribe using OpenAI
        transcribed_text = transcribe_voice(voice_path)
        
        if not transcribed_text.strip():
            response = "Could not transcribe voice message. Please try again."
            try:
                bot.edit_message_text(response, message.chat.id, processing_msg.message_id)
            except Exception:
                bot.reply_to(message, response)
            append_chat_history("user", "[voice message - transcription failed]", chat_id=message.chat.id)
            append_chat_history("assistant", response, chat_id=message.chat.id)
            return
        
        # Append transcribed text to chat history with voice indicator
        append_chat_history("user", f"[voice]: {transcribed_text}", chat_id=message.chat.id)
        
        # Get chat history BEFORE processing (so current message only appears once)
        from .telegram import get_chat_history
        chat_history = get_chat_history()
        
        # Get chat summary for broader context
        chat_summary = ""
        if os.path.exists(CHAT_SUMMARY_FILE):
            with open(CHAT_SUMMARY_FILE) as f:
                chat_summary = f.read().strip()
        
        # Build prompt with chat history context
        prompt_with_context = f"""CONVERSATION SUMMARY (auto-updated hourly):
{chat_summary if chat_summary else "No summary available yet."}

TELEGRAM CHAT HISTORY (recent):
{chat_history}

CURRENT USER MESSAGE (transcribed from voice):
{transcribed_text}

Please respond to the user's message above, considering the full context of our conversation history."""
        
        # Start typing indicator in background
        typing_stop = threading.Event()
        typing_thread = threading.Thread(
            target=send_typing_action,
            args=(message.chat.id, typing_stop),
            daemon=True
        )
        typing_thread.start()
        
        # Process transcribed text as normal message
        try:
            from .codex import CHAT_MODEL
            backend = (os.getenv("TAU_CHAT_BACKEND") or "codex").strip().lower()
            openai_model = os.getenv("TAU_OPENAI_CHAT_MODEL", "gpt-4o-mini")
            codex_model = os.getenv("TAU_CODEX_CHAT_MODEL", CHAT_MODEL)
            use_openai = backend in ("openai", "oa") or (backend == "auto" and openai_client is not None)

            if use_openai:
                response = run_openai_chat_streaming(
                    prompt_with_context,
                    chat_id=message.chat.id,
                    existing_message_id=processing_msg.message_id,
                    initial_text="🎤 Thinking...",
                    model=openai_model,
                    timeout_seconds=60,
                    max_tokens=500,
                )
            else:
                response = run_agent_ask_streaming(
                    prompt_with_context,
                    chat_id=message.chat.id,
                    existing_message_id=processing_msg.message_id,
                    initial_text="🎤 Thinking...",
                    model=codex_model,
                    timeout_seconds=600,
                )
            append_chat_history("assistant", response, chat_id=message.chat.id)
        finally:
            # Stop typing indicator
            typing_stop.set()
        
    except Exception as e:
        error_msg = f"Error processing voice message: {str(e)}"
        try:
            bot.edit_message_text(error_msg, message.chat.id, processing_msg.message_id)
        except Exception:
            bot.reply_to(message, error_msg)
        append_chat_history("assistant", error_msg, chat_id=message.chat.id)
    finally:
        # Clean up temporary voice file
        if voice_path and os.path.exists(voice_path):
            try:
                os.unlink(voice_path)
            except Exception:
                pass


@bot.message_handler(content_types=['photo', 'document', 'sticker', 'video', 'audio', 'animation', 'video_note', 'contact', 'location', 'venue', 'poll'])
def handle_other_content(message):
    """Handle non-text content types with a confirmation."""
    save_chat_metadata(message)
    if not authorize(message):
        return
    save_chat_id(message.chat.id)
    
    # Determine content type for the confirmation message
    content_type = message.content_type
    response = f"📨 Received {content_type}. I can process text and voice messages."
    bot.reply_to(message, response)
    append_chat_history("user", f"[{content_type}]", chat_id=message.chat.id)
    append_chat_history("assistant", response, chat_id=message.chat.id)


def send_typing_action(chat_id, stop_event):
    """Send typing action every 4 seconds until stop_event is set."""
    while not stop_event.is_set():
        try:
            bot.send_chat_action(chat_id, 'typing')
        except Exception:
            pass
        # Wait 4 seconds (typing indicator lasts ~5 seconds)
        stop_event.wait(4)


def run_agent_ask_streaming(
    prompt_with_context: str,
    *,
    chat_id: int,
    reply_to_message_id: int | None = None,
    existing_message_id: int | None = None,
    initial_text: str = "🤔 Thinking...",
    model: str | None = None,
    timeout_seconds: int = 600,
) -> str:
    """Run the Codex CLI and stream JSONL output by editing one Telegram message.
    
    Shows thinking process and tool operations with the same structure as /adapt:
    - Start with thinking emoji
    - Stream thought bubble with thinking
    - End with just the final answer
    """
    if model is None:
        from .codex import CHAT_MODEL
        model = CHAT_MODEL
    stream = TelegramStreamingMessage(
        chat_id,
        reply_to_message_id=reply_to_message_id,
        existing_message_id=existing_message_id,
        initial_text=initial_text,
        min_edit_interval_seconds=1.5,  # Faster updates for thinking
        min_chars_delta=15,  # More responsive updates
    )

    from .codex import build_cmd, parse_event, format_tool_update
    cmd = build_cmd(prompt_with_context, model=model, readonly=True, json_mode=True)

    start_time = time.time()
    raw_output = ""
    final_result_text: str | None = None
    stderr_lines: list[str] = []
    thinking_updates: list[str] = []
    thinking_text_buffer: str = ""
    current_tool: str | None = None
    last_thinking_snippet: str = ""
    
    # Cycling status messages for interactive feel
    thinking_phases = [
        "🤔 Thinking...",
        "📚 Reading context...",
        "🔍 Analyzing...",
        "💭 Processing...",
        "🧠 Reasoning...",
        "✨ Forming response...",
    ]
    
    def build_display():
        """Build the display message showing current thinking and actions.
        
        Uses same structure as /adapt:
        - 🤔 Thinking... (cycles through phases)
        - 💭 {thinking snippet}
        - {tool actions}
        """
        # Cycle through phases every 2 seconds
        elapsed = time.time() - start_time
        phase_idx = int(elapsed / 2) % len(thinking_phases)
        lines = [thinking_phases[phase_idx] + "\n"]
        
        # Show current thinking or the last complete snippet
        display_thinking = ""
        if thinking_text_buffer.strip():
            cleaned = thinking_text_buffer.strip().replace("\n", " ")
            # If we have a last complete snippet, use it as the base
            if last_thinking_snippet:
                display_thinking = last_thinking_snippet
                # If there's more text after the snippet, show a bit of it to show progress
                if len(cleaned) > len(last_thinking_snippet) + 5:
                    remaining = cleaned[len(last_thinking_snippet):].strip()
                    display_thinking += " " + remaining
            else:
                # No complete sentence yet, just show what we have
                display_thinking = cleaned
            
            # Truncate for display
            if len(display_thinking) > 150:
                display_thinking = "..." + display_thinking[-147:]
            
            lines.append(f"💭 {display_thinking}\n")
        
        # Show recent tool actions
        if thinking_updates:
            lines.append("")
            lines.extend(thinking_updates[-6:])
        
        return "\n".join(lines)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            stdin=subprocess.DEVNULL,
            cwd=WORKSPACE,
            bufsize=1,
            start_new_session=True,
        )
        processes.track(
            proc,
            label=f"agent:ask:{chat_id}",
            cmd=cmd,
            own_process_group=True,
        )
    except Exception as e:
        err = f"Error: {str(e)}"
        stream.set_text(err)
        return err

    sentinel = object()
    q: queue.Queue[object] = queue.Queue()

    def _stdout_reader():
        try:
            if proc.stdout is None:
                return
            for line in proc.stdout:
                q.put(line)
        finally:
            q.put(sentinel)

    reader_thread = threading.Thread(
        target=_stdout_reader,
        daemon=True,
        name="AgentAskStreamReader",
    )
    reader_thread.start()

    def _stderr_reader():
        try:
            if proc.stderr is None:
                return
            for line in proc.stderr:
                stderr_lines.append(line)
        except Exception:
            pass

    stderr_thread = threading.Thread(
        target=_stderr_reader,
        daemon=True,
        name="AgentAskStderrReader",
    )
    stderr_thread.start()

    timed_out = False

    # Track when we last updated the display (for periodic refresh)
    last_display_update = start_time
    
    try:
        while True:
            if time.time() - start_time > timeout_seconds:
                timed_out = True
                # Kill the whole process group (agent can spawn children).
                try:
                    import signal as _signal
                    os.killpg(proc.pid, _signal.SIGTERM)
                except Exception:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                try:
                    proc.wait(timeout=2)
                except Exception:
                    try:
                        import signal as _signal
                        os.killpg(proc.pid, _signal.SIGKILL)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                break

            try:
                item = q.get(timeout=0.25)
            except queue.Empty:
                item = None
                # Periodically refresh display even when no data (cycles thinking phases)
                now = time.time()
                if now - last_display_update >= 1.5:
                    stream.set_text(build_display())
                    last_display_update = now

            if item is sentinel:
                break

            if isinstance(item, str) and item:
                line = item.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    # Ignore non-JSON noise on stdout.
                    continue
                
                parsed = parse_event(event)
                if parsed is None:
                    continue

                kind = parsed["kind"]

                # Handle tool start events - show what the agent is doing
                if kind == "tool_start":
                    current_tool = parsed.get("tool_name")
                    update = format_tool_update(parsed)
                    if update:
                        thinking_updates.append(update)
                        stream.set_text(build_display())
                        last_display_update = time.time()

                # Handle tool done events
                elif kind == "tool_done":
                    if current_tool:
                        if thinking_updates:
                            thinking_updates[-1] = thinking_updates[-1].replace("...", " ✓")
                            stream.set_text(build_display())
                            last_display_update = time.time()
                        current_tool = None

                # Handle message text (agent reasoning / response)
                elif kind in ("message", "message_done"):
                    text = parsed.get("text", "")
                    if text:
                        if kind == "message_done":
                            thinking_text_buffer = text
                        elif len(text) > len(thinking_text_buffer):
                            thinking_text_buffer = text
                        raw_output = thinking_text_buffer

                        cleaned = thinking_text_buffer.strip().replace("\n", " ")

                        last_sentence_end = -1
                        for i in range(len(cleaned) - 1, -1, -1):
                            if cleaned[i] in '.!?' and (i == len(cleaned) - 1 or cleaned[i + 1] in ' \n\t'):
                                last_sentence_end = i
                                break

                        if last_sentence_end > 0:
                            complete_text = cleaned[:last_sentence_end + 1]
                            if len(complete_text) > 100:
                                snippet_start = len(complete_text) - 100
                                for i in range(snippet_start, len(complete_text)):
                                    if complete_text[i] in '.!?' and i + 1 < len(complete_text):
                                        snippet_start = i + 2
                                        break
                                snippet = complete_text[snippet_start:].strip()
                            else:
                                snippet = complete_text

                            if snippet and snippet != last_thinking_snippet:
                                last_thinking_snippet = snippet

                        stream.set_text(build_display())
                        last_display_update = time.time()

                elif kind == "turn_done":
                    if thinking_text_buffer:
                        final_result_text = thinking_text_buffer

                elif kind == "error":
                    err_text = parsed.get("text", "Agent error")
                    if not final_result_text:
                        final_result_text = f"Error: {err_text}"

            # If process exited, loop briefly to drain remaining output.
            if proc.poll() is not None and q.empty():
                continue
    finally:
        stream.finalize()
        processes.untrack(proc)

    if processes.pop_cancelled(proc.pid):
        msg = "❌ Cancelled."
        stream.set_text(msg)
        return msg

    if timed_out:
        msg = "⏰ Request timed out."
        stream.set_text(msg)
        return msg

    # Determine the output to use
    # Prefer final_result_text from "result" event if available, otherwise use raw_output
    output = (final_result_text or raw_output or "").strip()

    if not output:
        stderr_text = "".join(stderr_lines).strip()
        msg = stderr_text if stderr_text else "No response from agent."
        stream.set_text(msg)
        return msg

    # Extract just the final answer, removing all thinking/processing text
    final_answer = _extract_final_answer(output)
    
    # Final display: just the clean response (no checkmark prefix for cleaner look)
    stream.set_text(final_answer)
    return final_answer


def run_openai_chat_streaming(
    user_prompt: str,
    *,
    chat_id: int,
    reply_to_message_id: int | None = None,
    existing_message_id: int | None = None,
    initial_text: str = "🤔 Thinking...",
    model: str = "gpt-4o-mini",
    timeout_seconds: int = 45,
    max_tokens: int = 400,
    temperature: float = 0.2,
) -> str:
    """Stream a fast OpenAI chat completion into Telegram."""
    stream = TelegramStreamingMessage(
        chat_id,
        reply_to_message_id=reply_to_message_id,
        existing_message_id=existing_message_id,
        initial_text=initial_text,
        min_edit_interval_seconds=0.9,
        min_chars_delta=20,
    )

    if not openai_client:
        msg = "⚠️ OPENAI_API_KEY not set, fast chat unavailable."
        stream.set_text(msg)
        return msg

    start_time = time.time()
    parts: list[str] = []

    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Tau, an autonomous AI assistant running as a Telegram bot. "
                        "Answer the user directly and concisely. No preamble. No tool logs. No hidden reasoning. "
                        "When asked how to use you, explain conversationally: "
                        "users can just chat naturally, ask you to remind them of things, "
                        "set up recurring tasks, work on longer research in the background, "
                        "and even ask you to modify your own code. Give natural examples, not command syntax."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            timeout=timeout_seconds,
        )

        for chunk in resp:
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError("OpenAI request timed out")

            delta = None
            try:
                delta = chunk.choices[0].delta.content  # type: ignore[attr-defined]
            except Exception:
                delta = None

            if delta:
                parts.append(delta)
                stream.append(delta)

        stream.finalize()
        final = "".join(parts).strip()
        if not final:
            final = "No response."
            stream.set_text(final)
        return final

    except Exception as e:
        msg = f"Error: {str(e)}"
        stream.set_text(msg)
        return msg


def _extract_final_answer(output: str) -> str:
    """Extract just the final answer from agent output, removing thinking text.
    
    The agent may output:
    - Thinking process ("Let me count...", "Looking at...")
    - Tool usage descriptions
    - Multi-paragraph explanations
    - The actual answer
    
    We want only the actual answer.
    """
    output = output.strip()
    
    # Patterns that indicate thinking/processing (case insensitive)
    thinking_patterns = [
        r"^let me\b", r"^i'll\b", r"^i will\b", r"^counting\b", r"^looking at\b",
        r"^let's\b", r"^thinking\b", r"^checking\b", r"^analyzing\b", r"^first,?\b",
        r"^to answer\b", r"^the letters\b", r"^i can see\b", r"^i've\b", r"^i have\b",
        r"^\d+\.\s+\*\*", r"^here'?s?\b", r"^based on\b", r"^now\b", r"^okay\b",
        r"^i understand\b", r"^i see\b", r"^looked at\b", r"^found\b"
    ]
    
    # Patterns that indicate closing filler to remove
    closing_patterns = [
        r"\s*is there anything else.*\??\s*$",
        r"\s*let me know if.*$",
        r"\s*feel free to ask.*$",
        r"\s*would you like.*\??\s*$",
        r"\s*anything else.*\??\s*$",
        r"\s*how can i help you today\??\s*$",
        r"\s*how can i assist you further\??\s*$",
    ]
    
    # Split into paragraphs
    paragraphs = [p.strip() for p in output.split('\n\n') if p.strip()]
    
    if not paragraphs:
        return output
    
    # Filter out thinking paragraphs from the start
    result_paragraphs = []
    started_real_content = False
    
    for para in paragraphs:
        para_lower = para.lower()
        
        # Skip paragraphs that are clearly thinking/process
        is_thinking = False
        if not started_real_content:
            for pattern in thinking_patterns:
                if re.match(pattern, para_lower):
                    is_thinking = True
                    break
        
        if not is_thinking:
            started_real_content = True
            result_paragraphs.append(para)
    
    # If we filtered everything, use the original
    if not result_paragraphs:
        result_paragraphs = paragraphs
    
    # Remove closing filler from last paragraph
    if result_paragraphs:
        last = result_paragraphs[-1]
        for pattern in closing_patterns:
            last = re.sub(pattern, '', last, flags=re.IGNORECASE).strip()
        result_paragraphs[-1] = last
        
        # If last paragraph is now empty, remove it
        if not result_paragraphs[-1]:
            result_paragraphs = result_paragraphs[:-1]
    
    final = '\n\n'.join(result_paragraphs).strip()
    
    # Final cleanup: remove any remaining artifacts
    # Remove markdown code fence artifacts that might have bled through
    final = re.sub(r'^```[\s\S]*?```\s*', '', final).strip()
    # Remove leftover checkmarks or emojis at start
    final = re.sub(r'^[✅🤔💭🫡✅]\s*', '', final).strip()
    # Remove fragments like "` and identifies..." or "3. Preventing..."
    final = re.sub(r'^`?\s*and identifies.*?\.\s*', '', final, flags=re.IGNORECASE | re.DOTALL).strip()
    final = re.sub(r'^\d+\.\s+Preventing Duplication.*?\.\s*', '', final, flags=re.IGNORECASE | re.DOTALL).strip()
    
    return final if final else output


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    """Handle all non-command text messages.

    Security model:
    - Always log the message to per-chat history (group observer).
    - Only generate a response and invoke the agent for the owner.
    - Non-owner messages in groups are silently observed.
    - Non-owner private messages are silently ignored.
    """
    logger.info(f"=== MESSAGE RECEIVED: '{message.text[:50]}...' ===" if len(message.text or '') > 50 else f"=== MESSAGE RECEIVED: '{message.text}' ===")

    # Always persist metadata for the chat
    save_chat_metadata(message)

    # Determine sender display name for logging
    sender_username = None
    if message.from_user:
        sender_username = message.from_user.username or message.from_user.first_name

    # Skip if message has no text
    if not message.text:
        logger.info("No text in message, skipping")
        # Only respond to owner
        if authorize(message):
            bot.reply_to(message, "📨 Received your message.")
        return

    # --- Group observer: always log the message regardless of sender ---
    append_chat_history("user", message.text, chat_id=message.chat.id, username=sender_username)

    # --- Authorization gate ---
    if not authorize(message):
        # Non-owner: message was logged above, but we don't respond or process.
        logger.info(f"Non-owner message from user {message.from_user.id} in chat {message.chat.id} — logged only")
        return

    # Owner is speaking — save their private chat id for notifications
    if is_private_chat(message):
        save_chat_id(message.chat.id)

    # Get chat history BEFORE appending current message (so current message only appears once)
    from .telegram import get_chat_history
    chat_history = get_chat_history(chat_id=message.chat.id)
    logger.info(f"Chat history loaded: {len(chat_history)} chars")
    
    # Build prompt with minimal context for simple questions
    # Only include last 20 lines of chat for continuity, not the full history
    chat_lines = chat_history.strip().split('\n')
    recent_chat = '\n'.join(chat_lines[-20:]) if len(chat_lines) > 20 else chat_history
    
    # Backend + model selection:
    # - default is "codex": use Codex CLI for normal chat
    # - set TAU_CHAT_BACKEND=openai to use OpenAI directly
    # - set TAU_CHAT_BACKEND=auto to use OpenAI when OPENAI_API_KEY is available
    from .codex import CHAT_MODEL
    backend = (os.getenv("TAU_CHAT_BACKEND") or "codex").strip().lower()
    openai_model = os.getenv("TAU_OPENAI_CHAT_MODEL", "gpt-4o-mini")
    codex_model = os.getenv("TAU_CODEX_CHAT_MODEL", CHAT_MODEL)
    use_openai = backend in ("openai", "oa") or (backend == "auto" and openai_client is not None)

    # OpenAI prompt: keep it small for speed
    openai_prompt = f"""RECENT CONTEXT (for continuity):
{recent_chat}

USER: {message.text}"""

    # Codex agent prompt
    prompt_with_context = f"""You are Tau, an autonomous AI assistant running as a Telegram bot. Answer the user's question directly and concisely.

YOUR CAPABILITIES (share when asked how to use you):
Just talk to me like you would a friend. I understand natural language, so you don't need special commands.

• Ask me anything - questions, explanations, help with problems
• Set reminders - "remind me to call mom at 5pm" or "in 2 hours check if the server is up"
• Recurring tasks - "every morning send me the weather" or "every Monday remind me about the team meeting"
• Background work - "research this topic and get back to me" or "analyze this data when you have time"
• Code changes - "add a feature to do X" or "make the bot respond faster"
• Tools & web - I can search, run code, access APIs, and more

RECENT CONTEXT (for continuity):
{recent_chat}

USER: {message.text}

INSTRUCTIONS:
- Answer directly without preamble
- Be concise - just give the answer
- Do NOT say "Is there anything else..." or similar closing phrases
- Do NOT explain your thinking process in the response
- If the question is simple (like factual questions), give a short direct answer
- If asked how to use you, explain conversationally - no command syntax, just natural examples"""
    
    logger.info(f"Total prompt size: {len(prompt_with_context)} chars")
    if use_openai:
        logger.info(f"Using OpenAI fast chat model={openai_model}")
    else:
        logger.info(f"Using Codex agent model={codex_model}")
    
    # Start typing indicator in background
    typing_stop = threading.Event()
    typing_thread = threading.Thread(
        target=send_typing_action,
        args=(message.chat.id, typing_stop),
        daemon=True
    )
    typing_thread.start()
    
    start_time = datetime.now()
    logger.info("Generating response...")
    
    try:
        if use_openai:
            response = run_openai_chat_streaming(
                openai_prompt,
                chat_id=message.chat.id,
                reply_to_message_id=message.message_id,
                initial_text="🤔 Thinking...",
                model=openai_model,
                timeout_seconds=45,
                max_tokens=400,
            )
        else:
            response = run_agent_ask_streaming(
                prompt_with_context,
                chat_id=message.chat.id,
                reply_to_message_id=message.message_id,
                initial_text="🤔 Thinking...",
                model=codex_model,
                timeout_seconds=300,
            )
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Response completed in {elapsed:.1f}s (streamed)")
        append_chat_history("assistant", response, chat_id=message.chat.id)
        logger.info("Response streamed to Telegram")
        
    except subprocess.TimeoutExpired:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.error(f"Response TIMEOUT after {elapsed:.1f}s")
        error_msg = "Request timed out."
        bot.reply_to(message, error_msg)
        append_chat_history("assistant", error_msg, chat_id=message.chat.id)
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.error(f"Response ERROR after {elapsed:.1f}s: {str(e)}")
        error_msg = f"Error: {str(e)}"
        bot.reply_to(message, error_msg)
        append_chat_history("assistant", error_msg, chat_id=message.chat.id)
    finally:
        typing_stop.set()
        logger.info("=== MESSAGE HANDLING COMPLETE ===")

def main():
    """Start Tau: agent loop in background, Telegram bot in foreground."""
    logger.info("=" * 50)
    logger.info("TAU STARTING")
    logger.info("=" * 50)
    
    # Load persisted cron jobs
    load_crons()
    
    from .telegram import get_chat_id
    
    # Send startup message if we have a saved chat ID (only in debug mode)
    chat_id = get_chat_id()
    if chat_id:
        logger.info(f"Found saved chat_id: {chat_id}")
        if DEBUG_MODE:
            notify("⚚ Tau is online\n\nCommands:\n/task - Add a task\n/status - Check status\n/adapt - Self-modify\n/clear - Stop active agent processes\n/restart - Restart bot\n/kill - Stop the bot")
    else:
        logger.info("No saved chat_id, waiting for /start command")
        print("Tau starting... Send /start in Telegram to connect.")
    
    # Start agent loop in background thread
    agent_thread = threading.Thread(
        target=run_loop,
        args=(_stop_event,),
        daemon=True,
        name="AgentLoop"
    )
    agent_thread.start()
    
    # Start cron loop in background thread
    cron_thread = threading.Thread(
        target=run_cron_loop,
        args=(_stop_event,),
        daemon=True,
        name="CronLoop"
    )
    cron_thread.start()
    
    # Start chat summary loop in background thread
    summary_thread = threading.Thread(
        target=run_summary_loop,
        args=(_stop_event,),
        daemon=True,
        name="SummaryLoop"
    )
    summary_thread.start()
    
    # Start memory maintenance loop in background thread (runs daily)
    maintenance_thread = threading.Thread(
        target=run_memory_maintenance_loop,
        args=(_stop_event,),
        daemon=True,
        name="MemoryMaintenance"
    )
    maintenance_thread.start()
    
    # Run Telegram bot in main thread
    try:
        backoff_s = 1
        while not _stop_event.is_set():
            try:
                # Clean up any stale webhook/polling state before starting
                try:
                    bot.remove_webhook()
                except Exception:
                    pass
                bot.polling()
                backoff_s = 1  # reset on clean return
            except KeyboardInterrupt:
                print("\nShutting down...")
                break
            except Exception as e:
                # Telebot may raise on startup if Telegram is unreachable (e.g. DNS issues).
                # Don't crash-loop under supervisord; keep running and retry with backoff.
                logger.error(f"Telegram polling crashed: {e}", exc_info=True)
                # Stop any lingering polling threads before retrying to avoid
                # two concurrent getUpdates requests (409 Conflict).
                try:
                    bot.stop_polling()
                except Exception:
                    pass
                sleep_s = min(backoff_s, 60)
                logger.info(f"Retrying Telegram polling in {sleep_s}s...")
                time.sleep(sleep_s)
                backoff_s = min(backoff_s * 2, 60)
    finally:
        _stop_event.set()
        try:
            bot.stop_polling()
        except Exception:
            pass
        if DEBUG_MODE:
            notify("🛑 Tau stopped")


if __name__ == "__main__":
    main()
