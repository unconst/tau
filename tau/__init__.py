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

import base64
import requests as _requests
from .telegram import (
    bot, save_chat_id, notify, WORKSPACE, append_chat_history,
    TelegramStreamingMessage, authorize, is_owner, is_private_chat,
    is_group_chat, save_chat_metadata, list_chats, get_chat_history_for,
    send_to_chat, send_plan_approval, send_question_with_options,
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

# Track pending plan approvals: chat_id -> threading.Event (set=approved, clear=rejected)
_pending_plan_approvals: dict[int, dict] = {}
_plan_approval_lock = threading.Lock()

# Track pending ask_user responses: request_id -> {"event": threading.Event, "answer": str}
_pending_user_inputs: dict[str, dict] = {}
_user_input_lock = threading.Lock()

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

# Chutes API token (used for STT via Whisper on Chutes)
CHUTES_API_TOKEN = os.getenv("CHUTES_API_TOKEN")
WHISPER_URL = "https://chutes-whisper-large-v3.chutes.ai/transcribe"


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
    """Transcribe a voice file using Chutes Whisper API."""
    if not CHUTES_API_TOKEN:
        raise Exception("CHUTES_API_TOKEN not set — voice transcription unavailable")

    try:
        with open(voice_path, 'rb') as audio_file:
            audio_b64 = base64.b64encode(audio_file.read()).decode("utf-8")

        resp = _requests.post(
            WHISPER_URL,
            headers={
                "Authorization": f"Bearer {CHUTES_API_TOKEN}",
                "Content-Type": "application/json",
            },
            json={"audio_b64": audio_b64},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("text") or data.get("transcription") or ""
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

    from .llm import llm_chat
    try:
        return llm_chat(prompt)
    except Exception:
        return ""


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

                from .llm import llm_chat
                try:
                    summary = llm_chat(summary_prompt)
                except Exception:
                    logger.error("Chat summary timed out or failed")
                    time.sleep(SUMMARY_INTERVAL)
                    continue
            
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

                from .llm import run_baseagent
                response = run_baseagent(prompt_with_context)
                if response:
                    # Send result to user
                    try:
                        bot.send_message(job['chat_id'], f"⏰ Cron #{job['id']}:\n{response[:4000]}")
                        append_chat_history("assistant", f"[cron #{job['id']}]: {response}", chat_id=job['chat_id'])
                    except Exception as e:
                        logger.error(f"Failed to send cron result: {e}")
                        
            except Exception as e:
                logger.error(f"Cron job {job['id']} error: {e}")
        
        # Sleep for 1 second between checks
        time.sleep(1)


@bot.callback_query_handler(func=lambda call: call.data in ("plan_approve", "plan_reject"))
def handle_plan_callback(call):
    """Handle plan approval/rejection from inline keyboard."""
    chat_id = call.message.chat.id
    approved = call.data == "plan_approve"

    with _plan_approval_lock:
        pending = _pending_plan_approvals.get(chat_id)
        if pending:
            pending["approved"] = approved
            pending["event"].set()

    label = "✅ Plan approved" if approved else "❌ Plan rejected"
    try:
        bot.edit_message_reply_markup(chat_id, call.message.message_id, reply_markup=None)
        bot.answer_callback_query(call.id, label)
        bot.send_message(chat_id, label)
    except Exception:
        pass


@bot.callback_query_handler(func=lambda call: call.data.startswith("ask_user_"))
def handle_ask_user_callback(call):
    """Handle ask_user option selection from inline keyboard."""
    chat_id = call.message.chat.id
    option_idx = call.data.replace("ask_user_", "")

    with _user_input_lock:
        # Find the pending input for this chat
        for req_id, pending in list(_pending_user_inputs.items()):
            if pending.get("chat_id") == chat_id:
                options = pending.get("options", [])
                try:
                    idx = int(option_idx)
                    answer = options[idx] if idx < len(options) else option_idx
                except (ValueError, IndexError):
                    answer = option_idx
                pending["answer"] = answer
                pending["event"].set()
                break

    try:
        bot.edit_message_reply_markup(chat_id, call.message.message_id, reply_markup=None)
        bot.answer_callback_query(call.id, f"Selected: {answer[:30]}")
    except Exception:
        pass


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


@bot.message_handler(commands=['compress'])
def compress_context(message):
    """Manually trigger conversation summary compression.

    Regenerates the chat summary from the full conversation history,
    which can free context window space for longer conversations.
    """
    save_chat_metadata(message)
    if not authorize(message):
        return
    save_chat_id(message.chat.id)
    append_chat_history("user", "/compress", chat_id=message.chat.id)

    processing_msg = bot.reply_to(message, "🗜️ Compressing conversation context...")

    try:
        from .telegram import CHAT_HISTORY_FILE, get_chat_history

        # Force a full summary regeneration
        full_chat = ""
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE) as f:
                full_chat = f.read()

        if not full_chat.strip() or len(full_chat) < 200:
            response = "Not enough conversation history to compress."
            try:
                bot.edit_message_text(response, message.chat.id, processing_msg.message_id)
            except Exception:
                bot.reply_to(message, response)
            append_chat_history("assistant", response, chat_id=message.chat.id)
            return

        # Generate fresh compressed summary
        summary_prompt = f"""You are Tau's memory system. Compress this conversation into a concise summary.

FULL CHAT HISTORY ({len(full_chat)} chars):
{full_chat[-8000:]}

Create a summary that captures:
1. Key topics and decisions
2. User preferences
3. Ongoing tasks/projects
4. Important context for future interactions

Rules:
- Under 1500 characters
- Bullet points for scanning
- Focus on actionable info
- Current state, not play-by-play

Output ONLY the summary."""

        from .llm import llm_chat
        summary = llm_chat(summary_prompt)

        if summary and len(summary) > 50:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            summary_content = f"""# Chat Summary

<!-- Compressed via /compress command -->

## Summary

{summary}

---
*Compressed: {timestamp}*
"""
            save_summary_with_version(summary_content)

            # Also reset the chat position tracker so incremental updates start fresh
            new_chat, new_position = get_new_chat_since_last_summary()
            save_chat_position(new_position)

            response = f"✅ Context compressed ({len(full_chat)} chars → {len(summary)} char summary)"
        else:
            response = "Compression produced no useful output. Try again later."

        try:
            bot.edit_message_text(response, message.chat.id, processing_msg.message_id)
        except Exception:
            bot.reply_to(message, response)
        append_chat_history("assistant", response, chat_id=message.chat.id)

    except Exception as e:
        error_msg = f"Compression error: {str(e)}"
        try:
            bot.edit_message_text(error_msg, message.chat.id, processing_msg.message_id)
        except Exception:
            bot.reply_to(message, error_msg)
        append_chat_history("assistant", error_msg, chat_id=message.chat.id)


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
    """Run the agent CLI for adaptation and stream progress to Telegram.

    Shows a live progress dashboard while adapting, then a summary of changes.
    """
    stream = TelegramStreamingMessage(
        chat_id,
        reply_to_message_id=reply_to_message_id,
        initial_text="...",
        min_edit_interval_seconds=0.8,
        min_chars_delta=8,
    )

    from .llm import run_baseagent_streaming, parse_event, format_tool_inline

    start_time = time.time()
    streamed_text = ""
    raw_output = ""
    final_result_text: str | None = None
    tool_lines: list[str] = []
    active_tool_idx: int | None = None
    active_tool_parsed: dict | None = None
    tool_call_count = 0
    got_text_deltas = False
    _text_delta_pending = 0
    _TEXT_FLUSH_CHARS = 20
    _TEXT_FLUSH_SECS = 0.8
    _last_text_flush = 0.0

    def _build_live_text() -> str:
        parts = []
        if streamed_text.strip():
            parts.append(streamed_text.rstrip())
        if tool_lines:
            if parts:
                parts.append("")
            parts.extend(tool_lines[-8:])
        if not parts:
            elapsed = time.time() - start_time
            if elapsed < 2:
                return "..."
            return "⏳ adapting..."
        return "\n".join(parts)

    q: queue.Queue = queue.Queue()

    try:
        agent_thread = run_baseagent_streaming(
            prompt,
            readonly=False,
            event_queue=q,
            timeout_seconds=timeout_seconds,
        )
    except Exception as e:
        err = f"Error: {str(e)}"
        stream.set_text(err)
        return err

    timed_out = False
    last_display_update = start_time

    try:
        while True:
            if time.time() - start_time > timeout_seconds:
                timed_out = True
                break

            try:
                event = q.get(timeout=0.25)
            except queue.Empty:
                now = time.time()
                if _text_delta_pending > 0 and now - _last_text_flush >= _TEXT_FLUSH_SECS:
                    _text_delta_pending = 0
                    _last_text_flush = now
                    stream.set_text(_build_live_text())
                    last_display_update = now
                elif now - last_display_update >= 1.5:
                    stream.set_text(_build_live_text())
                    last_display_update = now
                continue

            if event is None:
                break

            etype = event.get("type", "")
            if etype == "stream.tool.completed":
                tool_call_count += 1

            parsed = parse_event(event)
            if parsed is None:
                continue

            kind = parsed["kind"]

            if kind == "text_delta":
                delta = parsed.get("text", "")
                if delta:
                    got_text_deltas = True
                    streamed_text += delta
                    _text_delta_pending += len(delta)
                    now = time.time()
                    if _text_delta_pending >= _TEXT_FLUSH_CHARS or now - _last_text_flush >= _TEXT_FLUSH_SECS:
                        _text_delta_pending = 0
                        _last_text_flush = now
                        stream.set_text(_build_live_text())
                        last_display_update = now

            elif kind == "tool_start":
                active_tool_parsed = parsed
                inline = format_tool_inline(parsed, done=False)
                tool_lines.append(inline)
                active_tool_idx = len(tool_lines) - 1
                stream.set_text(_build_live_text())
                last_display_update = time.time()

            elif kind == "tool_done":
                if active_tool_idx is not None and active_tool_idx < len(tool_lines):
                    done_inline = format_tool_inline(
                        active_tool_parsed or parsed, done=True
                    )
                    tool_lines[active_tool_idx] = done_inline
                    stream.set_text(_build_live_text())
                    last_display_update = time.time()
                active_tool_idx = None
                active_tool_parsed = None

            elif kind in ("message", "message_done"):
                text = parsed.get("text", "")
                if text:
                    if kind == "message_done":
                        raw_output = text
                    elif len(text) > len(raw_output):
                        raw_output = text
                    if not got_text_deltas and text.strip():
                        streamed_text = text
                        stream.set_text(_build_live_text())
                        last_display_update = time.time()

            elif kind == "error":
                err_text = parsed.get("text", "Agent error")
                if not final_result_text:
                    final_result_text = f"Error: {err_text}"

    finally:
        stream.finalize()

    if timed_out:
        msg = "⏰ Adaptation timed out."
        stream.set_text(msg)
        return msg

    output = (final_result_text or raw_output or streamed_text or "").strip()

    if output:
        summary = output
        summary = re.sub(r'```[\s\S]*?```', '[code]', summary)
        if len(summary) > 600:
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
        # Fallback: summarize tool operations
        completed = [l for l in tool_lines if "✓" in l]
        if completed:
            final_msg = "✅ Done\n\n" + "\n".join(completed[-10:])
        else:
            final_msg = "✅ Done"

    stream.set_text(final_msg)
    return output or "Adaptation complete"


@bot.message_handler(commands=['adapt'])
def adapt_bot(message):
    """Self-modify the bot using the agent, then restart."""
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
        initial_text="...",
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

    from .llm import run_baseagent_streaming, parse_event

    start_time = time.time()
    thinking_text_buffer: str = ""
    final_result_text: str | None = None

    q: queue.Queue = queue.Queue()

    try:
        agent_thread = run_baseagent_streaming(
            plan_prompt,
            readonly=True,
            event_queue=q,
            timeout_seconds=timeout_seconds,
        )
    except Exception as e:
        err = f"Error: {str(e)}"
        stream.set_text(err)
        return err, ""

    timed_out = False

    try:
        while True:
            if time.time() - start_time > timeout_seconds:
                timed_out = True
                break

            try:
                event = q.get(timeout=0.5)
            except queue.Empty:
                continue

            if event is None:
                break

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
                    if len(thinking_text_buffer) > 50:
                        final_result_text = thinking_text_buffer

            elif kind == "error":
                err_text = parsed.get("text", "Plan generation error")
                if not final_result_text:
                    final_result_text = f"Error: {err_text}"

    finally:
        stream.finalize()

    if timed_out:
        msg = "⏰ Plan generation timed out."
        stream.set_text(msg)
        return msg, ""

    plan_content = (final_result_text or thinking_text_buffer or "").strip()
    
    if not plan_content:
        msg = "Failed to generate plan."
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
            from .llm import CHAT_MODEL
            agent_model = os.getenv("TAU_CHAT_MODEL", CHAT_MODEL)

            response = run_agent_ask_streaming(
                prompt_with_context,
                chat_id=message.chat.id,
                existing_message_id=processing_msg.message_id,
                initial_text="🎤 Thinking...",
                model=agent_model,
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


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    """Handle photo messages — support video generation from images.

    If the user sends a photo with a caption, treat the caption as a video
    generation prompt (or as a general message about the image).
    If no caption, ask what they'd like to do with the image.
    """
    save_chat_metadata(message)
    if not authorize(message):
        return
    save_chat_id(message.chat.id)

    caption = (message.caption or "").strip()
    append_chat_history("user", f"[photo]{': ' + caption if caption else ''}", chat_id=message.chat.id)

    # Check if caption explicitly asks for video generation
    video_keywords = [
        "video", "animate", "animation", "motion", "move", "moving",
        "make it move", "bring to life", "come to life", "i2v",
        "generate video", "create video", "make video",
    ]
    wants_video = caption and any(kw in caption.lower() for kw in video_keywords)

    if wants_video:
        # Extract the prompt — remove the video-trigger keywords for a cleaner prompt
        prompt = caption
        # Remove common trigger phrases to get the actual motion description
        for phrase in ["generate a video of", "generate video of", "create a video of",
                       "create video of", "make a video of", "make video of",
                       "animate this:", "animate:", "video:", "make it move:",
                       "bring to life:", "animate this", "make a video",
                       "generate a video", "generate video", "create a video",
                       "create video", "make video"]:
            if prompt.lower().startswith(phrase):
                prompt = prompt[len(phrase):].strip()
                break

        if not prompt:
            prompt = "Smooth natural motion, cinematic quality"

        processing_msg = bot.reply_to(message, "🎬 Generating video... this may take a minute or two.")

        photo_path = None
        try:
            # Download the highest resolution photo
            file_id = message.photo[-1].file_id
            file_info = bot.get_file(file_id)
            downloaded = bot.download_file(file_info.file_path)

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(downloaded)
                photo_path = tmp.name

            from tau.tools.generate_video import send_video_message
            result = send_video_message(
                message.chat.id,
                prompt,
                photo_path,
                reply_to_message_id=message.message_id,
            )

            # Update the processing message
            try:
                bot.edit_message_text(
                    f"✅ Video generated: {prompt[:100]}",
                    message.chat.id,
                    processing_msg.message_id,
                )
            except Exception:
                pass

            append_chat_history("assistant", f"[video generated]: {prompt}", chat_id=message.chat.id)

        except Exception as e:
            error_msg = f"Video generation failed: {str(e)}"
            try:
                bot.edit_message_text(error_msg, message.chat.id, processing_msg.message_id)
            except Exception:
                bot.reply_to(message, error_msg)
            append_chat_history("assistant", error_msg, chat_id=message.chat.id)
        finally:
            if photo_path and os.path.exists(photo_path):
                try:
                    os.unlink(photo_path)
                except Exception:
                    pass
    elif caption:
        # Photo with a non-video caption — process as a regular message with image context
        # Save the photo temporarily and pass to the agent
        photo_path = None
        try:
            file_id = message.photo[-1].file_id
            file_info = bot.get_file(file_id)
            downloaded = bot.download_file(file_info.file_path)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(downloaded)
                photo_path = tmp.name

            # Let the agent handle it with image context
            from .llm import CHAT_MODEL, build_tau_system_prompt
            agent_model = os.getenv("TAU_CHAT_MODEL", CHAT_MODEL)
            tau_system_prompt = build_tau_system_prompt(
                chat_id=message.chat.id,
                user_message=caption,
            )

            prompt_with_context = (
                f"The user sent a photo (saved at {photo_path}) with caption: {caption}\n\n"
                f"If they want a video from this image, use the generate_video tool with "
                f"the image path and an appropriate prompt.\n"
                f"Otherwise, respond to their caption normally."
            )

            typing_stop = threading.Event()
            typing_thread = threading.Thread(
                target=send_typing_action,
                args=(message.chat.id, typing_stop),
                daemon=True,
            )
            typing_thread.start()

            try:
                response = run_agent_ask_streaming(
                    prompt_with_context,
                    chat_id=message.chat.id,
                    reply_to_message_id=message.message_id,
                    initial_text="🤔",
                    model=agent_model,
                    timeout_seconds=300,
                    system_prompt_override=tau_system_prompt,
                    readonly=False,
                )
                append_chat_history("assistant", response, chat_id=message.chat.id)
            finally:
                typing_stop.set()

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            bot.reply_to(message, error_msg)
            append_chat_history("assistant", error_msg, chat_id=message.chat.id)
        finally:
            if photo_path and os.path.exists(photo_path):
                try:
                    os.unlink(photo_path)
                except Exception:
                    pass
    else:
        # Photo with no caption — tell the user what they can do
        response = (
            "📷 Got your photo! You can:\n"
            "• Send it again with a caption like \"animate this — a bird flying\" to generate a video\n"
            "• Or just describe what you'd like me to do with it"
        )
        bot.reply_to(message, response)
        append_chat_history("assistant", response, chat_id=message.chat.id)


@bot.message_handler(content_types=['document', 'sticker', 'video', 'audio', 'animation', 'video_note', 'contact', 'location', 'venue', 'poll'])
def handle_other_content(message):
    """Handle non-text content types with a confirmation."""
    save_chat_metadata(message)
    if not authorize(message):
        return
    save_chat_id(message.chat.id)
    
    # Determine content type for the confirmation message
    content_type = message.content_type
    response = f"📨 Received {content_type}. I can process text, voice, and photo messages."
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
    system_prompt_override: str | None = None,
    readonly: bool = True,
    plan_first: bool = False,
) -> str:
    """Run the agent and stream JSONL output by editing one Telegram message.

    Shows a live progress dashboard while working:
    - Thinking snippets from the model
    - Tool operations (shell, file edit, web search, etc.)
    - Footer with elapsed time, tool count, context size, cost
    Then replaces everything with the final answer.
    """
    if model is None:
        from .llm import CHAT_MODEL
        model = CHAT_MODEL

    stream = TelegramStreamingMessage(
        chat_id,
        reply_to_message_id=reply_to_message_id,
        existing_message_id=existing_message_id,
        initial_text="...",
        min_edit_interval_seconds=0.8,
        min_chars_delta=8,
    )

    from .llm import run_baseagent_streaming, parse_event, format_tool_inline

    start_time = time.time()
    # Streamed text accumulator — holds the live text as the model generates it
    streamed_text = ""
    # Final complete message text (from item.completed agent_message)
    raw_output = ""
    final_result_text: str | None = None
    # Tool tracking
    tool_lines: list[str] = []  # compact inline tool indicators
    active_tool_idx: int | None = None  # index into tool_lines of current tool
    active_tool_parsed: dict | None = None
    tool_call_count = 0
    # Track whether we've seen any real text deltas
    got_text_deltas = False
    # Buffer text deltas to reduce Telegram edit jitter (~3 words at a time)
    _text_delta_pending = 0  # chars accumulated since last flush
    _TEXT_FLUSH_CHARS = 20   # flush after ~3 words
    _TEXT_FLUSH_SECS = 0.8   # or after this many seconds
    _last_text_flush = 0.0

    def _build_live_text() -> str:
        """Build the message shown while the agent is working.

        Layout:
          <streamed model text so far>

          ⏳ tool indicator (if tool running)
          ✓ tool indicator (completed tools)
        """
        parts = []

        # Main streamed text
        if streamed_text.strip():
            parts.append(streamed_text.rstrip())

        # Tool activity block (compact, appended after text)
        if tool_lines:
            # Show a blank line separator if there's text above
            if parts:
                parts.append("")
            parts.extend(tool_lines[-8:])

        if not parts:
            elapsed = time.time() - start_time
            if elapsed < 2:
                return "..."
            return "⏳ thinking..."

        return "\n".join(parts)

    q: queue.Queue = queue.Queue()

    # Enable chat_mode when a Tau system prompt is provided (conversational use)
    is_chat_mode = system_prompt_override is not None

    # Plan approval callback for plan_first mode
    plan_approval_cb = None
    if plan_first:
        def plan_approval_cb(plan_text: str) -> bool:
            """Block until user approves/rejects the plan via Telegram inline keyboard."""
            approval_event = threading.Event()
            approval_state = {"approved": False, "event": approval_event}
            with _plan_approval_lock:
                _pending_plan_approvals[chat_id] = approval_state

            send_plan_approval(chat_id, plan_text, reply_to_message_id=reply_to_message_id)

            # Wait up to 5 minutes for user response
            approval_event.wait(timeout=300)

            with _plan_approval_lock:
                _pending_plan_approvals.pop(chat_id, None)

            return approval_state.get("approved", False)

    try:
        agent_thread = run_baseagent_streaming(
            prompt_with_context,
            model=model,
            readonly=readonly,
            event_queue=q,
            timeout_seconds=timeout_seconds,
            system_prompt_override=system_prompt_override,
            chat_mode=is_chat_mode,
            plan_first=plan_first,
            plan_approval_callback=plan_approval_cb,
            chat_id=chat_id,
        )
    except Exception as e:
        err = f"Error: {str(e)}"
        stream.set_text(err)
        return err

    timed_out = False
    last_display_update = start_time

    try:
        while True:
            if time.time() - start_time > timeout_seconds:
                timed_out = True
                break

            try:
                event = q.get(timeout=0.25)
            except queue.Empty:
                # Flush buffered text deltas if enough time has passed
                now = time.time()
                if _text_delta_pending > 0 and now - _last_text_flush >= _TEXT_FLUSH_SECS:
                    _text_delta_pending = 0
                    _last_text_flush = now
                    stream.set_text(_build_live_text())
                    last_display_update = now
                elif now - last_display_update >= 1.5:
                    stream.set_text(_build_live_text())
                    last_display_update = now
                continue

            if event is None:
                break

            # Track tool count from raw events
            etype = event.get("type", "")
            if etype == "stream.tool.completed":
                tool_call_count += 1

            parsed = parse_event(event)
            if parsed is None:
                continue

            kind = parsed["kind"]

            # --- Real-time text streaming (buffered ~3 words) ---
            if kind == "text_delta":
                delta = parsed.get("text", "")
                if delta:
                    got_text_deltas = True
                    streamed_text += delta
                    _text_delta_pending += len(delta)
                    now = time.time()
                    if _text_delta_pending >= _TEXT_FLUSH_CHARS or now - _last_text_flush >= _TEXT_FLUSH_SECS:
                        _text_delta_pending = 0
                        _last_text_flush = now
                        stream.set_text(_build_live_text())
                        last_display_update = now

            # --- Tool started: add inline indicator ---
            elif kind == "tool_start":
                active_tool_parsed = parsed
                inline = format_tool_inline(parsed, done=False)
                tool_lines.append(inline)
                active_tool_idx = len(tool_lines) - 1
                stream.set_text(_build_live_text())
                last_display_update = time.time()

            # --- Tool done: update indicator to ✓ ---
            elif kind == "tool_done":
                if active_tool_idx is not None and active_tool_idx < len(tool_lines):
                    done_inline = format_tool_inline(
                        active_tool_parsed or parsed, done=True
                    )
                    tool_lines[active_tool_idx] = done_inline
                    stream.set_text(_build_live_text())
                    last_display_update = time.time()
                active_tool_idx = None
                active_tool_parsed = None

            # --- Full message updates (fallback if no deltas) ---
            elif kind in ("message", "message_done"):
                text = parsed.get("text", "")
                if text:
                    if kind == "message_done":
                        raw_output = text
                    elif len(text) > len(raw_output):
                        raw_output = text
                    # If we haven't gotten streaming deltas, use the full
                    # message text as the streamed display
                    if not got_text_deltas and text.strip():
                        streamed_text = text
                        stream.set_text(_build_live_text())
                        last_display_update = time.time()

            # --- Plan events ---
            elif kind == "plan_proposed":
                plan_text = parsed.get("text", "")
                if plan_text:
                    tool_lines.append("📋 Plan proposed — waiting for approval...")
                    stream.set_text(_build_live_text())
                    last_display_update = time.time()

            elif kind == "plan_approved":
                tool_lines.append("✅ Plan approved — executing...")
                stream.set_text(_build_live_text())
                last_display_update = time.time()

            elif kind == "plan_rejected":
                tool_lines.append("❌ Plan rejected")
                stream.set_text(_build_live_text())
                last_display_update = time.time()

            elif kind == "user_input_requested":
                question = parsed.get("text", "")
                if question:
                    tool_lines.append(f"❓ {question[:60]}")
                    stream.set_text(_build_live_text())
                    last_display_update = time.time()

            elif kind == "turn_done":
                if raw_output:
                    final_result_text = raw_output

            elif kind == "error":
                err_text = parsed.get("text", "Agent error")
                if not final_result_text:
                    final_result_text = f"Error: {err_text}"

    finally:
        stream.finalize()

    if timed_out:
        msg = "⏰ Request timed out."
        stream.set_text(msg)
        return msg

    output = (final_result_text or raw_output or streamed_text or "").strip()

    if not output:
        msg = "No response from agent."
        stream.set_text(msg)
        return msg

    final_answer = _extract_final_answer(output)
    stream.set_text(final_answer)
    return final_answer




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

    # Check if this is a reply to a pending ask_user question (text reply)
    with _user_input_lock:
        for req_id, pending in list(_pending_user_inputs.items()):
            if pending.get("chat_id") == message.chat.id and not pending["event"].is_set():
                pending["answer"] = message.text
                pending["event"].set()
                _pending_user_inputs.pop(req_id, None)
                # Don't process this as a normal message
                return

    # Model selection — always use the agent path via Chutes
    from .llm import CHAT_MODEL, build_tau_system_prompt
    agent_model = os.getenv("TAU_CHAT_MODEL", CHAT_MODEL)

    tau_system_prompt = build_tau_system_prompt(
        chat_id=message.chat.id,
        user_message=message.text,
    )
    logger.info(f"Using agent model={agent_model}, system_prompt={len(tau_system_prompt)} chars")
    
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
        response = run_agent_ask_streaming(
            message.text,
            chat_id=message.chat.id,
            reply_to_message_id=message.message_id,
            initial_text="🤔 Thinking...",
            model=agent_model,
            timeout_seconds=300,
            system_prompt_override=tau_system_prompt,
            readonly=False,
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
