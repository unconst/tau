import os
import subprocess
import sys
import threading
import tempfile
import logging
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from .telegram import bot, save_chat_id, notify, WORKSPACE, append_chat_history
from .agent import run_loop, TASKS_DIR, get_all_tasks, git_commit_changes, set_debug_mode
import re

# Debug mode flag - controls verbose notifications
DEBUG_MODE = False

# Storage for active cron jobs: list of {id, interval_seconds, prompt, next_run, chat_id}
_cron_jobs = []
_cron_lock = threading.Lock()
_next_cron_id = 1

# Configure logging
LOG_FILE = os.path.join(WORKSPACE, "logs", "tau.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import task-specific scripts from their task directories
task1_path = os.path.join(WORKSPACE, "context", "tasks", "task-1")
if os.path.exists(task1_path):
    sys.path.insert(0, task1_path)
    try:
        from bitcoin import run_hourly_scheduler
    except ImportError:
        run_hourly_scheduler = None
else:
    run_hourly_scheduler = None

# Import morning scheduler from task-6
task6_path = os.path.join(WORKSPACE, "context", "tasks", "task-6")
if os.path.exists(task6_path):
    sys.path.insert(0, task6_path)
    try:
        from morning_scheduler import run_morning_scheduler
    except ImportError:
        run_morning_scheduler = None
else:
    run_morning_scheduler = None

MEMORY_FILE = os.path.join(WORKSPACE, "context", "tasks", "memory.md")

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
                
                prompt_with_context = f"""TELEGRAM CHAT HISTORY:
{chat_history}

CRON JOB PROMPT (runs every {job['interval_seconds']}s):
{job['prompt']}

Please execute this scheduled task. Provide a fresh update based on the current state or by using your tools. Do not simply repeat previous messages."""
                
                result = subprocess.run(
                    [
                        "agent",
                        "--force",
                        "--model", "opus-4.5",
                        "--mode=ask",
                        "--output-format=text",
                        "--print",
                        prompt_with_context
                    ],
                    capture_output=True,
                    text=True,
                    stdin=subprocess.DEVNULL,
                    timeout=120
                )
                
                response = result.stdout.strip() if result.stdout else result.stderr.strip()
                if response:
                    # Send result to user
                    try:
                        bot.send_message(job['chat_id'], f"⏰ Cron #{job['id']}:\n{response[:4000]}")
                        append_chat_history("assistant", f"[cron #{job['id']}]: {response}")
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
    
    save_chat_id(message.chat.id)
    
    # Parse: /cron <interval> <prompt>
    text = message.text.replace('/cron', '', 1).strip()
    append_chat_history("user", f"/cron {text}")
    
    if not text:
        response = "Usage: /cron <interval> <prompt>\nExamples:\n  /cron 5min check the weather\n  /cron 30sec ping\n  /cron 1h summarize news"
        bot.reply_to(message, response)
        append_chat_history("assistant", response)
        return
    
    # Split into interval and prompt
    parts = text.split(None, 1)
    if len(parts) < 2:
        response = "Usage: /cron <interval> <prompt>\nNeed both an interval and a prompt."
        bot.reply_to(message, response)
        append_chat_history("assistant", response)
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
    
    # Format interval for display
    if interval_seconds < 60:
        interval_display = f"{interval_seconds}sec"
    elif interval_seconds < 3600:
        interval_display = f"{interval_seconds // 60}min"
    else:
        interval_display = f"{interval_seconds // 3600}h"
    
    response = f"✅ Cron #{cron_id} created: every {interval_display}\nPrompt: {prompt[:100]}"
    bot.reply_to(message, response)
    append_chat_history("assistant", response)


@bot.message_handler(commands=['crons'])
def list_crons(message):
    """List all active cron jobs."""
    save_chat_id(message.chat.id)
    append_chat_history("user", "/crons")
    
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
    append_chat_history("assistant", response)


@bot.message_handler(commands=['uncron'])
def remove_cron(message):
    """Remove a cron job by ID."""
    save_chat_id(message.chat.id)
    
    text = message.text.replace('/uncron', '', 1).strip()
    append_chat_history("user", f"/uncron {text}")
    
    if not text:
        response = "Usage: /uncron <id>\nUse /crons to see active jobs."
        bot.reply_to(message, response)
        append_chat_history("assistant", response)
        return
    
    try:
        cron_id = int(text.replace('#', ''))
    except ValueError:
        response = "Invalid cron ID. Use /crons to see active jobs."
        bot.reply_to(message, response)
        append_chat_history("assistant", response)
        return
    
    with _cron_lock:
        for i, job in enumerate(_cron_jobs):
            if job['id'] == cron_id:
                _cron_jobs.pop(i)
                response = f"✅ Cron #{cron_id} removed."
                bot.reply_to(message, response)
                append_chat_history("assistant", response)
                return
    
    response = f"Cron #{cron_id} not found. Use /crons to see active jobs."
    bot.reply_to(message, response)
    append_chat_history("assistant", response)


@bot.message_handler(commands=['start'])
def send_welcome(message):
    save_chat_id(message.chat.id)
    append_chat_history("user", f"/start")
    response = "Hello! I'm Tau. Commands:\n/task <description> - Add a task\n/status - See recent activity\n/adapt <prompt> - Self-modify\n/cron <interval> <prompt> - Schedule recurring prompt\n/crons - List active crons\n/uncron <id> - Remove a cron\n/restart - Restart bot\n/debug - Toggle debug mode"
    bot.reply_to(message, response)
    append_chat_history("assistant", response)


@bot.message_handler(commands=['debug'])
def toggle_debug(message):
    """Toggle debug mode on/off."""
    global DEBUG_MODE
    save_chat_id(message.chat.id)
    append_chat_history("user", "/debug")
    
    DEBUG_MODE = not DEBUG_MODE
    set_debug_mode(DEBUG_MODE)
    
    status = "on" if DEBUG_MODE else "off"
    response = f"Debug mode: {status}"
    bot.reply_to(message, response)
    append_chat_history("assistant", response)

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


@bot.message_handler(commands=['restart'])
def restart_bot(message):
    """Restart the bot process via supervisor."""
    append_chat_history("user", f"/restart")
    response = "Restarting..."
    bot.reply_to(message, response)
    append_chat_history("assistant", response)
    bot.stop_polling()
    
    # Try supervisor restart first
    if restart_via_supervisor():
        # Supervisor will restart us - just exit cleanly
        sys.exit(0)
    else:
        # Fallback to direct exec restart
        os.execv(sys.executable, [sys.executable] + sys.argv)

@bot.message_handler(commands=['adapt'])
def adapt_bot(message):
    """Self-modify the bot using cursor agent, then restart."""
    # Extract prompt after /adapt
    prompt = message.text.replace('/adapt', '', 1).strip()
    if not prompt:
        response = "Usage: /adapt <prompt>"
        bot.reply_to(message, response)
        append_chat_history("user", f"/adapt")
        append_chat_history("assistant", response)
        return
    
    append_chat_history("user", f"/adapt {prompt}")
    save_chat_id(message.chat.id)
    
    # Acknowledge with single emoji
    bot.reply_to(message, "🫡")
    append_chat_history("assistant", "🫡")
    
    try:
        result = subprocess.run(
            ["agent", "--force", "--model", "opus-4.5",
             "--output-format=text", "--print", prompt],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,  # Prevent agent from waiting for input
            timeout=600,  # 10 min timeout for code changes
            cwd=WORKSPACE
        )
        
        # Commit any changes made by the agent
        git_commit_changes(prompt)
        
        # Just checkmark when done
        bot.reply_to(message, "✅")
        append_chat_history("assistant", "✅")
        bot.stop_polling()
        
        # Try supervisor restart first
        if restart_via_supervisor():
            sys.exit(0)
        else:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        
    except subprocess.TimeoutExpired:
        error_msg = "Adaptation timed out."
        bot.reply_to(message, error_msg)
        append_chat_history("assistant", error_msg)
    except Exception as e:
        error_msg = f"Adaptation error: {str(e)}"
        bot.reply_to(message, error_msg)
        append_chat_history("assistant", error_msg)


@bot.message_handler(commands=['task'])
def add_task(message):
    """Add a task to its own directory."""
    save_chat_id(message.chat.id)
    
    task_text = message.text.replace('/task', '', 1).strip()
    append_chat_history("user", f"/task {task_text}")
    
    if not task_text:
        response = "Usage: /task <description>"
        bot.reply_to(message, response)
        append_chat_history("assistant", response)
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
    append_chat_history("assistant", response)


@bot.message_handler(commands=['status'])
def get_status(message):
    """Show recent high-level memory and task status."""
    save_chat_id(message.chat.id)
    append_chat_history("user", "/status")
    
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
        append_chat_history("assistant", response)
    else:
        response = status_msg[:4000]
        bot.reply_to(message, response)
        append_chat_history("assistant", response)


@bot.message_handler(content_types=['voice'])
def handle_voice_message(message):
    """Handle voice messages by transcribing and processing as text."""
    save_chat_id(message.chat.id)
    
    # Notify user that we're processing the voice message
    bot.reply_to(message, "🎤 Processing voice message...")
    
    voice_path = None
    try:
        # Download the voice file
        voice_path = download_voice_file(message.voice.file_id)
        
        # Transcribe using OpenAI
        transcribed_text = transcribe_voice(voice_path)
        
        if not transcribed_text.strip():
            response = "Could not transcribe voice message. Please try again."
            bot.reply_to(message, response)
            append_chat_history("user", "[voice message - transcription failed]")
            append_chat_history("assistant", response)
            return
        
        # Append transcribed text to chat history with voice indicator
        append_chat_history("user", f"[voice]: {transcribed_text}")
        
        # Get chat history BEFORE processing (so current message only appears once)
        from .telegram import get_chat_history
        chat_history = get_chat_history()
        
        # Build prompt with chat history context
        prompt_with_context = f"""TELEGRAM CHAT HISTORY:
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
            result = subprocess.run(
                [
                    "agent",
                    "--force",
                    "--model", "gemini-3-flash",
                    "--mode=ask",
                    "--output-format=text",
                    "--print",
                    prompt_with_context
                ],
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,  # Prevent agent from waiting for input
                timeout=120
            )
            response = result.stdout.strip() if result.stdout else result.stderr.strip()
            if not response:
                response = "No response from agent."
            bot.reply_to(message, response)
            append_chat_history("assistant", response)
        finally:
            # Stop typing indicator
            typing_stop.set()
        
    except Exception as e:
        error_msg = f"Error processing voice message: {str(e)}"
        bot.reply_to(message, error_msg)
        append_chat_history("assistant", error_msg)
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
    save_chat_id(message.chat.id)
    
    # Determine content type for the confirmation message
    content_type = message.content_type
    response = f"📨 Received {content_type}. I can process text and voice messages."
    bot.reply_to(message, response)
    append_chat_history("user", f"[{content_type}]")
    append_chat_history("assistant", response)


def send_typing_action(chat_id, stop_event):
    """Send typing action every 4 seconds until stop_event is set."""
    while not stop_event.is_set():
        try:
            bot.send_chat_action(chat_id, 'typing')
        except Exception:
            pass
        # Wait 4 seconds (typing indicator lasts ~5 seconds)
        stop_event.wait(4)


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    """Handle all non-command text messages by calling the agent."""
    logger.info(f"=== MESSAGE RECEIVED: '{message.text[:50]}...' ===" if len(message.text or '') > 50 else f"=== MESSAGE RECEIVED: '{message.text}' ===")
    save_chat_id(message.chat.id)
    
    # Skip if message has no text
    if not message.text:
        logger.info("No text in message, skipping")
        bot.reply_to(message, "📨 Received your message.")
        return
    
    # Get chat history BEFORE appending current message (so current message only appears once)
    from .telegram import get_chat_history
    chat_history = get_chat_history()
    logger.info(f"Chat history loaded: {len(chat_history)} chars")
    
    # Now append the user message
    append_chat_history("user", message.text)
    
    # Build prompt with chat history context
    prompt_with_context = f"""TELEGRAM CHAT HISTORY:
{chat_history}

CURRENT USER MESSAGE:
{message.text}

Please respond to the user's message above, considering the full context of our conversation history."""
    
    logger.info(f"Total prompt size: {len(prompt_with_context)} chars")
    
    # Start typing indicator in background
    typing_stop = threading.Event()
    typing_thread = threading.Thread(
        target=send_typing_action,
        args=(message.chat.id, typing_stop),
        daemon=True
    )
    typing_thread.start()
    
    start_time = datetime.now()
    logger.info("Calling agent CLI...")
    
    try:
        # CRITICAL: stdin=subprocess.DEVNULL prevents agent from waiting for input
        result = subprocess.run(
            [
                "agent",
                "--force",
                "--model", "gemini-3-flash",
                "--mode=ask",
                "--output-format=text",
                "--print",
                prompt_with_context
            ],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=60
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Agent completed in {elapsed:.1f}s, exit code: {result.returncode}")
        
        response = result.stdout.strip() if result.stdout else result.stderr.strip()
        if not response:
            logger.warning("Agent returned empty response")
            response = "No response from agent."
        else:
            logger.info(f"Response preview: {response[:100]}...")
        
        bot.reply_to(message, response[:4000])
        append_chat_history("assistant", response)
        logger.info("Response sent to Telegram")
        
    except subprocess.TimeoutExpired:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.error(f"Agent TIMEOUT after {elapsed:.1f}s")
        error_msg = "Request timed out."
        bot.reply_to(message, error_msg)
        append_chat_history("assistant", error_msg)
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.error(f"Agent ERROR after {elapsed:.1f}s: {str(e)}")
        error_msg = f"Error: {str(e)}"
        bot.reply_to(message, error_msg)
        append_chat_history("assistant", error_msg)
    finally:
        typing_stop.set()
        logger.info("=== MESSAGE HANDLING COMPLETE ===")
        logger.info("=== MESSAGE HANDLING COMPLETE ===")

def main():
    """Start Tau: agent loop in background, Telegram bot in foreground."""
    logger.info("=" * 50)
    logger.info("TAU STARTING")
    logger.info("=" * 50)
    
    from .telegram import get_chat_id
    
    # Send startup message if we have a saved chat ID (only in debug mode)
    chat_id = get_chat_id()
    if chat_id:
        logger.info(f"Found saved chat_id: {chat_id}")
        if DEBUG_MODE:
            notify("⚚ Tau is online\n\nCommands:\n/task - Add a task\n/status - Check status\n/adapt - Self-modify\n/restart - Restart bot")
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
    
    # Start Bitcoin price scheduler in background thread (if available)
    if run_hourly_scheduler:
        bitcoin_thread = threading.Thread(
            target=run_hourly_scheduler,
            args=(_stop_event,),
            daemon=True,
            name="BitcoinScheduler"
        )
        bitcoin_thread.start()
    
    # Start morning message scheduler in background thread (if available)
    if run_morning_scheduler:
        morning_thread = threading.Thread(
            target=run_morning_scheduler,
            args=(_stop_event,),
            daemon=True,
            name="MorningScheduler"
        )
        morning_thread.start()
    
    # Run Telegram bot in main thread
    try:
        bot.polling()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        _stop_event.set()
        if DEBUG_MODE:
            notify("🛑 Tau stopped")


if __name__ == "__main__":
    main()
