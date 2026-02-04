import os
import json
import queue
import subprocess
import sys
import threading
import tempfile
import time
import logging
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from .telegram import bot, save_chat_id, notify, WORKSPACE, append_chat_history, TelegramStreamingMessage
from .agent import run_loop, TASKS_DIR, get_all_tasks, git_commit_changes, set_debug_mode
from . import processes
import re

# Debug mode flag - controls verbose notifications
DEBUG_MODE = False

# Storage for active cron jobs: list of {id, interval_seconds, prompt, next_run, chat_id}
_cron_jobs = []
_cron_lock = threading.Lock()
_next_cron_id = 1

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

                cmd = [
                    "agent",
                    "--force",
                    "--model",
                    "opus-4.5",
                    "--mode=ask",
                    "--output-format=text",
                    "--print",
                    prompt_with_context,
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

                response = stdout.strip() if stdout.strip() else stderr.strip()
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
    response = "Hello! I'm Tau. Commands:\n/task <description> - Add a task\n/status - See recent activity\n/adapt <prompt> - Self-modify\n/cron <interval> <prompt> - Schedule recurring prompt\n/crons - List active crons\n/uncron <id> - Remove a cron\n/clear - Stop active agent processes\n/restart - Restart bot\n/kill - Stop the bot\n/debug - Toggle debug mode"
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
    save_chat_id(message.chat.id)
    append_chat_history("user", "/clear")

    stopped = processes.terminate_all(label_prefix="agent:", timeout_seconds=2.0)
    if stopped:
        response = f"🧹 Stopped {len(stopped)} active agent process(es)."
    else:
        response = "No active agent processes."

    bot.reply_to(message, response)
    append_chat_history("assistant", response)


@bot.message_handler(commands=['kill'])
def kill_bot(message):
    """Fully stop the bot process (and attempt to stop supervisord-managed service)."""
    save_chat_id(message.chat.id)
    append_chat_history("user", "/kill")

    response = "🛑 Killing bot..."
    bot.reply_to(message, response)
    append_chat_history("assistant", response)

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

    cmd = [
        "agent",
        "--force",
        "--model",
        "opus-4.5",
        "--output-format",
        "stream-json",
        "--stream-partial-output",
        "--print",
        prompt,
    ]

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
    
    def build_display():
        """Build the display message showing current thinking and actions."""
        lines = ["🫡 Adapting...\n"]
        
        # Show recent thinking snippet if we have one
        if last_thinking_snippet:
            # Show a brief snippet of the agent's current thinking
            snippet = last_thinking_snippet[:100]
            if len(last_thinking_snippet) > 100:
                snippet += "..."
            lines.append(f"💭 {snippet}\n")
        
        # Show recent tool actions
        if thinking_updates:
            lines.append("")
            lines.extend(thinking_updates[-6:])
        
        return "\n".join(lines)

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
                
                event_type = event.get("type")
                
                # Handle tool use events - show what the agent is doing
                if event_type == "tool_use":
                    tool_name = event.get("name", "tool")
                    tool_input = event.get("input", {})
                    current_tool = tool_name
                    
                    # Create a brief description of what's happening
                    if tool_name == "Read":
                        path = tool_input.get("path", "")
                        filename = os.path.basename(path) if path else "file"
                        update = f"📖 Reading {filename}..."
                    elif tool_name == "Write":
                        path = tool_input.get("path", "")
                        filename = os.path.basename(path) if path else "file"
                        update = f"✍️ Writing {filename}..."
                    elif tool_name == "StrReplace":
                        path = tool_input.get("path", "")
                        filename = os.path.basename(path) if path else "file"
                        update = f"🔧 Editing {filename}..."
                    elif tool_name == "Shell":
                        cmd_str = tool_input.get("command", "")[:40]
                        update = f"💻 {cmd_str}..."
                    elif tool_name == "Grep":
                        pattern = tool_input.get("pattern", "")[:25]
                        update = f"🔍 Searching: {pattern}..."
                    elif tool_name == "Glob":
                        pattern = tool_input.get("glob_pattern", "")[:25]
                        update = f"📁 Finding: {pattern}..."
                    elif tool_name == "SemanticSearch":
                        query = tool_input.get("query", "")[:40]
                        update = f"🧠 {query}..."
                    elif tool_name == "WebFetch":
                        url = tool_input.get("url", "")[:40]
                        update = f"🌐 Fetching: {url}..."
                    else:
                        update = f"🔧 {tool_name}..."
                    
                    thinking_updates.append(update)
                    stream.set_text(build_display())
                
                # Handle tool result events
                elif event_type == "tool_result":
                    if current_tool:
                        # Mark the tool as done
                        if thinking_updates:
                            thinking_updates[-1] = thinking_updates[-1].replace("...", " ✓")
                            stream.set_text(build_display())
                        current_tool = None
                
                # Handle assistant text output (thinking/explanation)
                elif event_type == "assistant":
                    msg = event.get("message") or {}
                    content = msg.get("content") or []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text", "")
                            if text:
                                thinking_text_buffer += text
                                
                                # Extract a meaningful snippet - only update at sentence boundaries
                                # This makes streaming more natural by waiting for complete thoughts
                                cleaned = thinking_text_buffer.strip().replace("\n", " ")
                                
                                # Find the last complete sentence (ends with . ! or ?)
                                last_sentence_end = -1
                                for i in range(len(cleaned) - 1, -1, -1):
                                    if cleaned[i] in '.!?' and (i == len(cleaned) - 1 or cleaned[i + 1] in ' \n\t'):
                                        last_sentence_end = i
                                        break
                                
                                if last_sentence_end > 0:
                                    # Get the last complete sentence(s) for display
                                    complete_text = cleaned[:last_sentence_end + 1]
                                    # Take the last ~100 chars of complete sentences
                                    if len(complete_text) > 100:
                                        # Find a sentence boundary within the last 100 chars
                                        snippet_start = len(complete_text) - 100
                                        for i in range(snippet_start, len(complete_text)):
                                            if complete_text[i] in '.!?' and i + 1 < len(complete_text):
                                                snippet_start = i + 2
                                                break
                                        snippet = complete_text[snippet_start:].strip()
                                    else:
                                        snippet = complete_text
                                    
                                    # Only update if this is a meaningful change
                                    if snippet and snippet != last_thinking_snippet:
                                        last_thinking_snippet = snippet
                                        stream.set_text(build_display())
                                
                                # Keep track of full text for final summary
                                if len(thinking_text_buffer) > 50:
                                    final_result_text = thinking_text_buffer
                
                elif event_type == "result":
                    res = event.get("result")
                    if isinstance(res, str):
                        final_result_text = res

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
    
    try:
        # Run adaptation with streaming progress
        result = run_adapt_streaming(
            prompt,
            chat_id=message.chat.id,
            reply_to_message_id=message.message_id,
            timeout_seconds=3600,  # 1 hour timeout
        )
        
        append_chat_history("assistant", f"✅ Adaptation complete: {result[:200]}")
        
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
            response = run_agent_ask_streaming(
                prompt_with_context,
                chat_id=message.chat.id,
                existing_message_id=processing_msg.message_id,
                initial_text="🎤 Processing voice message...",
                model="gemini-3-flash",
                timeout_seconds=600,
            )
            append_chat_history("assistant", response)
        finally:
            # Stop typing indicator
            typing_stop.set()
        
    except Exception as e:
        error_msg = f"Error processing voice message: {str(e)}"
        try:
            bot.edit_message_text(error_msg, message.chat.id, processing_msg.message_id)
        except Exception:
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


def run_agent_ask_streaming(
    prompt_with_context: str,
    *,
    chat_id: int,
    reply_to_message_id: int | None = None,
    existing_message_id: int | None = None,
    initial_text: str = "…",
    model: str = "gemini-3-flash",
    timeout_seconds: int = 600,
) -> str:
    """Run the agent CLI in ask mode and stream output by editing one Telegram message.
    
    Shows thinking process and tool operations similar to /adapt streaming.
    """
    stream = TelegramStreamingMessage(
        chat_id,
        reply_to_message_id=reply_to_message_id,
        existing_message_id=existing_message_id,
        initial_text=initial_text,
        min_edit_interval_seconds=1.5,  # Faster updates for thinking
        min_chars_delta=15,  # More responsive updates
    )

    cmd = [
        "agent",
        "--force",
        "--model",
        model,
        "--mode=ask",
        "--output-format",
        "stream-json",
        "--stream-partial-output",
        "--print",
        prompt_with_context,
    ]

    start_time = time.time()
    raw_output = ""
    final_result_text: str | None = None
    stderr_lines: list[str] = []
    thinking_updates: list[str] = []
    thinking_text_buffer: str = ""
    current_tool: str | None = None
    last_thinking_snippet: str = ""
    
    def build_display():
        """Build the display message showing current thinking and actions."""
        lines = []
        
        # Show recent thinking snippet if we have one
        if last_thinking_snippet:
            snippet = last_thinking_snippet[:100]
            if len(last_thinking_snippet) > 100:
                snippet += "..."
            lines.append(f"💭 {snippet}\n")
        
        # Show recent tool actions
        if thinking_updates:
            if lines:
                lines.append("")
            lines.extend(thinking_updates[-6:])
        
        # If we have raw output (the actual response), show it
        if raw_output.strip():
            if lines:
                lines.append("\n---\n")
            lines.append(raw_output)
        
        return "\n".join(lines) if lines else initial_text

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
                
                event_type = event.get("type")
                
                # Handle tool use events - show what the agent is doing
                if event_type == "tool_use":
                    tool_name = event.get("name", "tool")
                    tool_input = event.get("input", {})
                    current_tool = tool_name
                    
                    # Create a brief description of what's happening
                    if tool_name == "Read":
                        path = tool_input.get("path", "")
                        filename = os.path.basename(path) if path else "file"
                        update = f"📖 Reading {filename}..."
                    elif tool_name == "Write":
                        path = tool_input.get("path", "")
                        filename = os.path.basename(path) if path else "file"
                        update = f"✍️ Writing {filename}..."
                    elif tool_name == "StrReplace":
                        path = tool_input.get("path", "")
                        filename = os.path.basename(path) if path else "file"
                        update = f"🔧 Editing {filename}..."
                    elif tool_name == "Shell":
                        cmd_str = tool_input.get("command", "")[:40]
                        update = f"💻 {cmd_str}..."
                    elif tool_name == "Grep":
                        pattern = tool_input.get("pattern", "")[:25]
                        update = f"🔍 Searching: {pattern}..."
                    elif tool_name == "Glob":
                        pattern = tool_input.get("glob_pattern", "")[:25]
                        update = f"📁 Finding: {pattern}..."
                    elif tool_name == "SemanticSearch":
                        query = tool_input.get("query", "")[:40]
                        update = f"🧠 {query}..."
                    elif tool_name == "WebFetch":
                        url = tool_input.get("url", "")[:40]
                        update = f"🌐 Fetching: {url}..."
                    else:
                        update = f"🔧 {tool_name}..."
                    
                    thinking_updates.append(update)
                    stream.set_text(build_display())
                
                # Handle tool result events
                elif event_type == "tool_result":
                    if current_tool:
                        # Mark the tool as done
                        if thinking_updates:
                            thinking_updates[-1] = thinking_updates[-1].replace("...", " ✓")
                            stream.set_text(build_display())
                        current_tool = None
                
                # Handle assistant text output (thinking/explanation)
                elif event_type == "assistant":
                    msg = event.get("message") or {}
                    content = msg.get("content") or []
                    parts: list[str] = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text", "")
                            if text:
                                parts.append(text)
                                thinking_text_buffer += text
                    
                    candidate = "".join(parts)
                    if candidate:
                        # With --stream-partial-output this is typically a delta,
                        # but handle both delta and "full so far" outputs safely.
                        delta = candidate
                        if candidate.startswith(raw_output):
                            delta = candidate[len(raw_output):]
                        if delta:
                            raw_output += delta
                        
                        # Extract a meaningful snippet for thinking display
                        # Only update at sentence boundaries for natural streaming
                        cleaned = thinking_text_buffer.strip().replace("\n", " ")
                        
                        # Find the last complete sentence
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
                
                elif event_type == "result":
                    res = event.get("result")
                    if isinstance(res, str):
                        final_result_text = res

            # If process exited, loop briefly to drain remaining output.
            if proc.poll() is not None and q.empty():
                continue
    finally:
        stream.finalize()
        processes.untrack(proc)

    if processes.pop_cancelled(proc.pid):
        msg = "Cancelled."
        stream.set_text(msg)
        return msg

    if final_result_text:
        # Ensure we finish with the full terminal result if available.
        if raw_output and final_result_text.startswith(raw_output):
            missing = final_result_text[len(raw_output):]
            if missing:
                raw_output += missing
        elif not raw_output:
            raw_output = final_result_text

    output = raw_output.strip()

    if timed_out:
        msg = "Request timed out."
        stream.set_text(msg)
        return msg

    if not output:
        stderr_text = "".join(stderr_lines).strip()
        msg = stderr_text if stderr_text else "No response from agent."
        stream.set_text(msg)
        return msg

    # Final display: just the clean response without thinking artifacts
    stream.set_text(output)
    return output


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
        response = run_agent_ask_streaming(
            prompt_with_context,
            chat_id=message.chat.id,
            reply_to_message_id=message.message_id,
            initial_text="…",
            model="gemini-3-flash",
            timeout_seconds=300,
        )
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Agent completed in {elapsed:.1f}s (streamed)")
        append_chat_history("assistant", response)
        logger.info("Response streamed to Telegram")
        
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
