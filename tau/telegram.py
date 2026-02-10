"""Shared Telegram bot for Tau agent."""

import os
import json
import time
import logging
from pathlib import Path

# Load .env file before accessing any environment variables
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(WORKSPACE, ".env"))

import telebot
from datetime import datetime

logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("TAU_BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("TAU_BOT_TOKEN environment variable is required. Create a .env file with TAU_BOT_TOKEN=your_token")
CHAT_ID_FILE = os.path.join(WORKSPACE, "chat_id.txt")
CHAT_HISTORY_FILE = os.path.join(WORKSPACE, "context", "CHAT.md")
CHATS_DIR = os.path.join(WORKSPACE, "context", "chats")

bot = telebot.TeleBot(BOT_TOKEN)

# ---------------------------------------------------------------------------
# Owner authentication
# ---------------------------------------------------------------------------
OWNER_ID: str | None = os.getenv("OWNER_ID")


def is_owner(message) -> bool:
    """Check if the message sender is the bot owner."""
    if OWNER_ID is None:
        return False
    return str(message.from_user.id) == str(OWNER_ID)


def is_private_chat(message) -> bool:
    """Check if the message is from a private (1:1) chat."""
    return message.chat.type == "private"


def is_group_chat(message) -> bool:
    """Check if the message is from a group or supergroup chat."""
    return message.chat.type in ("group", "supergroup")


def bootstrap_owner(message) -> bool:
    """Register the first user who DMs the bot as the owner.

    Returns True if bootstrap occurred (caller should proceed), False if
    the message should be ignored (e.g. a group message with no owner set).
    """
    global OWNER_ID
    if OWNER_ID is not None:
        return False  # already bootstrapped

    # Only bootstrap from a private chat
    if not is_private_chat(message):
        return False

    OWNER_ID = str(message.from_user.id)

    # Persist to .env so it survives restarts
    env_path = os.path.join(WORKSPACE, ".env")
    with open(env_path, "a") as f:
        f.write(f"\nOWNER_ID={OWNER_ID}\n")

    logger.info(f"Owner bootstrapped: user_id={OWNER_ID}")
    return True


def is_main_chat(message) -> bool:
    """Check if the message is from the owner's main chat (the one used for /start)."""
    main_chat_id = get_chat_id()
    if main_chat_id is None:
        # No main chat saved yet — allow private chats so /start can bootstrap
        return is_private_chat(message)
    return message.chat.id == main_chat_id


def authorize(message, *, require_owner: bool = True, require_main_chat: bool = True) -> bool:
    """Central authorization check for every handler.

    Returns True if the handler should proceed, False to silently skip.

    Behavior:
    - If no OWNER_ID: bootstrap from a private DM, ignore everything else.
    - Non-owner: silently ignore.
    - Owner outside main chat: silently ignore (logging happens separately).
    - Owner in main chat: allowed.
    """
    # Bootstrap mode: no owner registered yet
    if OWNER_ID is None:
        if bootstrap_owner(message):
            return True  # first DM user is now the owner, proceed
        return False  # ignore until owner is set

    if not require_owner:
        return True

    if not is_owner(message):
        return False

    # Commands and responses only in the owner's main chat
    if require_main_chat and not is_main_chat(message):
        return False

    return True


def think(msg: str):
    """Send brief thinking update to Telegram."""
    chat_id = get_chat_id()
    if chat_id:
        try:
            bot.send_message(chat_id, f"💭 {msg[:60]}")
        except Exception:
            pass  # Don't crash if Telegram fails


def split_message(msg: str, max_length: int = 4000) -> list[str]:
    """Split a long message into chunks that fit within Telegram's limit.
    
    Tries to split at natural boundaries (newlines, sentences, words) to maintain readability.
    """
    if len(msg) <= max_length:
        return [msg]
    
    chunks = []
    remaining = msg
    
    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break
        
        # Find a good split point within the limit
        chunk = remaining[:max_length]
        
        # Try to split at double newline (paragraph break)
        split_idx = chunk.rfind('\n\n')
        if split_idx > max_length // 2:
            chunks.append(remaining[:split_idx].rstrip())
            remaining = remaining[split_idx:].lstrip()
            continue
        
        # Try to split at single newline
        split_idx = chunk.rfind('\n')
        if split_idx > max_length // 2:
            chunks.append(remaining[:split_idx].rstrip())
            remaining = remaining[split_idx:].lstrip()
            continue
        
        # Try to split at sentence boundary (. ! ?)
        for sep in ['. ', '! ', '? ']:
            split_idx = chunk.rfind(sep)
            if split_idx > max_length // 2:
                chunks.append(remaining[:split_idx + 1].rstrip())
                remaining = remaining[split_idx + 1:].lstrip()
                break
        else:
            # Try to split at word boundary (space)
            split_idx = chunk.rfind(' ')
            if split_idx > max_length // 2:
                chunks.append(remaining[:split_idx].rstrip())
                remaining = remaining[split_idx:].lstrip()
            else:
                # Hard split if no good boundary found
                chunks.append(remaining[:max_length])
                remaining = remaining[max_length:]
    
    return chunks


def _is_message_not_modified_error(exc: Exception) -> bool:
    """Best-effort check for Telegram 'message is not modified' edit errors."""
    try:
        return "message is not modified" in str(exc).lower()
    except Exception:
        return False


class TelegramStreamingMessage:
    """Stream text into Telegram by editing a single message.

    - Edits are throttled to avoid Telegram rate limits.
    - If the text exceeds Telegram's message limit, it rolls over to a new
      message and continues streaming by editing the latest message.
    """

    def __init__(
        self,
        chat_id: int,
        *,
        reply_to_message_id: int | None = None,
        existing_message_id: int | None = None,
        initial_text: str = "🤔",
        max_length: int = 4000,
        min_edit_interval_seconds: float = 1.0,
        min_chars_delta: int = 12,
    ):
        self.chat_id = chat_id
        self.reply_to_message_id = reply_to_message_id
        self.initial_text = initial_text
        self.max_length = max_length
        self.min_edit_interval_seconds = min_edit_interval_seconds
        self.min_chars_delta = min_chars_delta

        self._message_ids: list[int] = []
        self._current_message_id: int | None = None
        self._current_text: str = ""
        self._last_edited_text: str = ""
        self._last_edit_at: float = 0.0

        if existing_message_id is not None:
            self._current_message_id = existing_message_id
            self._message_ids.append(existing_message_id)
            # Assume the existing message already contains initial_text.
            self._last_edited_text = initial_text
            self._last_edit_at = time.time()
        else:
            try:
                msg = bot.send_message(
                    chat_id,
                    initial_text,
                    reply_to_message_id=reply_to_message_id,
                )
                self._current_message_id = msg.message_id
                self._message_ids.append(msg.message_id)
                self._last_edited_text = initial_text
                self._last_edit_at = time.time()
            except Exception:
                # If we can't create the placeholder message, keep state but
                # allow the caller to continue without crashing.
                self._current_message_id = None

    def append(self, delta: str):
        if not delta or self._current_message_id is None:
            return
        self._current_text += delta
        self._flush_overflow()
        self._maybe_edit(force=False)

    def finalize(self):
        """Force a final edit so the latest message is up to date."""
        if self._current_message_id is None:
            return
        self._current_text = self._current_text.rstrip()
        self._maybe_edit(force=True)

    def set_text(self, text: str):
        """Replace the current (latest) message text."""
        if self._current_message_id is None:
            return
        self._current_text = text
        # If caller sets text directly, respect Telegram limits.
        self._flush_overflow()
        self._maybe_edit(force=True)

    def _maybe_edit(self, *, force: bool):
        if self._current_message_id is None:
            return

        text_to_send = self._current_text if self._current_text else self.initial_text
        if not text_to_send:
            text_to_send = " "

        if text_to_send == self._last_edited_text:
            return

        # If we're still on the placeholder, allow the first real output to
        # appear immediately (even if within the normal throttle window).
        force_edit = force or (self._last_edited_text == self.initial_text and bool(self._current_text))

        if not force_edit:
            now = time.time()
            if now - self._last_edit_at < self.min_edit_interval_seconds:
                return
            if abs(len(text_to_send) - len(self._last_edited_text)) < self.min_chars_delta:
                return

        try:
            bot.edit_message_text(text_to_send, self.chat_id, self._current_message_id)
            self._last_edited_text = text_to_send
            self._last_edit_at = time.time()
        except Exception as e:
            if _is_message_not_modified_error(e):
                return
            # Don't crash the bot if Telegram edit fails.
            return

    def _flush_overflow(self):
        """If current text exceeds limit, send new messages and continue streaming."""
        if self._current_message_id is None:
            return

        while len(self._current_text) > self.max_length:
            chunks = split_message(self._current_text, max_length=self.max_length)
            if len(chunks) <= 1:
                break

            first, rest = chunks[0], chunks[1:]

            # Finalize the current message with the first chunk.
            self._current_text = first
            self._maybe_edit(force=True)

            # Send any fully-formed middle chunks.
            for chunk in rest[:-1]:
                try:
                    msg = bot.send_message(self.chat_id, chunk)
                    self._message_ids.append(msg.message_id)
                except Exception:
                    # If sending fails, stop attempting further rollovers.
                    return

            # The last chunk becomes the new "current" message.
            last_chunk = rest[-1]
            try:
                msg = bot.send_message(self.chat_id, last_chunk)
                self._message_ids.append(msg.message_id)
                self._current_message_id = msg.message_id
                self._current_text = last_chunk
                self._last_edited_text = last_chunk
                self._last_edit_at = time.time()
            except Exception:
                return


def notify(msg: str):
    """Send a notification to Telegram (no emoji prefix).
    
    If the message is too long, it will be split into multiple messages.
    """
    chat_id = get_chat_id()
    if chat_id:
        try:
            chunks = split_message(msg)
            for i, chunk in enumerate(chunks):
                # Add continuation indicator for multi-part messages
                if len(chunks) > 1:
                    part_indicator = f"({i + 1}/{len(chunks)}) " if i > 0 else ""
                    chunk = part_indicator + chunk
                bot.send_message(chat_id, chunk)
        except Exception:
            pass


def get_chat_id() -> int | None:
    """Get stored chat ID."""
    if os.path.exists(CHAT_ID_FILE):
        try:
            return int(open(CHAT_ID_FILE).read().strip())
        except (ValueError, IOError):
            return None
    return None


def save_chat_id(chat_id: int):
    """Save chat ID for agent to use."""
    open(CHAT_ID_FILE, "w").write(str(chat_id))


def _chat_dir_for(chat_id: int) -> str:
    """Return the directory path for a specific chat's storage."""
    return os.path.join(CHATS_DIR, str(chat_id))


def _chat_file_for(chat_id: int) -> str:
    """Return the CHAT.md path for a specific chat."""
    return os.path.join(_chat_dir_for(chat_id), "CHAT.md")


def save_chat_metadata(message):
    """Persist chat metadata (title, type, etc.) to meta.json.

    Called once per chat on first encounter and updated on subsequent messages.
    """
    chat = message.chat
    chat_dir = _chat_dir_for(chat.id)
    os.makedirs(chat_dir, exist_ok=True)
    meta_path = os.path.join(chat_dir, "meta.json")

    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    meta.update({
        "chat_id": chat.id,
        "type": chat.type,
        "title": chat.title or getattr(chat, "first_name", None) or str(chat.id),
        "username": getattr(chat, "username", None),
        "updated_at": datetime.now().isoformat(),
    })

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def append_chat_history(role: str, content: str, chat_id: int | None = None, username: str | None = None):
    """Append a message to the per-chat history file.

    Args:
        role: Either 'user' or 'assistant'
        content: The message content
        chat_id: Telegram chat ID. If None, falls back to the owner's saved chat ID.
        username: Optional display name for the sender (useful in groups).
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Determine which chat file to write to
    if chat_id is not None:
        chat_file = _chat_file_for(chat_id)
    else:
        # Fallback: try the owner's stored chat id, or the legacy file
        owner_chat = get_chat_id()
        if owner_chat:
            chat_file = _chat_file_for(owner_chat)
        else:
            chat_file = CHAT_HISTORY_FILE

    os.makedirs(os.path.dirname(chat_file), exist_ok=True)

    # Ensure file exists with header
    if not os.path.exists(chat_file):
        with open(chat_file, "w") as f:
            f.write("# Telegram Chat History\n\n")
            f.write("This file contains the conversation history for this chat.\n")
            f.write("It is automatically updated as new messages are sent and received.\n\n")
            f.write("---\n\n")

    # Build the header — include username when available (group chats)
    sender = role.upper()
    if username and role.lower() == "user":
        sender = f"{role.upper()} ({username})"

    # Append new message
    with open(chat_file, "a") as f:
        f.write(f"## {sender} - {timestamp}\n\n")
        f.write(f"{content}\n\n")
        f.write("---\n\n")


def _tail_lines(filepath: str, n: int) -> list[str]:
    """Read the last *n* lines of a file efficiently using seek from the end.

    For small files this just reads the whole file.  For large files it reads
    backward in chunks until enough newlines are found, avoiding a full load.
    """
    CHUNK = 8192
    try:
        with open(filepath, "rb") as f:
            f.seek(0, 2)  # seek to end
            size = f.tell()
            if size <= CHUNK:
                # Small file — just read it all
                f.seek(0)
                return f.read().decode("utf-8", errors="replace").splitlines(keepends=True)

            # Large file — read backward in chunks
            data = b""
            pos = size
            lines_found = 0
            while pos > 0 and lines_found <= n:
                read_size = min(CHUNK, pos)
                pos -= read_size
                f.seek(pos)
                chunk = f.read(read_size)
                data = chunk + data
                lines_found = data.count(b"\n")

            all_lines = data.decode("utf-8", errors="replace").splitlines(keepends=True)
            return all_lines[-n:] if len(all_lines) > n else all_lines
    except Exception:
        return []


def get_chat_history(max_lines: int = 100, chat_id: int | None = None) -> str:
    """Get recent chat history for a specific chat.

    Args:
        max_lines: Maximum number of lines to return (default 100)
        chat_id: Telegram chat ID. If None, uses the owner's saved chat.

    Returns:
        Recent chat history, truncated to max_lines.
        Uses an efficient tail-read to avoid loading the entire file.
    """
    if chat_id is not None:
        target_file = _chat_file_for(chat_id)
    else:
        owner_chat = get_chat_id()
        if owner_chat:
            target_file = _chat_file_for(owner_chat)
        else:
            target_file = CHAT_HISTORY_FILE

    if not os.path.exists(target_file):
        return ""

    recent = _tail_lines(target_file, max_lines)
    if not recent:
        return ""

    return "".join(recent)


# ---------------------------------------------------------------------------
# Multi-chat utilities (for agent use)
# ---------------------------------------------------------------------------

def list_chats() -> list[dict]:
    """List all known chats with their metadata.

    Returns a list of dicts with keys: chat_id, type, title, username, updated_at.
    """
    chats = []
    if not os.path.isdir(CHATS_DIR):
        return chats

    for entry in os.listdir(CHATS_DIR):
        meta_path = os.path.join(CHATS_DIR, entry, "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                chats.append(meta)
            except Exception:
                continue
    return chats


def get_chat_history_for(chat_id: int, max_lines: int = 100) -> str:
    """Get chat history for an arbitrary chat by ID."""
    return get_chat_history(max_lines=max_lines, chat_id=chat_id)


def send_to_chat(chat_id: int, text: str):
    """Send a message to a specific Telegram chat (group or private).

    The bot must be a member of the target chat.
    """
    chunks = split_message(text)
    for chunk in chunks:
        bot.send_message(chat_id, chunk)
