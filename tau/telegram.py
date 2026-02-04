"""Shared Telegram bot for Tau agent."""

import os
import time
from pathlib import Path

# Load .env file before accessing any environment variables
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(WORKSPACE, ".env"))

import telebot
from datetime import datetime

BOT_TOKEN = os.getenv("TAU_BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("TAU_BOT_TOKEN environment variable is required. Create a .env file with TAU_BOT_TOKEN=your_token")
CHAT_ID_FILE = os.path.join(WORKSPACE, "chat_id.txt")
CHAT_HISTORY_FILE = os.path.join(WORKSPACE, "context", "CHAT.md")

bot = telebot.TeleBot(BOT_TOKEN)


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
        initial_text: str = "…",
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


def append_chat_history(role: str, content: str):
    """Append a message to context/CHAT.md history file.
    
    Args:
        role: Either 'user' or 'assistant'
        content: The message content
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Ensure file exists with header
    if not os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "w") as f:
            f.write("# Telegram Chat History\n\n")
            f.write("This file contains the complete history of all Telegram conversations.\n")
            f.write("It is automatically updated as new messages are sent and received.\n\n")
            f.write("---\n\n")
    
    # Append new message
    with open(CHAT_HISTORY_FILE, "a") as f:
        f.write(f"## {role.upper()} - {timestamp}\n\n")
        f.write(f"{content}\n\n")
        f.write("---\n\n")


def get_chat_history(max_lines: int = 100) -> str:
    """Get recent chat history from context/CHAT.md.
    
    Args:
        max_lines: Maximum number of lines to return (default 100)
    
    Returns:
        Recent chat history, truncated to max_lines
    """
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE) as f:
            lines = f.readlines()
        
        if len(lines) <= max_lines:
            return "".join(lines)
        
        # Return header (first 10 lines + last (max_lines - 10) lines
        header = lines[:10]
        recent = lines[-(max_lines - 10):]
        return "".join(header) + "\n[... earlier messages truncated ...]\n\n" + "".join(recent)
    return ""
