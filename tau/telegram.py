"""Shared Telegram bot for Tau agent."""

import os
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
