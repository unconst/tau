"""Shared Telegram bot for Tau agent."""

import os
import telebot
from datetime import datetime
from pathlib import Path

BOT_TOKEN = os.getenv("TAU_BOT_TOKEN", "8355192805:AAFnd-QRdqOdTnxitWhTqZDGZQCnBi8rEpI")
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


def notify(msg: str):
    """Send a notification to Telegram (no emoji prefix)."""
    chat_id = get_chat_id()
    if chat_id:
        try:
            bot.send_message(chat_id, msg[:4000])
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


def get_chat_history() -> str:
    """Get the full chat history from context/CHAT.md."""
    if os.path.exists(CHAT_HISTORY_FILE):
        return open(CHAT_HISTORY_FILE).read()
    return ""
