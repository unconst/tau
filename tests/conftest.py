import os
import sys
import types
import importlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


class FakeBot:
    """Minimal TeleBot replacement used for e2e tests (no network)."""

    def __init__(self, token: str | None = None):
        self.token = token
        self.sent_messages: list[dict] = []
        self.edits: list[dict] = []
        self.chat_actions: list[dict] = []
        self.stopped_polling = False
        self._next_message_id = 1

    def reset(self):
        self.sent_messages.clear()
        self.edits.clear()
        self.chat_actions.clear()
        self.stopped_polling = False
        self._next_message_id = 1

    # --- telebot API surface (subset) ---
    def message_handler(self, *, commands=None, func=None, content_types=None):
        # Decorator used at import time by tau; we don't need registration for tests.
        def decorator(fn):
            return fn

        return decorator

    def send_message(self, chat_id, text, reply_to_message_id=None):
        message_id = self._next_message_id
        self._next_message_id += 1
        self.sent_messages.append(
            {
                "chat_id": chat_id,
                "text": text,
                "reply_to_message_id": reply_to_message_id,
                "message_id": message_id,
            }
        )
        return SimpleNamespace(
            message_id=message_id,
            chat=SimpleNamespace(id=chat_id),
            text=text,
        )

    def reply_to(self, message, text):
        chat_id = message.chat.id
        reply_to_message_id = getattr(message, "message_id", None)
        return self.send_message(chat_id, text, reply_to_message_id=reply_to_message_id)

    def edit_message_text(self, text, chat_id, message_id):
        self.edits.append({"chat_id": chat_id, "message_id": message_id, "text": text})
        return True

    def send_chat_action(self, chat_id, action):
        self.chat_actions.append({"chat_id": chat_id, "action": action})
        return True

    def stop_polling(self):
        self.stopped_polling = True

    def polling(self):
        # Not used in tests.
        return None


class FakeChat:
    def __init__(self, chat_id: int):
        self.id = chat_id


class FakeVoice:
    def __init__(self, file_id: str):
        self.file_id = file_id


class FakeMessage:
    """Minimal message object compatible with tau handlers."""

    def __init__(
        self,
        text: str | None,
        *,
        chat_id: int = 123,
        message_id: int = 1,
        content_type: str = "text",
        voice_file_id: str | None = None,
    ):
        self.text = text
        self.chat = FakeChat(chat_id)
        self.message_id = message_id
        self.content_type = content_type
        if voice_file_id is not None:
            self.voice = FakeVoice(voice_file_id)


def _install_fake_modules(monkeypatch):
    """Install fake third-party modules before importing tau."""
    # Fake telebot
    telebot_mod = types.ModuleType("telebot")
    telebot_mod.TeleBot = FakeBot
    monkeypatch.setitem(sys.modules, "telebot", telebot_mod)

    # Fake dotenv (avoid reading local .env in tests)
    dotenv_mod = types.ModuleType("dotenv")

    def load_dotenv(*args, **kwargs):
        return False

    dotenv_mod.load_dotenv = load_dotenv
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_mod)

    # Fake openai (avoid network/client init)
    openai_mod = types.ModuleType("openai")

    class OpenAI:
        def __init__(self, *args, **kwargs):
            self.audio = SimpleNamespace(
                transcriptions=SimpleNamespace(
                    create=lambda *a, **k: SimpleNamespace(text="")
                )
            )

    openai_mod.OpenAI = OpenAI
    monkeypatch.setitem(sys.modules, "openai", openai_mod)


@dataclass(frozen=True)
class TauTestApp:
    tau: types.ModuleType
    telegram: types.ModuleType
    agent: types.ModuleType
    bot: FakeBot
    workspace: Path


@pytest.fixture()
def fake_bot():
    return FakeBot(token="123456:ABCDEF")


@pytest.fixture()
def make_message():
    def _make(text: str | None, **kwargs):
        return FakeMessage(text, **kwargs)

    return _make


@pytest.fixture()
def tau_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_bot: FakeBot):
    """
    Import tau with Telegram/OpenAI mocked, then patch all filesystem paths to a temp workspace.
    """
    _install_fake_modules(monkeypatch)

    # Ensure token is present so tau.telegram doesn't raise.
    monkeypatch.setenv("TAU_BOT_TOKEN", "123456:ABCDEF")

    # Import modules under test (lazy; tests should not import tau at module import time).
    tau = importlib.import_module("tau")
    telegram = importlib.import_module("tau.telegram")
    agent = importlib.import_module("tau.agent")

    # Temp workspace layout
    workspace = Path(tmp_path)
    (workspace / "context" / "tasks").mkdir(parents=True, exist_ok=True)
    (workspace / "context" / "logs").mkdir(parents=True, exist_ok=True)

    # Patch telegram paths and bot
    telegram.WORKSPACE = str(workspace)
    telegram.CHAT_ID_FILE = str(workspace / "chat_id.txt")
    telegram.CHAT_HISTORY_FILE = str(workspace / "context" / "CHAT.md")
    telegram.bot = fake_bot

    # Patch tau module globals that were imported from telegram/agent
    tau.WORKSPACE = str(workspace)
    tau.LOG_FILE = str(workspace / "context" / "logs" / "tau.log")
    tau.MEMORY_FILE = str(workspace / "context" / "tasks" / "memory.md")
    tau.bot = fake_bot

    # Patch agent paths (functions read these globals from tau.agent)
    agent.WORKSPACE = str(workspace)
    agent.TASKS_DIR = str(workspace / "context" / "tasks")
    agent.MEMORY_FILE = str(workspace / "context" / "tasks" / "memory.md")
    agent.IDENTITY_FILE = str(workspace / "context" / "IDENTITY.md")
    agent.MEMORY_SYSTEM_FILE = str(workspace / "context" / "MEMORY-SYSTEM.md")
    agent.CHAT_HISTORY_FILE = str(workspace / "context" / "CHAT.md")

    # tau.add_task uses TASKS_DIR imported into tau at import time
    tau.TASKS_DIR = agent.TASKS_DIR

    # Reset global state between tests
    tau._cron_jobs = []
    tau._next_cron_id = 1
    fake_bot.reset()

    # Never touch the real git repo in tests
    def _no_git_commit(*args, **kwargs):
        return None

    agent.git_commit_changes = _no_git_commit
    tau.git_commit_changes = _no_git_commit

    return TauTestApp(tau=tau, telegram=telegram, agent=agent, bot=fake_bot, workspace=workspace)

