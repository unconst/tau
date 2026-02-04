import json
from pathlib import Path

import pytest


def test_task_creates_files_and_replies(tau_app, make_message):
    app = tau_app

    msg = make_message("/task do something", chat_id=111, message_id=10)
    app.tau.add_task(msg)

    # Reply content
    assert any(
        m["chat_id"] == 111 and "✅ Task added (task-1): do something" in m["text"]
        for m in app.bot.sent_messages
    )

    # Files created inside temp workspace
    task_dir = app.workspace / "context" / "tasks" / "task-1"
    assert (task_dir / "task.md").exists()
    assert (task_dir / "memory.md").exists()

    # Chat history appended
    chat_file = app.workspace / "context" / "CHAT.md"
    assert chat_file.exists()
    content = chat_file.read_text()
    assert "/task do something" in content
    assert "✅ Task added (task-1): do something" in content


def test_status_reports_tasks_and_recent_activity(tau_app, make_message):
    app = tau_app

    # Create one task
    app.tau.add_task(make_message("/task check status", chat_id=222, message_id=1))

    # Add some high-level memory content
    memory_file = Path(app.tau.MEMORY_FILE)
    memory_file.parent.mkdir(parents=True, exist_ok=True)
    memory_file.write_text("# Memory\n\n### 2026-01-01 00:00\nDid a thing.\n")

    app.bot.reset()
    app.tau.get_status(make_message("/status", chat_id=222, message_id=2))

    replies = [m["text"] for m in app.bot.sent_messages if m["chat_id"] == 222]
    assert replies, "expected a /status reply"
    body = replies[-1]
    assert "Total tasks: 1" in body
    assert "Incomplete: 1" in body
    assert "Active tasks:" in body
    assert "check status" in body
    assert "📝 Recent activity:" in body
    assert "Did a thing." in body


def test_cron_lifecycle_create_list_remove(tau_app, make_message):
    app = tau_app

    app.tau.add_cron(make_message("/cron 5min check the weather", chat_id=333, message_id=1))
    app.tau.list_crons(make_message("/crons", chat_id=333, message_id=2))
    app.tau.remove_cron(make_message("/uncron 1", chat_id=333, message_id=3))
    app.tau.list_crons(make_message("/crons", chat_id=333, message_id=4))

    texts = [m["text"] for m in app.bot.sent_messages if m["chat_id"] == 333]
    assert any("✅ Cron #1 created: every 5min" in t for t in texts)
    assert any("Active cron jobs:" in t and "#1: every 5min" in t for t in texts)
    assert any("✅ Cron #1 removed." in t for t in texts)
    assert any("No active cron jobs." in t for t in texts)


class _FakePopen:
    def __init__(self, stdout_lines, stderr_lines=None):
        self.stdout = stdout_lines
        self.stderr = stderr_lines or []
        self._returncode = 0

    def poll(self):
        return self._returncode

    def terminate(self):
        self._returncode = -15

    def kill(self):
        self._returncode = -9

    def wait(self, timeout=None):
        return self._returncode


def test_handle_message_streams_agent_and_appends_chat_history(tau_app, make_message, monkeypatch):
    app = tau_app

    # Make typing thread a no-op
    monkeypatch.setattr(app.tau, "send_typing_action", lambda chat_id, stop_event: None)

    stdout_events = [
        json.dumps(
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Hello, "}]} }
        )
        + "\n",
        json.dumps(
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "world!"}]} }
        )
        + "\n",
        json.dumps({"type": "result", "result": "Hello, world!"}) + "\n",
    ]

    def fake_popen(*args, **kwargs):
        return _FakePopen(stdout_events, [])

    monkeypatch.setattr(app.tau.subprocess, "Popen", fake_popen)

    user_msg = make_message("hi", chat_id=444, message_id=42)
    app.tau.handle_message(user_msg)

    # run_agent_ask_streaming should create a placeholder message (initial_text)
    assert any(m["chat_id"] == 444 and m["text"] == "…" for m in app.bot.sent_messages)

    # And it should edit the message at least once (streaming)
    assert app.bot.edits, "expected Telegram edits during streaming"
    assert any("Hello, world!" in e["text"] for e in app.bot.edits)

    # Chat history should include both user and assistant
    chat_file = app.workspace / "context" / "CHAT.md"
    content = chat_file.read_text()
    assert "## USER" in content
    assert "hi" in content
    assert "## ASSISTANT" in content
    assert "Hello, world!" in content

