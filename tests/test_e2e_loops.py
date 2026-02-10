import time as _time
from pathlib import Path
from types import SimpleNamespace


def test_run_cron_loop_executes_due_job_and_sends_message(tau_app, monkeypatch):
    app = tau_app

    # Create one due cron job
    app.tau._cron_jobs.append(
        {
            "id": 1,
            "interval_seconds": 60,
            "prompt": "do the thing",
            "next_run": 0,  # due immediately
            "chat_id": 555,
        }
    )

    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout="cron-output\n", stderr="", returncode=0)

    monkeypatch.setattr(app.tau.subprocess, "run", fake_run)

    stop_event = app.tau.threading.Event()

    def fake_sleep(_seconds):
        stop_event.set()
        return None

    monkeypatch.setattr(_time, "sleep", fake_sleep)

    app.tau.run_cron_loop(stop_event)

    # Sent message to chat
    assert any(
        m["chat_id"] == 555 and m["text"].startswith("⏰ Cron #1:\ncron-output")
        for m in app.bot.sent_messages
    )

    # Chat history appended
    chat_file = app.workspace / "context" / "CHAT.md"
    assert chat_file.exists()
    content = chat_file.read_text()
    assert "[cron #1]: cron-output" in content


def test_run_loop_processes_one_task_and_cleans_up(tau_app, monkeypatch):
    app = tau_app

    # Create a single task in the temp workspace
    task_dir = Path(app.agent.TASKS_DIR) / "task-1"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.md").write_text("## do work\n- created: 2026-01-01 00:00\n")

    # Make the agent produce a completion marker
    monkeypatch.setattr(app.agent, "run_agent", lambda prompt: "Did work. Task complete")

    stop_event = app.tau.threading.Event()

    def fake_sleep(_seconds):
        stop_event.set()
        return None

    monkeypatch.setattr(_time, "sleep", fake_sleep)

    app.agent.run_loop(stop_event)

    # Task should be marked complete in task.md
    task_md = (task_dir / "task.md").read_text().lower()
    assert "- status: complete" in task_md

    # memory.md should have been cleaned up after completion
    assert not (task_dir / "memory.md").exists()

    # High-level memory should exist and contain a summary
    high_level = Path(app.agent.MEMORY_FILE)
    assert high_level.exists()
    assert "did work" in high_level.read_text().lower()

