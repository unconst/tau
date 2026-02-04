"""Track and terminate active subprocesses (e.g. Cursor agent CLI).

This lets Telegram commands like /clear stop any in-flight agent work.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ProcessRecord:
    pid: int
    label: str
    started_at: float
    cmd: tuple[str, ...] | None
    own_process_group: bool


_lock = threading.Lock()
_tracked: dict[int, tuple[subprocess.Popen[str], ProcessRecord]] = {}
# pid -> timestamp; used to surface "Cancelled." instead of partial output
_cancelled: dict[int, float] = {}


def _prune_cancelled(*, now: float | None = None, max_age_seconds: float = 600.0) -> None:
    """Prevent unbounded growth and reduce pid-reuse false positives."""
    if max_age_seconds <= 0:
        return
    if now is None:
        now = time.time()
    cutoff = now - max_age_seconds
    stale = [pid for pid, ts in _cancelled.items() if ts < cutoff]
    for pid in stale:
        _cancelled.pop(pid, None)


def track(
    proc: subprocess.Popen[str],
    *,
    label: str,
    cmd: Sequence[str] | None = None,
    own_process_group: bool = False,
) -> ProcessRecord:
    """Register a subprocess so it can be cancelled later."""
    now = time.time()
    record = ProcessRecord(
        pid=proc.pid,
        label=label,
        started_at=now,
        cmd=tuple(cmd) if cmd is not None else None,
        own_process_group=own_process_group,
    )
    with _lock:
        _tracked[proc.pid] = (proc, record)
        _prune_cancelled(now=now)
    return record


def untrack(proc: subprocess.Popen[str] | None) -> None:
    """Unregister a subprocess (best-effort)."""
    if proc is None:
        return
    with _lock:
        _tracked.pop(proc.pid, None)
        # Intentionally do NOT remove from _cancelled here; the owning caller
        # should call pop_cancelled(pid) to observe cancellation.


def list_active(*, label_prefix: str | None = None) -> list[ProcessRecord]:
    """List currently-running tracked subprocesses."""
    with _lock:
        items = list(_tracked.items())

    active: list[ProcessRecord] = []
    dead_pids: list[int] = []
    for pid, (proc, record) in items:
        if label_prefix and not record.label.startswith(label_prefix):
            continue
        try:
            if proc.poll() is None:
                active.append(record)
            else:
                dead_pids.append(pid)
        except Exception:
            dead_pids.append(pid)

    if dead_pids:
        with _lock:
            for pid in dead_pids:
                _tracked.pop(pid, None)

    return active


def pop_cancelled(pid: int) -> bool:
    """Return True once if pid was cancelled via terminate_all()."""
    with _lock:
        ts = _cancelled.pop(pid, None)
        if ts is None:
            _prune_cancelled()
            return False
        _prune_cancelled()
        return True


def _signal(proc: subprocess.Popen[str], record: ProcessRecord, sig: int) -> None:
    try:
        if proc.poll() is not None:
            return
    except Exception:
        return

    try:
        if os.name == "posix" and record.own_process_group:
            # When started with start_new_session=True, pid is also the pgid.
            os.killpg(proc.pid, sig)
        else:
            proc.send_signal(sig)
    except ProcessLookupError:
        return
    except Exception:
        # Fallback to Popen helpers.
        try:
            if sig == signal.SIGKILL:
                proc.kill()
            else:
                proc.terminate()
        except Exception:
            return


def terminate_all(*, label_prefix: str = "agent:", timeout_seconds: float = 2.0) -> list[ProcessRecord]:
    """Terminate all tracked processes whose label starts with label_prefix.

    Returns the list of processes that were targeted.
    """
    with _lock:
        items = list(_tracked.items())

    targets: list[tuple[subprocess.Popen[str], ProcessRecord]] = []
    for _, (proc, record) in items:
        if label_prefix and not record.label.startswith(label_prefix):
            continue
        try:
            if proc.poll() is None:
                targets.append((proc, record))
        except Exception:
            continue

    if not targets:
        return []

    now = time.time()
    with _lock:
        for proc, _ in targets:
            _cancelled[proc.pid] = now
        _prune_cancelled(now=now)

    # First try a graceful stop.
    for proc, record in targets:
        _signal(proc, record, signal.SIGTERM)

    deadline = time.time() + max(timeout_seconds, 0.0)
    while time.time() < deadline:
        if all(proc.poll() is not None for proc, _ in targets):
            break
        time.sleep(0.05)

    # Escalate remaining.
    for proc, record in targets:
        try:
            if proc.poll() is None:
                _signal(proc, record, signal.SIGKILL)
        except Exception:
            pass

    # Best-effort cleanup of finished processes.
    with _lock:
        for proc, _ in targets:
            try:
                if proc.poll() is not None:
                    _tracked.pop(proc.pid, None)
            except Exception:
                _tracked.pop(proc.pid, None)

    return [record for _, record in targets]

