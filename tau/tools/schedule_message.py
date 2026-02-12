#!/usr/bin/env python3
"""Schedule a message or task for later.

Usage:
    python -m tau.tools.schedule_message --at "14:00" "Meeting reminder"
    python -m tau.tools.schedule_message --in "2h" "Check on training job"
    python -m tau.tools.schedule_message --cron "0 9 * * *" "Daily standup"
"""

import subprocess
import argparse
import os
import re
import threading

WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _parse_delay_to_seconds(delay_str: str) -> int:
    """Parse a delay string like '1m', '30m', '2h', '1d' into seconds."""
    delay_str = delay_str.strip().lower()
    match = re.match(r'^(\d+)\s*([a-z]*)$', delay_str)
    if not match:
        try:
            return int(delay_str) * 60  # bare number = minutes
        except ValueError:
            return 60  # default 1 minute
    number = int(match.group(1))
    unit = match.group(2)
    if unit in ('s', 'sec', 'second', 'seconds'):
        return number
    elif unit in ('m', 'min', 'minute', 'minutes'):
        return number * 60
    elif unit in ('h', 'hr', 'hour', 'hours'):
        return number * 3600
    elif unit in ('d', 'day', 'days'):
        return number * 86400
    else:
        return number * 60  # default to minutes


def schedule_at(time_spec: str, message: str):
    """Schedule a one-time message using 'at'.

    Raises an exception if 'at' is not available or fails.
    """
    cmd = f'cd {WORKSPACE} && source .venv/bin/activate && python -m tau.tools.send_message "{message}"'
    proc = subprocess.Popen(
        ["at", time_spec],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    _, stderr = proc.communicate(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"at command failed: {stderr.strip()}")
    print(f"Scheduled for {time_spec}: {stderr.strip()}")


def schedule_delay_thread(delay_str: str, message: str):
    """Schedule a one-time message using a Python background thread.

    This is a fallback for systems where the 'at' command doesn't work
    (e.g. macOS where atrun is disabled by default).
    """
    seconds = _parse_delay_to_seconds(delay_str)

    def _send():
        from tau.telegram import notify
        notify(message)
        print(f"Timer fired after {seconds}s: {message[:50]}")

    timer = threading.Timer(seconds, _send)
    timer.daemon = True  # Don't prevent process exit
    timer.start()
    print(f"Timer scheduled: {seconds}s from now")


def schedule_cron(cron_spec: str, message: str):
    """Add a recurring cron job."""
    cmd = f'cd {WORKSPACE} && source .venv/bin/activate && python -m tau.tools.send_message "{message}"'
    current = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    existing = current.stdout if current.returncode == 0 else ""
    new_cron = f"{cron_spec} {cmd}\n"
    subprocess.run(["crontab", "-"], input=existing + new_cron, text=True)
    print(f"Added cron job: {cron_spec}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("message", help="Message to send")
    parser.add_argument("--at", dest="at_time", help="Time for 'at' (e.g., '14:00', 'now + 2 hours')")
    parser.add_argument("--in", dest="in_time", help="Relative time (e.g., '2h', '30m')")
    parser.add_argument("--cron", help="Cron schedule (e.g., '0 9 * * *')")
    args = parser.parse_args()
    
    if args.at_time:
        schedule_at(args.at_time, args.message)
    elif args.in_time:
        try:
            time_spec = f"now + {args.in_time.replace('h', ' hours').replace('m', ' minutes')}"
            schedule_at(time_spec, args.message)
        except Exception:
            print("'at' command failed, using Python timer fallback")
            schedule_delay_thread(args.in_time, args.message)
            # Keep process alive so the timer fires
            import time
            seconds = _parse_delay_to_seconds(args.in_time)
            time.sleep(seconds + 5)
    elif args.cron:
        schedule_cron(args.cron, args.message)


if __name__ == "__main__":
    main()
