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

WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def schedule_at(time_spec: str, message: str):
    """Schedule a one-time message using 'at'."""
    cmd = f'cd {WORKSPACE} && source .venv/bin/activate && python -m tau.tools.send_message "{message}"'
    proc = subprocess.Popen(
        ["at", time_spec],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    _, stderr = proc.communicate(cmd)
    print(f"Scheduled for {time_spec}: {stderr.strip()}")


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
        schedule_at(f"now + {args.in_time.replace('h', ' hours').replace('m', ' minutes')}", args.message)
    elif args.cron:
        schedule_cron(args.cron, args.message)


if __name__ == "__main__":
    main()
