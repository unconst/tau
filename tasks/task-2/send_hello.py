#!/usr/bin/env python3
"""Script to send a message "hello" via Telegram 3 minutes from now."""

import os
import sys
import time

# Add workspace root to path to import tau modules
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, workspace_root)

from tau.telegram import notify


def main():
    """Wait 3 minutes then send 'hello' via Telegram."""
    # Wait 3 minutes (180 seconds)
    print("Waiting 3 minutes before sending message...")
    time.sleep(180)
    
    # Send the message
    message = "hello"
    notify(message)
    print(f"Message sent: {message}")


if __name__ == "__main__":
    main()
