#!/usr/bin/env python3
"""Send a message to the Telegram chat.

Agents can call this script to send messages back to the user on Telegram.

Usage:
    python -m tau.tools.send_message "Your message here"
    python -m tau.tools.send_message "Task completed successfully"
    
The script uses the existing Telegram integration and sends messages to the same
chat where the user receives agent updates (chat ID stored in chat_id.txt).
"""

import sys
from tau.telegram import notify


def main():
    """Send message from command line arguments."""
    if len(sys.argv) < 2:
        print('Usage: python -m tau.tools.send_message "your message"', file=sys.stderr)
        sys.exit(1)
    
    message = " ".join(sys.argv[1:])
    notify(message)
    print(f"Message sent: {message[:50]}..." if len(message) > 50 else f"Message sent: {message}")


if __name__ == "__main__":
    main()
