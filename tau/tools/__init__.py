"""Tau tools module.

This module contains tools that the agent can use to interact with the user and system.
Each tool is a standalone script that can be run from the command line.

Available tools:
- send_message: Send a text message to the user via Telegram
- send_voice: Send a voice message (TTS) to the user via Telegram
"""

# Tool registry for programmatic access
TOOLS = {
    "send_message": {
        "command": "python -m tau.tools.send_message",
        "description": "Send a text message to the user via Telegram",
        "usage": 'python -m tau.tools.send_message "Your message here"',
    },
    "send_voice": {
        "command": "python -m tau.tools.send_voice",
        "description": "Send a voice message (TTS) to the user via Telegram",
        "usage": 'python -m tau.tools.send_voice "Your message here"',
    },
}


def get_tools_documentation() -> str:
    """Generate documentation string for all available tools."""
    lines = ["AVAILABLE TOOLS:", ""]
    for name, info in TOOLS.items():
        lines.append(f"- {name}: {info['description']}")
        lines.append(f"  Usage: {info['usage']}")
        lines.append("")
    return "\n".join(lines)
