"""Tau tools module.

This module contains tools that the agent can use to interact with the user and system.
Each tool is a standalone script that can be run from the command line.

Available tools:
- send_message: Send a text message to the user via Telegram
- send_voice: Send a voice message (TTS) to the user via Telegram
- search_skills: Search for creative AI skills/tools (Eve/Eden.art ecosystem)
- lium: Manage GPU pods on the Lium network (create, delete, SSH, exec)
- commands: Execute tau bot commands (task, plan, status, adapt, cron, etc.)
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
    "search_skills": {
        "command": "python -m tau.tools.search_skills",
        "description": "Search for creative AI skills/tools (image, video, audio generation, social media, etc.)",
        "usage": 'python -m tau.tools.search_skills "query" or python -m tau.tools.search_skills --category image',
        "examples": [
            'python -m tau.tools.search_skills                    # List all skills',
            'python -m tau.tools.search_skills "image"            # Search for image skills',
            'python -m tau.tools.search_skills --category video   # Filter by category',
            'python -m tau.tools.search_skills --details flux     # Get skill details',
        ],
    },
    "lium": {
        "command": "python -m tau.tools.lium",
        "description": "Manage GPU pods on the Lium network (Bittensor Subnet 51)",
        "usage": "python -m tau.tools.lium COMMAND [OPTIONS]",
        "env_required": ["LIUM_API_KEY"],
        "examples": [
            "python -m tau.tools.lium ls                          # List available GPU nodes",
            "python -m tau.tools.lium ls H100                     # List only H100 GPUs",
            "python -m tau.tools.lium ls --max-price 2.5          # Filter by max price",
            "python -m tau.tools.lium ps                          # List your active pods",
            "python -m tau.tools.lium up 1                        # Create pod on executor #1",
            "python -m tau.tools.lium up --gpu H100 --name my-pod # Create named pod with H100",
            "python -m tau.tools.lium rm my-pod                   # Remove a pod",
            "python -m tau.tools.lium rm all                      # Remove all pods",
            "python -m tau.tools.lium exec my-pod 'nvidia-smi'    # Execute command on pod",
            "python -m tau.tools.lium ssh my-pod                  # SSH into pod",
            "python -m tau.tools.lium scp my-pod ./file.py        # Copy file to pod",
            "python -m tau.tools.lium templates                   # List available templates",
        ],
    },
    "commands": {
        "command": "python -m tau.tools.commands",
        "description": "Execute tau bot commands programmatically (self-directed behavior)",
        "usage": "python -m tau.tools.commands COMMAND [ARGS]",
        "examples": [
            'python -m tau.tools.commands task "Research X"       # Create a task for yourself',
            'python -m tau.tools.commands plan "Build feature Y"  # Create an execution plan',
            'python -m tau.tools.commands status                  # Check task/memory status',
            'python -m tau.tools.commands adapt "Add feature Z"   # Self-modify code (triggers restart)',
            'python -m tau.tools.commands cron 1h "Check status"  # Schedule recurring prompt',
            'python -m tau.tools.commands crons                   # List active cron jobs',
            'python -m tau.tools.commands uncron 1                # Remove cron job #1',
            'python -m tau.tools.commands clear                   # Stop active agent processes',
        ],
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
