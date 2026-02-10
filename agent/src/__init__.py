"""
Tau Agent - An autonomous coding agent.

Uses Chutes API via httpx for LLM interaction.

Usage:
    python agent.py --instruction "Your task here..."

Heavy modules (tools, core.loop, core.executor) are NOT re-exported here
to avoid circular imports.  Import them directly::

    from src.tools.registry import ToolRegistry
    from src.core.loop import run_agent_loop
"""

__version__ = "1.0.0"
__author__ = "Platform Network"

# Only re-export lightweight, leaf-node modules that don't trigger cycles.
from src.config.defaults import CONFIG
from src.output.jsonl import emit

__all__ = [
    "CONFIG",
    "emit",
    "__version__",
]
