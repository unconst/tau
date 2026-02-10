"""
Tau Agent - An autonomous coding agent.

Uses Chutes API via httpx for LLM interaction.

Usage:
    python agent.py --instruction "Your task here..."
"""

__version__ = "1.0.0"
__author__ = "Platform Network"

# Import main components for convenience
from src.config.defaults import CONFIG
from src.output.jsonl import emit
from src.tools.registry import ToolRegistry

__all__ = [
    "CONFIG",
    "ToolRegistry",
    "emit",
    "__version__",
]
