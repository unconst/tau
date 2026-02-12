"""
Default configuration for the autonomous agent.

All settings are hardcoded - no CLI arguments needed.
"""

from __future__ import annotations

import os
from typing import Any, Dict

# Main configuration
CONFIG: Dict[str, Any] = {
    # ==========================================================================
    # Model Settings (simulates --model gpt-5.2 -c model_reasoning_effort=xhigh)
    # ==========================================================================
    # Model to use via Chutes API (OpenAI-compatible)
    "model": os.environ.get("LLM_MODEL", "deepseek-ai/DeepSeek-V3.2-TEE"),
    # Provider
    "provider": "chutes",
    # Reasoning effort: none, minimal, low, medium, high, xhigh (not used for Claude)
    "reasoning_effort": "none",
    # Token limits
    "max_tokens": 16384,
    # Temperature (0 = deterministic)
    "temperature": 0.0,
    # ==========================================================================
    # Agent Execution Settings
    # ==========================================================================
    # Maximum iterations before stopping
    "max_iterations": 200,
    # Maximum tokens for tool output truncation (middle-out strategy)
    "max_output_tokens": 2500,  # ~10KB
    # Timeout for shell commands (seconds)
    "shell_timeout": 60,
    # ==========================================================================
    # Context Management
    # ==========================================================================
    # NOTE: Context window, output reserve, compaction threshold, and pruning
    # constants are now defined per-model in ModelTier (see llm/router.py).
    # Do NOT add model_context_limit, output_token_max, auto_compact_threshold,
    # prune_protect, or prune_minimum here — they must flow from ModelTier.
    # ==========================================================================
    # Prompt Caching
    # ==========================================================================
    # Enable prompt caching
    "cache_enabled": True,
    # Note: Caching behavior depends on the model/provider
    # System prompt should be large enough to meet provider thresholds
    # ==========================================================================
    # Execution Flags
    # ==========================================================================
    # --dangerously-bypass-approvals-and-sandbox
    "bypass_approvals": True,
    "bypass_sandbox": True,
    # --skip-git-repo-check
    "skip_git_check": True,
    # --enable unified_exec
    "unified_exec": True,
    # --json (always JSONL output)
    "json_output": True,
    # ==========================================================================
    # Double Confirmation for Task Completion
    # ==========================================================================
    # Require double confirmation before marking task complete
    # Disabled for fully autonomous operation in evaluation mode
    "require_completion_confirmation": False,
}


def get_config() -> Dict[str, Any]:
    """Get the configuration dictionary."""
    return CONFIG.copy()


def get(key: str, default: Any = None) -> Any:
    """Get a configuration value."""
    return CONFIG.get(key, default)
