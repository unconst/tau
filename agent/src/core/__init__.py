"""Core module - agent loop, session management, and context compaction.

Heavy submodules (``executor``, ``loop``) are NOT re-exported here to
avoid circular imports.  Import them directly::

    from src.core.loop import run_agent_loop
    from src.core.executor import AgentExecutor
"""

# Compaction module (context management) — no cross-package deps, safe to
# import eagerly.
from src.core.compaction import (
    AUTO_COMPACT_THRESHOLD,
    MODEL_CONTEXT_LIMIT,
    OUTPUT_TOKEN_MAX,
    PRUNE_MARKER,
    PRUNE_MINIMUM,
    PRUNE_PROTECT,
    WorkingSet,
    estimate_message_tokens,
    estimate_tokens,
    estimate_total_tokens,
    get_working_set,
    is_overflow,
    manage_context,
    needs_compaction,
    prune_old_tool_outputs,
    run_compaction,
)

# Session management — only depends on src.config.models, safe to import.
from src.core.session import AgentContext, Session, ShellResult, SimpleAgentContext, TokenUsage

__all__ = [
    # Compaction
    "manage_context",
    "WorkingSet",
    "get_working_set",
    "estimate_tokens",
    "estimate_message_tokens",
    "estimate_total_tokens",
    "is_overflow",
    "needs_compaction",
    "prune_old_tool_outputs",
    "run_compaction",
    "MODEL_CONTEXT_LIMIT",
    "OUTPUT_TOKEN_MAX",
    "AUTO_COMPACT_THRESHOLD",
    "PRUNE_PROTECT",
    "PRUNE_MINIMUM",
    "PRUNE_MARKER",
    # Session
    "AgentContext",
    "SimpleAgentContext",
    "ShellResult",
    "Session",
    "TokenUsage",
]
