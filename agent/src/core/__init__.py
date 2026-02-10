"""Core module - agent loop, session management, and context compaction."""

# Compaction module (context management)
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
from src.core.executor import (
    AgentExecutor,
    ExecutionResult,
    RiskLevel,
    SandboxPolicy,
)

# Import run_agent_loop
from src.core.loop import run_agent_loop

# Session management
from src.core.session import AgentContext, Session, ShellResult, SimpleAgentContext, TokenUsage

__all__ = [
    # Executor
    "AgentExecutor",
    "ExecutionResult",
    "RiskLevel",
    "SandboxPolicy",
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
    # Loop
    "run_agent_loop",
    # Session
    "AgentContext",
    "SimpleAgentContext",
    "ShellResult",
    "Session",
    "TokenUsage",
]
