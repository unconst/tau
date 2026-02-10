"""Dynamic model routing based on task complexity.

The ``ModelRouter`` selects which model to use for each LLM call based on
heuristics about the current task's complexity:

- **simple** tasks (single tool call, short context) use a fast/cheap model
- **medium** tasks (multi-step but scoped) use the default model
- **complex** tasks (multi-file changes, long context) use the strongest model

Callers can also override per-call (e.g. cheap model for verification steps).

Each ``ModelTier`` now carries per-model metadata (context window, capabilities,
truncation policy) so the rest of the system can adapt behavior per model
instead of using hardcoded global constants.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class TaskComplexity(Enum):
    """Classification of task complexity."""

    SIMPLE = auto()    # Quick lookups, single-file reads, simple questions
    MEDIUM = auto()    # Multi-step tasks within a scoped area
    COMPLEX = auto()   # Multi-file changes, architecture decisions, long reasoning


@dataclass
class ModelTier:
    """A model configuration for a specific tier.

    Attributes:
        model: Model identifier string.
        max_tokens: Maximum output tokens to request.
        temperature: Sampling temperature (0.0 = deterministic).
        reasoning_effort: Reasoning effort hint ("none", "low", "medium", "high").
        context_window: Total context window size in tokens.
        output_reserve: Tokens to reserve for the model's output.
        auto_compact_threshold: Fraction of usable context that triggers compaction.
        supports_reasoning: Whether the model supports extended reasoning.
        supports_parallel_tools: Whether the model can emit parallel tool calls.
        supports_temperature: Whether the model accepts a temperature parameter.
        tool_output_max_tokens: Max tokens per tool output before truncation at
            execution time (0 = no execution-time truncation).
    """

    model: str
    max_tokens: int = 16384
    temperature: float = 0.0
    reasoning_effort: str = "none"

    # Per-model context / capability metadata
    context_window: int = 200_000
    output_reserve: int = 32_000
    auto_compact_threshold: float = 0.85
    supports_reasoning: bool = False
    supports_parallel_tools: bool = True
    supports_temperature: bool = True
    tool_output_max_tokens: int = 0  # 0 = defer to compaction

    @property
    def usable_context(self) -> int:
        """Context window minus output reserve."""
        return self.context_window - self.output_reserve

    @property
    def compact_trigger(self) -> int:
        """Token count at which compaction should trigger."""
        return int(self.usable_context * self.auto_compact_threshold)


@dataclass
class RouterConfig:
    """Configuration for the model router."""

    # Model for each complexity tier
    fast: ModelTier = field(default_factory=lambda: ModelTier(
        model=os.environ.get("TAU_FAST_MODEL", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"),
        max_tokens=8192,
        context_window=32_768,
        output_reserve=4_096,
        supports_reasoning=False,
        supports_parallel_tools=True,
        supports_temperature=True,
        tool_output_max_tokens=4_000,
    ))
    default: ModelTier = field(default_factory=lambda: ModelTier(
        model=os.environ.get("TAU_AGENT_MODEL", os.environ.get("LLM_MODEL", "zai-org/GLM-4.7-TEE")),
        max_tokens=16384,
        context_window=131_072,
        output_reserve=16_384,
        supports_reasoning=False,
        supports_parallel_tools=True,
        supports_temperature=True,
        tool_output_max_tokens=8_000,
    ))
    strong: ModelTier = field(default_factory=lambda: ModelTier(
        model=os.environ.get("TAU_STRONG_MODEL", os.environ.get("LLM_MODEL", "deepseek-ai/DeepSeek-R1-0528-TEE")),
        max_tokens=16384,
        reasoning_effort="high",
        context_window=131_072,
        output_reserve=32_000,
        auto_compact_threshold=0.80,
        supports_reasoning=True,
        supports_parallel_tools=False,
        supports_temperature=False,
        tool_output_max_tokens=8_000,
    ))

    # Enable/disable routing (when disabled, always use default)
    enabled: bool = True


class ModelRouter:
    """Selects the appropriate model based on task complexity.

    Usage::

        router = ModelRouter()

        # Auto-select based on context
        tier = router.select(messages, iteration=3, tool_count=2)
        model = tier.model

        # Or use explicit overrides
        tier = router.for_subagent("explore")
        tier = router.for_verification()
    """

    def __init__(self, config: Optional[RouterConfig] = None):
        self.config = config or RouterConfig()

    def select(
        self,
        messages: List[Dict[str, Any]],
        iteration: int = 0,
        tool_count: int = 0,
        is_verification: bool = False,
    ) -> ModelTier:
        """Select a model tier based on the current state.

        Args:
            messages: Current conversation messages.
            iteration: Current iteration number.
            tool_count: Number of tool calls so far.
            is_verification: Whether this is a verification step.

        Returns:
            The selected ModelTier.
        """
        if not self.config.enabled:
            return self.config.default

        # Verification steps use fast model (just checking, not creating)
        if is_verification:
            return self.config.fast

        complexity = self.classify(messages, iteration, tool_count)

        if complexity == TaskComplexity.SIMPLE:
            return self.config.fast
        elif complexity == TaskComplexity.COMPLEX:
            return self.config.strong
        else:
            return self.config.default

    def classify(
        self,
        messages: List[Dict[str, Any]],
        iteration: int = 0,
        tool_count: int = 0,
    ) -> TaskComplexity:
        """Classify the current task's complexity.

        Heuristics:
        - Short instruction + few messages = SIMPLE
        - Many iterations or long context = COMPLEX
        - Everything else = MEDIUM
        """
        # Count total tokens roughly
        total_chars = sum(
            len(str(m.get("content", "")))
            for m in messages
        )

        # Get instruction length
        instruction = ""
        for m in messages:
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, str):
                    instruction = content
                    break

        instruction_len = len(instruction)

        # Simple: short instruction, early iteration, small context
        if instruction_len < 200 and iteration < 3 and total_chars < 10000:
            return TaskComplexity.SIMPLE

        # Complex: many iterations, large context, or complex instruction
        if iteration > 15 or total_chars > 100000 or instruction_len > 2000:
            return TaskComplexity.COMPLEX

        # Complex: instruction mentions multiple files or architectural changes
        complex_keywords = [
            "refactor", "redesign", "migrate", "architecture",
            "across all", "every file", "entire codebase",
            "multiple files", "comprehensive",
        ]
        if any(kw in instruction.lower() for kw in complex_keywords):
            return TaskComplexity.COMPLEX

        return TaskComplexity.MEDIUM

    def for_subagent(self, subagent_type: str) -> ModelTier:
        """Get the model tier for a subagent.

        Args:
            subagent_type: 'explore' or 'execute'.

        Returns:
            The appropriate ModelTier.
        """
        if subagent_type == "explore":
            return self.config.fast
        return self.config.default

    def for_verification(self) -> ModelTier:
        """Get the model tier for verification steps."""
        return self.config.fast

    def for_compaction(self) -> ModelTier:
        """Get the model tier for context compaction/summarization."""
        return self.config.fast
