"""
Context compaction system for SuperAgent.

Implements intelligent context management:
1. Token-based overflow detection (tiktoken when available)
2. Relevance-aware tool output pruning
3. Working-set protection (files the agent is actively editing)
4. Smart output compression (keep errors, trim listings)
5. AI-powered conversation compaction (summarization)

This replaces naive sliding window truncation which breaks cache.
"""

from __future__ import annotations

import re
import sys
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from src.core.history_manager import HistoryManager
from src.utils.tokens import estimate_tokens

if TYPE_CHECKING:
    from src.llm.client import LLMClient

# =============================================================================
# Constants (matching OpenCode)
# =============================================================================

# Fallback context limits — used ONLY when no per-model metadata is available.
# In normal operation, callers pass ``context_window`` / ``output_reserve`` /
# ``auto_compact_threshold`` from ``ModelTier`` (see llm/router.py).
# These fallbacks match the default ModelTier values.
MODEL_CONTEXT_LIMIT = 131_072  # Match default ModelTier
OUTPUT_TOKEN_MAX = 16_384      # Match default ModelTier
AUTO_COMPACT_THRESHOLD = 0.75  # Match default ModelTier

# Pruning constants — tuned for lean context
PRUNE_PROTECT = 20_000  # Protect this many tokens of recent tool output
PRUNE_MINIMUM = 8_000   # Only prune if we can recover at least this many tokens
PRUNE_MARKER = "[Old tool result content cleared]"

# Relevance scores for tool output types (higher = more important to keep)
RELEVANCE_SCORES: Dict[str, float] = {
    "error": 1.0,           # Always keep errors
    "write_file": 0.9,      # File modification results
    "apply_patch": 0.9,     # Patch results
    "str_replace": 0.9,     # Edit results
    "shell_command": 0.7,   # Command output (depends on content)
    "grep_files": 0.5,      # Search results
    "read_file": 0.4,       # File contents (can re-read)
    "list_dir": 0.3,        # Directory listings (least important)
    "glob_files": 0.3,      # File listings
    "web_search": 0.6,      # Web results
    "spawn_subagent": 0.8,  # Subagent results
}

# Compaction prompts
COMPACTION_PROMPT = """You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.

Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data, examples, or references needed to continue
- Which files were modified and how
- Any errors encountered and how they were resolved

Be concise, structured, and focused on helping the next LLM seamlessly continue the work. Use bullet points and clear sections."""

SUMMARY_PREFIX = """Another language model started to solve this problem and produced a summary of its thinking process. You also have access to the state of the tools that were used. Use this to build on the work that has already been done and avoid duplicating work.

Here is the summary from the previous context:

"""


# =============================================================================
# Token Estimation
# =============================================================================


def estimate_message_tokens(msg: Dict[str, Any]) -> int:
    """Estimate tokens for a single message."""
    tokens = 0

    content = msg.get("content")
    if isinstance(content, str):
        tokens += estimate_tokens(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                tokens += estimate_tokens(part.get("text", ""))
                if part.get("type") == "image_url":
                    tokens += 1000

    tool_calls = msg.get("tool_calls", [])
    for tc in tool_calls:
        func = tc.get("function", {})
        tokens += estimate_tokens(func.get("name", ""))
        tokens += estimate_tokens(func.get("arguments", ""))

    # Role overhead (~4 tokens)
    tokens += 4

    return tokens


def estimate_total_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estimate total tokens for all messages."""
    return sum(estimate_message_tokens(m) for m in messages)


# =============================================================================
# Working Set Tracking
# =============================================================================


class WorkingSet:
    """Tracks files the agent is actively working on.

    Files in the working set get their tool outputs protected from pruning
    for longer, since the agent is likely to reference them again.
    """

    def __init__(self):
        self._files: Dict[str, float] = {}  # path -> last_access_time
        self._max_age = 600.0  # 10 minutes

    def touch(self, path: str) -> None:
        """Mark a file as recently accessed."""
        self._files[path] = time.time()

    def is_active(self, path: str) -> bool:
        """Check if a file is in the active working set."""
        if path not in self._files:
            return False
        age = time.time() - self._files[path]
        if age > self._max_age:
            del self._files[path]
            return False
        return True

    def extract_paths_from_message(self, msg: Dict[str, Any]) -> Set[str]:
        """Extract file paths mentioned in a message."""
        paths = set()
        content = msg.get("content", "")
        if isinstance(content, str):
            # Match common file path patterns
            for match in re.finditer(r'[\w./\-]+\.\w{1,10}', content):
                paths.add(match.group())
        return paths

    def update_from_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Scan recent messages to update the working set."""
        # Look at last 10 messages for file paths
        for msg in messages[-10:]:
            role = msg.get("role", "")
            if role in ("assistant", "tool"):
                for path in self.extract_paths_from_message(msg):
                    self.touch(path)


# Global working set instance
_working_set = WorkingSet()


def get_working_set() -> WorkingSet:
    """Get the global working set tracker."""
    return _working_set


# =============================================================================
# Overflow Detection
# =============================================================================


def get_usable_context(
    context_window: int = MODEL_CONTEXT_LIMIT,
    output_reserve: int = OUTPUT_TOKEN_MAX,
) -> int:
    """Get usable context window (total - reserved for output)."""
    return context_window - output_reserve


def is_overflow(
    total_tokens: int,
    threshold: float = AUTO_COMPACT_THRESHOLD,
    context_window: int = MODEL_CONTEXT_LIMIT,
    output_reserve: int = OUTPUT_TOKEN_MAX,
) -> bool:
    """Check if context is overflowing based on token count."""
    usable = get_usable_context(context_window, output_reserve)
    return total_tokens > usable * threshold


def needs_compaction(
    messages: List[Dict[str, Any]],
    context_window: int = MODEL_CONTEXT_LIMIT,
    output_reserve: int = OUTPUT_TOKEN_MAX,
) -> bool:
    """Check if messages need compaction."""
    total_tokens = estimate_total_tokens(messages)
    return is_overflow(total_tokens, context_window=context_window, output_reserve=output_reserve)


# =============================================================================
# Smart Output Compression
# =============================================================================


def _get_tool_name_for_message(messages: List[Dict[str, Any]], msg_index: int) -> str:
    """Try to determine which tool produced a tool result message.

    Looks at the preceding assistant message's tool_calls to match by tool_call_id.
    """
    msg = messages[msg_index]
    tool_call_id = msg.get("tool_call_id", "")

    # Search backward for the assistant message with matching tool_calls
    for i in range(msg_index - 1, -1, -1):
        prev = messages[i]
        if prev.get("role") == "assistant":
            for tc in prev.get("tool_calls", []):
                if tc.get("id") == tool_call_id:
                    return tc.get("function", {}).get("name", "unknown")
            break

    return "unknown"


def compress_tool_output(content: str, tool_name: str) -> str:
    """Compress a tool output based on its type.

    Aggressive compression for context efficiency:
    - Errors: keep fully
    - Shell commands: keep first 5 + last 20 lines
    - File reads: keep first 10 + last 10 lines
    - Directory listings: keep first 15 entries
    - Search results: keep first 15 matches
    """
    if not content or len(content) < 300:
        return content

    lines = content.split("\n")

    # Always keep error content fully
    if any(kw in content[:200].lower() for kw in ["error", "failed", "exception", "traceback"]):
        return content

    if tool_name in ("shell_command",):
        if len(lines) > 30:
            exit_lines = [l for l in lines[-5:] if "exit code" in l.lower()]
            kept = lines[:5] + [f"\n[... {len(lines) - 25} lines trimmed ...]"] + lines[-20:]
            if exit_lines:
                kept.extend(exit_lines)
            return "\n".join(kept)

    elif tool_name in ("read_file",):
        if len(lines) > 25:
            return "\n".join(
                lines[:10]
                + [f"\n[... {len(lines) - 20} lines trimmed ...]"]
                + lines[-10:]
            )

    elif tool_name in ("list_dir", "glob_files"):
        if len(lines) > 20:
            return "\n".join(lines[:15] + [f"\n[... {len(lines) - 15} more entries ...]"])

    elif tool_name in ("grep_files",):
        if len(lines) > 20:
            return "\n".join(lines[:15] + [f"\n[... {len(lines) - 15} more matches ...]"])

    elif tool_name in ("spawn_subagent",):
        if len(lines) > 40:
            return "\n".join(
                lines[:15]
                + [f"\n[... {len(lines) - 30} lines trimmed ...]"]
                + lines[-15:]
            )

    return content


# =============================================================================
# Tool Output Pruning
# =============================================================================


def _log(msg: str) -> None:
    """Log to stderr."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [compaction] {msg}", file=sys.stderr, flush=True)


def prune_old_tool_outputs(
    messages: List[Dict[str, Any]],
    protect_last_turns: int = 2,
) -> List[Dict[str, Any]]:
    """
    Prune old tool outputs with relevance awareness.

    Strategy:
    1. Go backwards through messages
    2. Skip first 2 user turns (most recent)
    3. Accumulate tool output tokens weighted by relevance
    4. Protect outputs for files in the working set
    5. Compress outputs before pruning entirely
    6. Only fully prune if we can recover > PRUNE_MINIMUM tokens

    Args:
        messages: List of messages
        protect_last_turns: Number of recent user turns to skip

    Returns:
        Messages with old tool outputs pruned/compressed
    """
    if not messages:
        return messages

    # Update working set from recent context
    _working_set.update_from_messages(messages)

    total = 0
    pruned = 0
    to_prune: List[int] = []
    to_compress: List[int] = []
    turns = 0

    for msg_index in range(len(messages) - 1, -1, -1):
        msg = messages[msg_index]

        if msg.get("role") == "user":
            turns += 1

        if turns < protect_last_turns:
            continue

        if msg.get("role") == "tool":
            content = msg.get("content", "")

            if content == PRUNE_MARKER:
                break

            estimate = estimate_tokens(content)
            total += estimate

            # Get tool name for relevance scoring
            tool_name = _get_tool_name_for_message(messages, msg_index)
            relevance = RELEVANCE_SCORES.get(tool_name, 0.5)

            # Check if any file in working set is mentioned
            paths_in_content = _working_set.extract_paths_from_message(msg)
            is_working_set = any(_working_set.is_active(p) for p in paths_in_content)

            # Protect working set outputs and high-relevance outputs longer
            effective_protect = PRUNE_PROTECT
            if is_working_set:
                effective_protect = int(PRUNE_PROTECT * 1.3)  # Modest protection boost
            elif relevance >= 0.8:
                effective_protect = int(PRUNE_PROTECT * 1.2)

            if total > effective_protect:
                # Low relevance: prune entirely
                if relevance < 0.5:
                    pruned += estimate
                    to_prune.append(msg_index)
                # Medium relevance: compress first
                elif relevance < 0.8:
                    to_compress.append(msg_index)
                    # Estimate compression savings (~50%)
                    pruned += estimate // 2
                else:
                    # High relevance: only prune if very old
                    if total > effective_protect * 2:
                        to_compress.append(msg_index)
                        pruned += estimate // 3

    _log(f"Prune scan: {total} total tokens, {pruned} recoverable, "
         f"{len(to_prune)} prune + {len(to_compress)} compress")

    if pruned <= PRUNE_MINIMUM:
        _log(f"Prune skipped: only {pruned} tokens recoverable (min: {PRUNE_MINIMUM})")
        return messages

    _log(f"Pruning {len(to_prune)} + compressing {len(to_compress)} tool outputs")

    indices_to_prune = set(to_prune)
    indices_to_compress = set(to_compress)
    result = []

    for i, msg in enumerate(messages):
        if i in indices_to_prune:
            result.append({**msg, "content": PRUNE_MARKER})
        elif i in indices_to_compress:
            tool_name = _get_tool_name_for_message(messages, i)
            compressed = compress_tool_output(msg.get("content", ""), tool_name)
            result.append({**msg, "content": compressed})
        else:
            result.append(msg)

    return result


# =============================================================================
# AI Compaction
# =============================================================================


def run_compaction(
    llm: "LLMClient",
    messages: List[Dict[str, Any]],
    system_prompt: str,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Compact conversation history using AI summarization.

    Process:
    1. Send all messages + compaction prompt to LLM
    2. Get summary response
    3. Create new message list:
       - Original system prompt
       - Summary as user message (with prefix)
       - Ready for continuation

    Args:
        llm: LLM client for summarization
        messages: Current message history
        system_prompt: Original system prompt to preserve
        model: Model to use (defaults to current)

    Returns:
        Compacted message list
    """
    _log("Starting AI compaction...")

    compaction_messages = messages.copy()
    compaction_messages.append({"role": "user", "content": COMPACTION_PROMPT})

    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        try:
            response = llm.chat(
                compaction_messages,
                model=model,
                max_tokens=4096,
            )

            summary = response.text or ""

            if not summary:
                _log("Compaction failed: empty response")
                continue

            summary_tokens = estimate_tokens(summary)
            _log(f"Compaction complete: {summary_tokens} token summary")

            compacted = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": SUMMARY_PREFIX + summary},
            ]
            return compacted

        except Exception as e:
            _log(f"Compaction failed (attempt {attempt}/{max_attempts}): {e}")
            time.sleep(min(2.0, attempt * 0.5))

    _log("Compaction fallback: keeping deterministically trimmed history")
    return messages


# =============================================================================
# Main Context Management
# =============================================================================


def manage_context(
    messages: List[Dict[str, Any]],
    system_prompt: str,
    llm: "LLMClient",
    force_compaction: bool = False,
    _token_budget: Optional["_TokenBudget"] = None,
    context_window: int = MODEL_CONTEXT_LIMIT,
    output_reserve: int = OUTPUT_TOKEN_MAX,
    auto_compact_threshold: float = AUTO_COMPACT_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Main context management function.

    Called before each LLM request to ensure context fits.

    Strategy:
    1. Estimate current token usage (incrementally if budget provided)
    2. If under threshold, return as-is
    3. Try smart pruning + compression first
    4. If still over threshold, run AI compaction

    Args:
        messages: Current message history
        system_prompt: Original system prompt (preserved through compaction)
        llm: LLM client (for compaction)
        force_compaction: Force compaction even if under threshold
        _token_budget: Optional incremental token tracker (avoids re-counting)
        context_window: Model's total context window (from ModelTier).
        output_reserve: Tokens to reserve for output (from ModelTier).
        auto_compact_threshold: Fraction threshold for compaction trigger.

    Returns:
        Managed message list (possibly compacted)
    """
    _is_over = lambda t: is_overflow(
        t, threshold=auto_compact_threshold,
        context_window=context_window, output_reserve=output_reserve,
    )

    if _token_budget is not None:
        total_tokens = _token_budget.update(messages)
    else:
        total_tokens = estimate_total_tokens(messages)

    usable = get_usable_context(context_window, output_reserve)
    usage_pct = (total_tokens / usable) * 100 if usable else 100.0

    _log(f"Context: {total_tokens} tokens ({usage_pct:.1f}% of {usable})")

    if not force_compaction and not _is_over(total_tokens):
        # Even under token threshold, prune if message count is excessive
        if len(messages) > 60:
            _log(f"Message count high ({len(messages)}), pruning old tool outputs...")
            pruned = prune_old_tool_outputs(messages)
            if len(pruned) < len(messages):
                if _token_budget is not None:
                    _token_budget.reset(pruned)
                return pruned
        return messages

    _log("Context overflow detected, managing...")

    # Step 1: Try smart pruning + compression
    pruned = prune_old_tool_outputs(messages)
    pruned_tokens = estimate_total_tokens(pruned)

    if not _is_over(pruned_tokens) and not force_compaction:
        _log(f"Pruning sufficient: {total_tokens} -> {pruned_tokens} tokens")
        if _token_budget is not None:
            _token_budget.reset(pruned)
        return pruned

    # Step 2: Pair-aware deterministic trimming before AI compaction
    preserve_system = bool(pruned and pruned[0].get("role") == "system")
    system_message = pruned[0] if preserve_system else None
    trimmed = pruned
    trim_attempts = 0
    while _is_over(estimate_total_tokens(trimmed)) and len(trimmed) > 2 and trim_attempts < 20:
        trimmed = HistoryManager.remove_first_item(
            trimmed,
            preserve_system_prompt=preserve_system,
        )
        trim_attempts += 1
    if preserve_system and system_message and (
        not trimmed or trimmed[0].get("role") != "system"
    ):
        trimmed = [system_message] + trimmed
    if trim_attempts:
        _log(f"Pair-aware trimming removed {trim_attempts} oldest items")

    # Step 3: Run AI compaction if needed
    if not _is_over(estimate_total_tokens(trimmed)) and not force_compaction:
        _log("Pair-aware trimming was sufficient")
        if _token_budget is not None:
            _token_budget.reset(trimmed)
        return trimmed

    _log("Pruning + trimming insufficient, running AI compaction...")
    compacted = run_compaction(llm, trimmed, system_prompt)
    compacted_tokens = estimate_total_tokens(compacted)

    _log(f"Compaction result: {total_tokens} -> {compacted_tokens} tokens")

    if _token_budget is not None:
        _token_budget.reset(compacted)

    return compacted


class _TokenBudget:
    """Incremental token counter — avoids re-estimating the entire history.

    Tracks how many messages have already been counted and only estimates
    tokens for newly appended messages.
    """

    def __init__(self) -> None:
        self._counted: int = 0  # number of messages already counted
        self._total: int = 0    # running token total

    def update(self, messages: List[Dict[str, Any]]) -> int:
        """Return total tokens, only counting new messages since last call."""
        n = len(messages)
        if n < self._counted:
            # Messages were removed (compaction) — full recount
            self.reset(messages)
            return self._total
        # Estimate only the new messages
        for msg in messages[self._counted:]:
            self._total += estimate_message_tokens(msg)
        self._counted = n
        return self._total

    def reset(self, messages: List[Dict[str, Any]]) -> None:
        """Force a full recount (after compaction / pruning)."""
        self._total = estimate_total_tokens(messages)
        self._counted = len(messages)
