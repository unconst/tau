"""Token estimation with optional tiktoken support.

Uses tiktoken for accurate counting when available,
falling back to a character-based heuristic.
"""

from __future__ import annotations

from typing import Optional

# Try to import tiktoken for accurate counting
_tiktoken_enc = None
_tiktoken_available = False

try:
    import tiktoken
    # Use cl100k_base (GPT-4/Claude-compatible) as a good universal tokenizer
    _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
    _tiktoken_available = True
except ImportError:
    pass


def estimate_tokens(text: str, model: Optional[str] = None) -> int:
    """Estimate token count for text.

    Uses tiktoken when available for accurate counting.
    Falls back to the ~4 chars/token heuristic.

    Args:
        text: Text to estimate tokens for.
        model: Optional model name (unused currently, for future per-model tokenizers).

    Returns:
        Estimated token count.
    """
    if not text:
        return 0

    if _tiktoken_available and _tiktoken_enc is not None:
        return len(_tiktoken_enc.encode(text))

    # Fallback: ~4 characters per token for English text
    return (len(text) // 4) + 1


def is_tiktoken_available() -> bool:
    """Check if tiktoken is available for accurate token counting."""
    return _tiktoken_available
