"""Conversation history utilities with pair-aware trimming."""

from __future__ import annotations

from typing import Any, Dict, List


class HistoryManager:
    """Maintains message history integrity for compaction/trimming."""

    @staticmethod
    def remove_first_item(
        messages: List[Dict[str, Any]],
        preserve_system_prompt: bool = False,
    ) -> List[Dict[str, Any]]:
        if not messages:
            return messages

        remove_index = 0
        if preserve_system_prompt and messages[0].get("role") == "system" and len(messages) > 1:
            remove_index = 1

        removed = messages[remove_index]
        remaining = messages[:remove_index] + messages[remove_index + 1 :]
        return HistoryManager._remove_corresponding_for(remaining, removed)

    @staticmethod
    def _remove_corresponding_for(
        messages: List[Dict[str, Any]],
        removed: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        role = removed.get("role")
        if role == "tool":
            tool_call_id = removed.get("tool_call_id")
            if not tool_call_id:
                return messages
            filtered: list[Dict[str, Any]] = []
            for msg in messages:
                if msg.get("role") != "assistant":
                    filtered.append(msg)
                    continue
                tool_calls = msg.get("tool_calls", []) or []
                kept_calls = [c for c in tool_calls if c.get("id") != tool_call_id]
                if kept_calls != tool_calls:
                    updated = dict(msg)
                    updated["tool_calls"] = kept_calls
                    filtered.append(updated)
                else:
                    filtered.append(msg)
            return filtered

        if role == "assistant":
            ids = {
                c.get("id")
                for c in (removed.get("tool_calls", []) or [])
                if isinstance(c, dict) and c.get("id")
            }
            if not ids:
                return messages
            return [
                m
                for m in messages
                if not (m.get("role") == "tool" and m.get("tool_call_id") in ids)
            ]

        if role == "user":
            # Maintain turn coherence: when dropping a user entry, drop the
            # immediately following assistant response if present.
            if messages and messages[0].get("role") == "assistant":
                return messages[1:]
        return messages

