# Context Loader

Read this file first to determine which context to load.

## Core Identity
- `context/IDENTITY.md`
- `context/MEMORY-SYSTEM.md`

## Conversation Context (Auto-loaded)
- `context/CHAT_SUMMARY.md` - Hourly summary of full conversation (always included in agent calls)

## Multi-Chat Storage
Chat history is stored per-chat under `context/chats/<chat_id>/`:
- `context/chats/<chat_id>/CHAT.md` - Conversation history for that chat
- `context/chats/<chat_id>/meta.json` - Chat metadata (title, type, username, updated_at)

The owner's 1:1 private chat is the primary conversation.
Group chats the bot is added to are observed and logged (all messages from all users).
Only the owner can issue commands or get responses — other users' messages are logged silently.

**Programmatic access (from Python):**
- `list_chats()` — returns a list of dicts with metadata for every known chat
- `get_chat_history_for(chat_id, max_lines=100)` — returns recent history for a specific chat
- `send_to_chat(chat_id, text)` — sends a message to any chat the bot is a member of
These are importable from `tau.telegram`.

## Task-Specific Context
Load only for the active task (e.g. if the user asks about a specific task ID):
- `context/tasks/task-{id}/task.md`
- `context/tasks/task-{id}/memory.md`

## History (Load ONLY if continuity is needed)
- `context/tasks/memory.md` (high-level milestones)

## Skills Reference
Load only if relevant to the request:
- `context/skills/agent.md` - Cursor CLI agent documentation
- `context/skills/eve-skills.md` - Eve creative skills
- `context/skills/lium-skills.md` - Lium GPU management
- `context/skills/self-scheduling.md` - Creating tasks, reminders, and cron jobs for yourself
