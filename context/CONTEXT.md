# Context Loader

Read this file first to determine which context to load.

## Core Identity
- `context/IDENTITY.md`
- `context/MEMORY-SYSTEM.md`

## Conversation Context (Auto-loaded)
- `context/CHAT_SUMMARY.md` - Hourly summary of full conversation (always included in agent calls)
- `context/CHAT.md` (full conversation history - only load if deep history needed)

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
