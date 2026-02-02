# Memory System

Tau uses a hierarchical memory system to organize task-specific information and high-level summaries.

## Structure

### Task Directories

Each task has its own directory under `tasks/`:
- `tasks/task-1/` - First task
- `tasks/task-2/` - Second task
- etc.

Each task directory contains:
- `task.md` - The task description and metadata
- `memory.md` - Detailed memory for that specific task

### High-Level Memory

`tasks/memory.md` contains only high-level summaries:
- Key milestones and completions
- Important decisions
- Brief status updates

Detailed technical information belongs in task-specific memory files.

### Telegram Chat History

`context/CHAT.md` contains the complete history of all Telegram conversations:
- **Purpose**: Provides full context of all user interactions and communications via Telegram
- **Content**: All user messages, commands, and assistant responses with timestamps
- **Usage**: This history is automatically included in all agent calls to maintain conversation continuity
- **Updates**: Automatically updated as new messages are sent and received through Telegram
- **Context**: Helps the agent understand user preferences, ongoing discussions, and the full context of requests across multiple agent calls

The chat history is particularly important because:
- It preserves the full conversation context that might not be captured in task memory
- It shows the evolution of user requests and how they relate to each other
- It helps maintain consistency in responses and understanding user intent
- It provides a complete audit trail of all Telegram interactions

## Memory Writing Guidelines

### Task Memory (`tasks/task-{id}/memory.md`)

Write detailed, specific information:
- Technical details: code changes, file paths, function names
- API responses and error messages
- Step-by-step actions taken
- Relevant context and decisions
- Implementation details

### High-Level Memory (`tasks/memory.md`)

Write concise summaries:
- What was accomplished (not how)
- Key milestones reached
- Task completions
- Important blockers or decisions

The system automatically extracts summaries from detailed task memory, but you can also write explicit high-level summaries when needed.

## Task Management

- Tasks are created via `/task` command in Telegram
- Each task gets a unique ID (`task-1`, `task-2`, etc.)
- Tasks are never deleted, only marked complete in their memory files
- Completion is detected by looking for phrases like "task complete" or "completed" in task memory
