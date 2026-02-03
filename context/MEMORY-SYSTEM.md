# Tau Memory System

Tau uses a hierarchical memory model.

## Memory Layers

### 1. Task Memory

**Path:** `context/tasks/task-{id}/memory.md`

**Purpose:**
- Detailed, technical, chronological record of task execution
- Code changes, commands, errors, decisions

**Rules:**
- Append-only
- Never repeat prior actions
- Mark completion explicitly when done

**Task Directory Contents:**
- `task.md` - Task description and metadata
- `memory.md` - Detailed memory for that task
- Other files - Task-specific outputs, scripts, data

---

### 2. High-Level Memory

**Path:** `context/tasks/memory.md`

**Purpose:**
- Concise summaries only
- Milestones, completions, major decisions

**Rules:**
- No implementation detail
- One or two sentences per entry
- Derived from task memory

---

### 3. Chat Memory

**Path:** `context/CHAT.md`

**Purpose:**
- Full Telegram conversation history
- User intent, preferences, continuity

**Rules:**
- Never edited or summarized
- Used only for understanding context
- Automatically updated by the system
