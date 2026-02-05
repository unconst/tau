# Tau Memory System

Tau uses a hierarchical three-tier memory model to prevent memory rot and optimize context usage.

Your logs are in `context/logs`.

## Memory Tiers

### 1. Core Memory (Long-term)

**Path:** `context/memory/CORE_MEMORY.md`

**Purpose:**
- Persistent facts, preferences, and decisions
- User preferences that don't change often
- Key architectural decisions
- Learned patterns and recurring themes

**Rules:**
- Updated rarely (only when significant persistent facts are discovered)
- Extracted automatically from old summaries during tier migration
- Manually editable for important corrections
- Never automatically deleted

**Lifespan:** Permanent

---

### 2. Mid-Term Memory

**Path:** `context/memory/MID_TERM.md`

**Purpose:**
- Compressed summaries of recent weeks (7-30 days)
- Context from the recent past that's still relevant
- Auto-migrated from short-term summaries

**Rules:**
- Auto-populated from summaries older than 7 days
- Compressed to ~50% of original size
- Migrated to archive after 30 days
- Provides continuity without bloating context

**Lifespan:** 7-30 days

---

### 3. Short-Term Memory (Chat Summary)

**Path:** `context/CHAT_SUMMARY.md`

**Purpose:**
- Hourly-updated summary of recent conversation
- Current context, ongoing tasks, recent decisions
- What the user is working on right now

**Rules:**
- Updated hourly by the summary loop
- Uses incremental updates when possible (only processes new chat)
- Versioned in `context/summaries/` for rollback capability
- Keeps last 5 versions

**Lifespan:** 0-7 days

---

### 4. Task Memory (Working Memory)

**Path:** `context/tasks/task-{id}/memory.md`

**Purpose:**
- Detailed, technical, chronological record of task execution
- Code changes, commands, errors, decisions
- Active working memory for current tasks

**Rules:**
- Append-only during task execution
- Never repeat prior actions
- Mark completion explicitly when done
- Cleaned up after task completion (archived)

**Lifespan:** Duration of task

---

### 5. High-Level Memory

**Path:** `context/tasks/memory.md`

**Purpose:**
- Concise summaries of completed work
- Milestones, completions, major decisions
- Quick reference for recent activity

**Rules:**
- No implementation detail
- One or two sentences per entry
- Auto-compressed when exceeds 50 entries
- Old entries (>90 days) archived automatically

**Lifespan:** 0-90 days (active), then archived

---

## Memory Flow

```
Chat History → Short-Term Summary (hourly)
                    ↓
               (after 7 days)
                    ↓
              Mid-Term Memory (compressed)
                    ↓
               (after 30 days)
                    ↓
         Core Facts Extracted → Core Memory
                    ↓
              Archive (full text)
```

## Maintenance

Memory maintenance runs automatically:
- **On startup:** Initial cleanup and compression
- **Daily:** Full maintenance cycle
  - Compress high-level memory (archive >90 days, keep 50 active)
  - Migrate summaries through tiers
  - Detect and archive stale entries

## Directory Structure

```
context/
├── CHAT.md                    # Full chat history (never edited)
├── CHAT_SUMMARY.md            # Current short-term summary
├── MEMORY-SYSTEM.md           # This file
├── IDENTITY.md                # Agent identity
├── .chat_position             # Tracks incremental summary position
├── memory/
│   ├── CORE_MEMORY.md         # Long-term persistent facts
│   └── MID_TERM.md            # Mid-term compressed summaries
├── summaries/
│   └── summary_YYYYMMDD_HHMMSS.md  # Versioned summaries (max 5)
├── archive/
│   ├── memory_YYYYMM.md       # Archived high-level memory
│   ├── summaries_YYYYMM.md    # Archived summaries
│   └── stale_YYYYMMDD.md      # Archived stale entries
├── tasks/
│   ├── memory.md              # High-level activity memory
│   └── task-{id}/
│       ├── task.md            # Task description
│       └── memory.md          # Task-specific working memory
└── logs/
    └── tau.log                # Application logs
```

## Rollback

If issues arise with memory:

1. **Summary rollback:** Check `context/summaries/` for previous versions
2. **Memory recovery:** Check `context/archive/` for archived entries
3. **Full reset:** Remove tier files and let them regenerate from chat history
