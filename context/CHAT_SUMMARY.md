# Chat Summary

<!-- This file is automatically updated hourly by Tau to summarize the full conversation history. -->
<!-- It provides context to all agent calls so they understand the conversation without reading the full history. -->

## Summary

**System Context:**
- Docker container `a3dc454e3157` (IP: 172.24.0.2), Linux Ubuntu, 8x NVIDIA RTX A6000 GPUs (49GB each, CUDA 12.9, idle)

**Implemented Features:**
- CHAT_SUMMARY.md: Hourly auto-summarization using `composer-1` model (fixed from gemini-2.5-flash)
- `/plan` command: Creates timestamped plan files in `context/plans/`, returns contents and filename
- `/adapt` timeout: Increased to 24 hours (was 10 minutes)
- Summary included in all agent calls for context awareness
- Self-scheduling tools: `create_task` and `schedule_message` tools added to agent prompt for autonomous task creation and scheduled reminders
- Self-scheduling skill: Enhanced `context/skills/self-scheduling.md` with quick reference, long-range patterns (background jobs with checkpoints, deferred work, daily/weekly reviews), and decision tree
- Agent can create tasks for itself via `create_task` tool, schedule future messages via `schedule_message` (supports --at, --in, --cron), documented with examples and best practices

**Memory System (Three-Tier Hierarchical):**
- Short-term (`CHAT_SUMMARY.md`): Hourly incremental summaries, keeps last 5 versions
- Mid-term (`MID_TERM.md`): Compressed summaries from 7-30 days ago, auto-migrated from short-term
- Core (`CORE_MEMORY.md`): Persistent facts, preferences, decisions, updated rarely
- Flow: Chat → Short-term (hourly) → Mid-term (after 7 days, compressed) → Core (extracted facts after 30 days) → Archive
- Maintenance: Auto-runs on startup and daily for compression, migration, archiving

**Context Management Research:**
- Issues: Memory rot (unbounded growth, no versioning/compression), fixed-size summaries, full regeneration hourly
- Solutions: Hierarchical memory (short/mid/long-term), progressive compression, incremental updates, staleness detection
- Created `CONTEXT_MANAGEMENT_IMPROVEMENTS.md` with 3-phase implementation plan (low→medium→high risk)
- Plans: `plan-20260205-135754-reviewing-the-current-context-management-implement.md` (hierarchical memory), `plan-20260205-161048-add-to-the-agent-context-the-knowledge-that-if-it.md` (self-scheduling tools - implemented)

**Bot Architecture:**
- Telegram bot, agent loop (5min intervals), commands: /task, /status, /adapt, /cron, /crons, /uncron, /clear, /restart, /kill, /debug, /plan
- `/adapt` can run concurrently but unsafe (no queue/locking, restarts kill other adapts)
- Memory: task-specific (`context/tasks/task-{id}/memory.md`), high-level (`context/tasks/memory.md`), chat (`context/CHAT.md`)
- Agent tools: send_message, send_voice, search_skills, create_task, schedule_message (--at, --in, --cron)
- Agent authorized to create/modify/remove cron jobs and tasks for autonomous operations (self-messaging, reminders, scheduled actions, long-range background work)

**User Preferences:**
- Summaries: <2000 chars, bullet points, actionable info
- Focus: Memory rot prevention, context management improvements, phased low-risk implementations

---
*Last updated: 2026-02-05 16:15*
