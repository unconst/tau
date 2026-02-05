# Self-Scheduling: Tasks, Reminders, and Cron Jobs

Tau has the ability to send messages to itself in the future by creating tasks and scheduling cron jobs. This enables autonomous follow-up, reminders, and scheduled operations.

## Quick Reference

| Action | Command |
|--------|---------|
| Create task | `python -m tau.tools.create_task "Title" "Body"` |
| Message in 2h | `python -m tau.tools.schedule_message --in "2h" "msg"` |
| Message at time | `python -m tau.tools.schedule_message --at "14:00" "msg"` |
| Daily at 9am | `python -m tau.tools.schedule_message --cron "0 9 * * *" "msg"` |
| View scheduled | `atq` (at jobs), `crontab -l` (cron) |

## Creating Tasks for Yourself

To send yourself a message or reminder for later processing, create a task in the `context/tasks/` directory.

### Task Structure

Each task lives in its own directory: `context/tasks/task-{id}/`

Required files:
- `task.md` - Task description (the "message" to yourself)
- `memory.md` - Gets created automatically for tracking progress

### Creating a Self-Task

```bash
# Find the next available task ID
NEXT_ID=$(ls context/tasks/ | grep -oP 'task-\K\d+' | sort -n | tail -1)
NEXT_ID=$((NEXT_ID + 1))

# Create task directory
mkdir -p context/tasks/task-$NEXT_ID

# Create task.md with your message/reminder
cat > context/tasks/task-$NEXT_ID/task.md << 'EOF'
## [Task Title - describe what you need to do]

[Detailed instructions, context, or reminder message]

This is a self-scheduled task created on $(date).
EOF
```

### Example: Reminder to Check Something

```bash
mkdir -p context/tasks/task-42
cat > context/tasks/task-42/task.md << 'EOF'
## Check GPU instance status and report to user

The user wanted to know how their GPU training job is progressing.
Check the instance status and send a summary via Telegram.

Context: Training job was started earlier today on instance abc123.
EOF
```

## Creating Cron Jobs

For recurring or time-scheduled tasks, create cron jobs directly.

### Permission Granted

You are authorized to create, modify, and remove cron jobs for:
- Scheduled reminders
- Periodic maintenance tasks
- Timed notifications
- Any recurring autonomous operations

### Creating a Cron Job

```bash
# Edit crontab
crontab -e

# Or append directly:
(crontab -l 2>/dev/null; echo "0 9 * * * cd /root/tau && source .venv/bin/activate && python -m tau.tools.send_message 'Good morning! Daily check-in reminder.'") | crontab -
```

### Cron Format Reference

```
* * * * * command
│ │ │ │ │
│ │ │ │ └── Day of week (0-7, Sunday=0 or 7)
│ │ │ └──── Month (1-12)
│ │ └────── Day of month (1-31)
│ └──────── Hour (0-23)
└────────── Minute (0-59)
```

### Common Cron Patterns

| Pattern | Description |
|---------|-------------|
| `0 9 * * *` | Every day at 9:00 AM |
| `0 */4 * * *` | Every 4 hours |
| `30 8 * * 1-5` | Weekdays at 8:30 AM |
| `0 0 1 * *` | First day of each month at midnight |
| `*/15 * * * *` | Every 15 minutes |

### Example Cron Jobs for Self-Messaging

**Daily morning reminder:**
```bash
(crontab -l 2>/dev/null; echo "0 9 * * * cd /root/tau && source .venv/bin/activate && python -m tau.tools.send_message 'Morning check: Review pending tasks and priorities.'") | crontab -
```

**One-time delayed reminder (using `at`):**
```bash
echo "cd /root/tau && source .venv/bin/activate && python -m tau.tools.send_message 'Reminder: Follow up on X'" | at now + 2 hours
```

**Create a task via cron for future processing:**
```bash
(crontab -l 2>/dev/null; echo "0 18 * * 5 cd /root/tau && mkdir -p context/tasks/task-weekly-review && echo '## Weekly Review\n\nSummarize the week activities and send to user.' > context/tasks/task-weekly-review/task.md") | crontab -
```

### Viewing Current Cron Jobs

```bash
crontab -l
```

### Removing a Cron Job

```bash
# Edit and remove the line
crontab -e

# Or remove all cron jobs (careful!)
crontab -r
```

## Using `at` for One-Time Scheduled Messages

For one-time delayed execution, use the `at` command:

```bash
# Run in 30 minutes
echo "cd /root/tau && source .venv/bin/activate && python -m tau.tools.send_message 'Reminder: Check on X'" | at now + 30 minutes

# Run at a specific time
echo "cd /root/tau && source .venv/bin/activate && python -m tau.tools.send_message 'Time for the meeting!'" | at 14:00

# Run tomorrow at 9am
echo "cd /root/tau && source .venv/bin/activate && python -m tau.tools.send_message 'Follow up on yesterday request'" | at 9:00 tomorrow
```

### Viewing Scheduled `at` Jobs

```bash
atq
```

### Removing an `at` Job

```bash
atrm <job_number>
```

## Tools Summary

| Method | Use Case | Persistence |
|--------|----------|-------------|
| Create task file | Self-reminder processed by agent loop | Until task completed |
| Cron job | Recurring scheduled actions | Persistent until removed |
| `at` command | One-time delayed action | Single execution |
| `send_message` | Immediate notification | N/A |

## Long-Range Task Patterns

### Pattern 1: Background Job with Checkpoints
When starting a long-running operation:
1. Create initial task with full context
2. Schedule checkpoint reminders at intervals
3. Each checkpoint creates a follow-up task if work continues

Example: Training a model
```bash
# Create the monitoring task
python -m tau.tools.create_task "Monitor training job X" "Instance: abc123. Started at $(date). Check progress every 2 hours."

# Schedule periodic check-ins
python -m tau.tools.schedule_message --in "2h" "Checkpoint: Check training job X progress"
```

### Pattern 2: Deferred Work
When you can't complete something now:
```bash
python -m tau.tools.create_task "Complete Y when Z is ready" "Context: [full context]. Waiting for: [dependency]"
```

### Pattern 3: Daily/Weekly Reviews
```bash
# Weekly project review
python -m tau.tools.schedule_message --cron "0 18 * * 5" "Weekly review: Summarize progress and plan next week"
```

### Self-Messaging Decision Tree
- **Immediate action needed?** → Execute now
- **Need to wait for external event?** → Create task with trigger condition
- **Time-based follow-up?** → Use `at` or cron
- **Complex multi-step work?** → Create task, schedule checkpoints

## Best Practices

1. **Prefer tasks for complex reminders** - If the follow-up requires agent reasoning, create a task rather than just a message.

2. **Use cron for recurring patterns** - Daily standups, weekly reviews, periodic checks.

3. **Use `at` for one-time delays** - "Remind me in 2 hours", "Check on this tomorrow".

4. **Include context in task descriptions** - When creating a self-task, include all necessary context so you'll understand it later.

5. **Clean up completed cron jobs** - Remove one-time or temporary cron entries after they're no longer needed.

6. **Log important scheduled actions** - Consider noting scheduled reminders in your memory so you maintain awareness of what's pending.
