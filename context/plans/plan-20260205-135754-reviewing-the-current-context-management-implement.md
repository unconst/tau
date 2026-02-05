# Plan: Reviewing the current context management implementation, then researching best practices.

Reviewing the summary loop implementation and researching context management best practices:

Searching for more specific research on context management and memory rot prevention:

Drafting an analysis and improvement plan:

Creating an analysis and improvement plan:

## Findings

Current issues:
1. Summary overwrites entire file — no versioning or rollback
2. High-level memory grows unbounded — no compression/archival
3. No staleness detection — old info never expires
4. Fixed-size summaries — doesn't adapt to information density
5. No incremental updates — full regeneration every hour

Research-backed solutions:
1. Hierarchical memory (MemGPT/MemoryOS): Short-term → Mid-term → Long-term tiers
2. Progressive compression (Livia): Older entries compressed more aggressively
3. Incremental updates (Mem0): Only update changed sections, not full regeneration
4. Memory entropy (Dynamic Affective Memory): Track staleness and importance
5. Graph memory (Mem0 Graph): Link related memories for better retrieval

Recommended immediate improvements:
1. Summary versioning — keep last 5 versions for rollback
2. High-level memory compression — archive entries >90 days, keep last 50 active
3. Incremental summary updates — compare new chat to old summary, update only changes
4. Staleness detection — auto-archive entries >30 days with no recent references

Created detailed implementation plan in context/CONTEXT_MANAGEMENT_IMPROVEMENTS.md with:
- Code examples for each improvement
- Phased rollout plan (low → medium → high risk)
- Metrics to track
- Testing strategy
- Risk mitigation

Should I implement any of these improvements? I recommend starting with Phase 1 (versioning and compression) as they're low-risk and address the main memory rot issues.

Implement a plan for the long short medium memory design.

*Generated: 2026-02-05 13:59:24*

---

Now let me also read the `__init__.py` to understand the summary loop implementation.Now I have a complete picture of the current implementation. Let me create the comprehensive execution plan.

# Execution Plan: Tiered Memory System (Short/Medium/Long-term)

## Goal

Implement a hierarchical memory architecture with three tiers—**short-term**, **mid-term**, and **long-term**—to prevent memory rot, enable progressive compression, and optimize context window usage. This replaces the current single-summary approach with a research-backed tiered system inspired by MemGPT, Mem0, and cognitive memory models.

---

## Prerequisites

1. **Code understanding**: Familiarity with the existing files:
   - `tau/__init__.py` - Contains `run_summary_loop()` (lines 125-244)
   - `tau/agent.py` - Contains memory functions and `run_loop()`
   - `context/CHAT_SUMMARY.md` - Current summary file
   - `context/tasks/memory.md` - High-level memory
   - `context/MEMORY-SYSTEM.md` - Memory documentation

2. **Backup**: Git commit of current state before changes

3. **Python environment**: Virtual environment with existing dependencies (`uv`, `openai`)

4. **Testing ability**: Access to Telegram bot for end-to-end testing

---

## Steps

### Phase 1: Foundation — Summary Versioning and Compression

#### Step 1.1: Create directory structure

Create new directories for tiered memory:
```bash
mkdir -p context/summaries
mkdir -p context/archive
mkdir -p context/memory
```

#### Step 1.2: Add versioning to summary generation

Modify `run_summary_loop()` in `tau/__init__.py` to save versioned summaries:

```python
# Add near top of file with other constants
SUMMARY_VERSIONS_DIR = os.path.join(WORKSPACE, "context", "summaries")
MAX_SUMMARY_VERSIONS = 5

def save_summary_with_version(summary: str):
    """Save summary with versioning for rollback capability."""
    os.makedirs(SUMMARY_VERSIONS_DIR, exist_ok=True)
    
    # Save new version with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_file = os.path.join(SUMMARY_VERSIONS_DIR, f"summary_{timestamp}.md")
    with open(version_file, "w") as f:
        f.write(summary)
    
    # Update main summary file
    with open(CHAT_SUMMARY_FILE, "w") as f:
        f.write(summary)
    
    # Rotate old versions (keep only MAX_SUMMARY_VERSIONS)
    versions = sorted(Path(SUMMARY_VERSIONS_DIR).glob("summary_*.md"))
    for old_version in versions[:-MAX_SUMMARY_VERSIONS]:
        old_version.unlink()
    
    logger.info(f"Summary version saved: {version_file}")
```

Update `run_summary_loop()` to call `save_summary_with_version(summary_content)` instead of writing directly.

#### Step 1.3: Add high-level memory compression

Add to `tau/agent.py`:

```python
from datetime import timedelta

ARCHIVE_DIR = os.path.join(WORKSPACE, "context", "archive")
MAX_ACTIVE_ENTRIES = 50
ARCHIVE_AGE_DAYS = 90

def compress_high_level_memory():
    """Archive old entries and keep only recent active memory."""
    if not os.path.exists(MEMORY_FILE):
        return
    
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    content = read_file(MEMORY_FILE)
    
    cutoff_date = datetime.now() - timedelta(days=ARCHIVE_AGE_DAYS)
    archive_entries = []
    active_entries = []
    
    # Parse entries (format: ### YYYY-MM-DD HH:MM\ncontent)
    current_entry = []
    entry_date = None
    
    for line in content.split("\n"):
        if line.startswith("### "):
            if current_entry and entry_date:
                if entry_date < cutoff_date:
                    archive_entries.append("\n".join(current_entry))
                else:
                    active_entries.append("\n".join(current_entry))
            current_entry = [line]
            try:
                date_str = line[4:].strip()[:16]  # "YYYY-MM-DD HH:MM"
                entry_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
            except:
                entry_date = datetime.now()
        elif current_entry:
            current_entry.append(line)
    
    # Handle last entry
    if current_entry and entry_date:
        if entry_date < cutoff_date:
            archive_entries.append("\n".join(current_entry))
        else:
            active_entries.append("\n".join(current_entry))
    
    # Archive old entries
    if archive_entries:
        archive_file = os.path.join(ARCHIVE_DIR, f"memory_{datetime.now().strftime('%Y%m')}.md")
        with open(archive_file, "a") as f:
            f.write("\n\n".join(archive_entries) + "\n")
    
    # Keep only last MAX_ACTIVE_ENTRIES
    active_entries = active_entries[-MAX_ACTIVE_ENTRIES:]
    
    # Rewrite active memory
    with open(MEMORY_FILE, "w") as f:
        f.write("# Memory\n\n<!-- High-level summaries only -->\n\n")
        f.write("\n\n".join(active_entries))
```

Call `compress_high_level_memory()` at the start of `run_loop()` (once per startup).

---

### Phase 2: Three-Tier Memory System

#### Step 2.1: Create memory tier files

Create `context/memory/CORE_MEMORY.md` for long-term facts:

```markdown
# Core Memory

<!-- Persistent facts, preferences, and decisions. Updated rarely. -->

## User Preferences
<!-- Updated when user explicitly states preferences -->

## Key Decisions
<!-- Major architectural or project decisions -->

## Learned Patterns
<!-- Reusable patterns discovered during operation -->
```

Create `context/memory/MID_TERM.md` for mid-term context:

```markdown
# Mid-Term Memory

<!-- Recent activity summaries (last 7-30 days). Auto-compressed from short-term. -->
```

#### Step 2.2: Implement tier migration functions

Add to `tau/__init__.py`:

```python
CORE_MEMORY_FILE = os.path.join(WORKSPACE, "context", "memory", "CORE_MEMORY.md")
MID_TERM_FILE = os.path.join(WORKSPACE, "context", "memory", "MID_TERM.md")
SHORT_TERM_DAYS = 7
MID_TERM_DAYS = 30

def extract_core_facts(summary_content: str) -> list[str]:
    """Extract important facts that should be stored in core memory."""
    # Use agent to identify core facts
    extract_prompt = f"""Analyze this summary and extract ONLY persistent facts that:
1. User preferences or requirements
2. Important decisions made
3. Technical constraints discovered
4. Recurring patterns

Summary:
{summary_content}

Return a bullet list of core facts, or "NONE" if no new core facts.
Be very selective - only truly persistent information."""
    
    # Run via agent in ask mode
    cmd = ["agent", "--force", "--model", "composer-1", "--mode=ask", 
           "--output-format=text", "--print", extract_prompt]
    result = subprocess.run(cmd, capture_output=True, text=True, 
                           cwd=WORKSPACE, timeout=120)
    
    facts = result.stdout.strip()
    if facts == "NONE" or not facts:
        return []
    
    return [f.strip() for f in facts.split("\n") if f.strip().startswith("-")]

def migrate_to_tiers():
    """Migrate old summaries through memory tiers."""
    # Get all versioned summaries
    versions = sorted(Path(SUMMARY_VERSIONS_DIR).glob("summary_*.md"))
    
    for version_file in versions:
        # Parse timestamp from filename
        timestamp_str = version_file.stem.replace("summary_", "")
        try:
            file_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except:
            continue
        
        age_days = (datetime.now() - file_date).days
        
        if age_days > MID_TERM_DAYS:
            # Extract core facts before archiving
            content = version_file.read_text()
            core_facts = extract_core_facts(content)
            
            if core_facts:
                with open(CORE_MEMORY_FILE, "a") as f:
                    f.write(f"\n### Extracted {datetime.now().strftime('%Y-%m-%d')}\n")
                    f.write("\n".join(core_facts) + "\n")
            
            # Archive the summary
            archive_file = os.path.join(ARCHIVE_DIR, f"summaries_{file_date.strftime('%Y%m')}.md")
            with open(archive_file, "a") as f:
                f.write(f"\n---\n{content}\n")
            
            version_file.unlink()
        
        elif age_days > SHORT_TERM_DAYS:
            # Move to mid-term (compress)
            content = version_file.read_text()
            compressed = compress_summary(content)
            
            with open(MID_TERM_FILE, "a") as f:
                f.write(f"\n### {file_date.strftime('%Y-%m-%d')}\n{compressed}\n")

def compress_summary(content: str, target_ratio: float = 0.5) -> str:
    """Compress a summary to target ratio of original size."""
    target_chars = int(len(content) * target_ratio)
    
    compress_prompt = f"""Compress this summary to approximately {target_chars} characters.
Keep only the most important information.

Original:
{content}

Output ONLY the compressed summary."""
    
    cmd = ["agent", "--force", "--model", "composer-1", "--mode=ask",
           "--output-format=text", "--print", compress_prompt]
    result = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=WORKSPACE, timeout=120)
    
    return result.stdout.strip()[:target_chars + 100]  # Allow slight overage
```

#### Step 2.3: Update prompt template to use tiers

Modify `PROMPT_TEMPLATE` in `tau/agent.py` to include tiered memory:

```python
PROMPT_TEMPLATE = """You are Tau, a single-threaded autonomous agent.

{identity}

{memory_rules}

---

CORE MEMORY (persistent facts):
{core_memory}

MID-TERM MEMORY (recent weeks):
{mid_term_memory}

CONVERSATION SUMMARY (auto-updated hourly):
{chat_summary}

TELEGRAM CHAT (recent):
{chat_history}

INCOMPLETE TASKS:
{tasks}

HIGH-LEVEL MEMORY (recent):
{high_level_memory}

CURRENT TASK MEMORY (recent):
{task_memory}

---
...
"""
```

Update the `run_loop()` function to read and inject the new memory tiers.

---

### Phase 3: Incremental Updates and Staleness Detection

#### Step 3.1: Implement incremental summary updates

Replace full regeneration with incremental updates in `run_summary_loop()`:

```python
def generate_incremental_summary(old_summary: str, new_chat_delta: str) -> str:
    """Generate summary incrementally, updating only changed parts."""
    prompt = f"""Compare the existing summary with new chat history.
Update the summary to incorporate new information.

EXISTING SUMMARY:
{old_summary}

NEW CHAT HISTORY (since last update):
{new_chat_delta}

Rules:
1. Preserve unchanged information
2. Add new important information  
3. Update information that has changed
4. Remove outdated information
5. Keep under 2000 characters

Output ONLY the updated summary."""

    cmd = ["agent", "--force", "--model", "composer-1", "--mode=ask",
           "--output-format=text", "--print", prompt]
    result = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=WORKSPACE, timeout=300)
    return result.stdout.strip()
```

Track last-processed chat position to only process new messages:

```python
LAST_CHAT_POSITION_FILE = os.path.join(WORKSPACE, "context", ".chat_position")

def get_new_chat_since_last_summary() -> str:
    """Get chat history added since last summary generation."""
    last_pos = 0
    if os.path.exists(LAST_CHAT_POSITION_FILE):
        with open(LAST_CHAT_POSITION_FILE) as f:
            try:
                last_pos = int(f.read().strip())
            except:
                pass
    
    with open(CHAT_HISTORY_FILE) as f:
        full_chat = f.read()
    
    new_chat = full_chat[last_pos:]
    
    # Save new position
    with open(LAST_CHAT_POSITION_FILE, "w") as f:
        f.write(str(len(full_chat)))
    
    return new_chat
```

#### Step 3.2: Add staleness detection

```python
def detect_and_mark_stale_entries():
    """Detect memory entries that haven't been referenced recently."""
    if not os.path.exists(MEMORY_FILE):
        return
    
    content = read_file(MEMORY_FILE)
    recent_chat = get_chat_history(max_lines=500)
    
    # Parse entries and check staleness
    stale_entries = []
    active_entries = []
    
    current_entry = []
    entry_date = None
    entry_content = ""
    
    for line in content.split("\n"):
        if line.startswith("### "):
            if current_entry:
                # Check if entry is stale
                age_days = (datetime.now() - entry_date).days if entry_date else 0
                is_referenced = any(
                    keyword.lower() in recent_chat.lower() 
                    for keyword in entry_content.split()[:10]  # Check first 10 words
                )
                
                if age_days > 30 and not is_referenced:
                    stale_entries.append("\n".join(current_entry))
                else:
                    active_entries.append("\n".join(current_entry))
            
            current_entry = [line]
            try:
                date_str = line[4:].strip()[:16]
                entry_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
            except:
                entry_date = datetime.now()
            entry_content = ""
        elif current_entry:
            current_entry.append(line)
            entry_content += line + " "
    
    # Handle last entry
    if current_entry:
        active_entries.append("\n".join(current_entry))
    
    # Archive stale entries
    if stale_entries:
        archive_file = os.path.join(ARCHIVE_DIR, f"stale_{datetime.now().strftime('%Y%m%d')}.md")
        with open(archive_file, "w") as f:
            f.write("# Archived Stale Entries\n\n")
            f.write("\n\n".join(stale_entries))
        
        # Rewrite active memory
        with open(MEMORY_FILE, "w") as f:
            f.write("# Memory\n\n")
            f.write("\n\n".join(active_entries))
        
        logger.info(f"Archived {len(stale_entries)} stale memory entries")
```

---

### Phase 4: Integration and Testing

#### Step 4.1: Add maintenance loop

Create a daily maintenance routine in `tau/__init__.py`:

```python
def run_memory_maintenance_loop(stop_event=None):
    """Background loop for memory maintenance (runs daily)."""
    MAINTENANCE_INTERVAL = 86400  # 24 hours
    
    while True:
        if stop_event and stop_event.is_set():
            break
        
        try:
            logger.info("Running daily memory maintenance...")
            
            # 1. Compress high-level memory
            compress_high_level_memory()
            
            # 2. Migrate through tiers
            migrate_to_tiers()
            
            # 3. Detect and archive stale entries
            detect_and_mark_stale_entries()
            
            logger.info("Memory maintenance complete")
            
        except Exception as e:
            logger.error(f"Memory maintenance error: {e}")
        
        time.sleep(MAINTENANCE_INTERVAL)
```

Start this in `main()`:

```python
# Start memory maintenance loop
maintenance_thread = threading.Thread(
    target=run_memory_maintenance_loop,
    args=(_stop_event,),
    daemon=True,
    name="MemoryMaintenance"
)
maintenance_thread.start()
```

#### Step 4.2: Update documentation

Update `context/MEMORY-SYSTEM.md` to reflect the new three-tier architecture.

#### Step 4.3: Testing checklist

1. **Versioning test**: Verify summaries are saved with timestamps in `context/summaries/`
2. **Compression test**: Add 60+ entries to memory, verify compression to 50
3. **Tier migration test**: Create old summaries, run maintenance, verify migration
4. **Staleness test**: Add entries >30 days old, verify archival
5. **Integration test**: Run full bot for 24+ hours, verify all loops working

---

## Success Criteria

1. **Summary versioning works**: `context/summaries/` contains timestamped summaries, max 5 retained
2. **Compression works**: `context/tasks/memory.md` never exceeds 50 active entries
3. **Three tiers exist and populate**:
   - `context/CHAT_SUMMARY.md` — Short-term (hourly updates)
   - `context/memory/MID_TERM.md` — Mid-term (7-30 days, compressed)
   - `context/memory/CORE_MEMORY.md` — Long-term (persistent facts)
4. **Archive directory grows**: `context/archive/` contains old entries
5. **No data loss**: Archived data is readable and recoverable
6. **Agent context includes all tiers**: Prompt template injects core/mid-term memory
7. **Incremental updates work**: Summary changes are deltas, not full regeneration

---

## Potential Issues

| Risk | Mitigation |
|------|------------|
| **Data loss during migration** | Always archive before deletion; test with copies first |
| **Agent timeout on large summaries** | Set 300s timeout; chunk large summaries |
| **Compression loses important info** | Keep originals in archive; tune extraction prompts |
| **Memory files grow despite compression** | Monitor file sizes; add hard limits |
| **Core memory polluted with noise** | Use strict extraction criteria; manual review initially |
| **Race conditions in loops** | Use file locks for concurrent access |
| **Agent API costs increase** | Use cheaper models for compression; cache results |
| **Supervisor restarts during maintenance** | Make operations idempotent; use atomic writes |

---

## Rollback Plan

If issues arise:

1. Stop the bot: `/kill` or `supervisorctl stop tau`
2. Restore from git: `git checkout -- tau/__init__.py tau/agent.py`
3. Restore summary: Copy latest from `context/summaries/` to `CHAT_SUMMARY.md`
4. Remove new directories if desired: `rm -rf context/memory context/summaries context/archive`
5. Restart: `tauctl start`
