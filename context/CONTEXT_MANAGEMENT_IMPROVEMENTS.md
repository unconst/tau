# Context Management Improvements for Tau

## Current System Analysis

### Architecture
- **CHAT_SUMMARY.md**: Hourly overwrite of full conversation summary (2000 char limit)
- **CHAT.md**: Append-only full conversation history (never edited)
- **tasks/memory.md**: Append-only high-level summaries (grows unbounded)
- **tasks/task-{id}/memory.md**: Task-specific detailed memory (cleaned after completion)

### Identified Issues

1. **Memory Rot Risks**
   - Summary overwrites entire file - no versioning or staleness detection
   - High-level memory grows unbounded (no compression/archival)
   - No importance scoring or relevance decay
   - Old information never expires or consolidates

2. **Context Window Inefficiency**
   - Fixed-size summaries don't adapt to information density
   - No semantic retrieval - always loads full summary
   - No hierarchical memory tiers (all context treated equally)

3. **Update Mechanisms**
   - Hourly summary is time-based, not event-based
   - No incremental updates - full regeneration each time
   - No memory consolidation/merging of related information

4. **Retrieval Limitations**
   - No semantic search for relevant context
   - No recency weighting in memory access
   - No dynamic context selection based on query

## Research-Based Solutions

### 1. Hierarchical Memory Architecture (MemGPT/MemoryOS)

**Three-Tier System:**
- **Short-term**: Last 20-50 messages (immediate context)
- **Mid-term**: Recent summaries (last 7-30 days, compressed)
- **Long-term**: Core facts, preferences, decisions (highly compressed, rarely changed)

**Implementation:**
- Keep CHAT.md as short-term (already working)
- CHAT_SUMMARY.md becomes mid-term (with versioning)
- New `context/CORE_MEMORY.md` for long-term facts

### 2. Progressive Memory Compression (Livia/TBC)

**Temporal Binary Compression:**
- Older entries compressed more aggressively
- Recent entries keep detail, older entries become summaries
- Automatic archival of entries older than threshold

**Dynamic Importance Memory Filter:**
- Score memory entries by:
  - Recency (exponential decay)
  - User interaction frequency
  - Task completion relevance
  - Cross-references (how often cited)

### 3. Incremental Summary Updates (Mem0)

**Instead of full regeneration:**
- Compare new chat history to previous summary
- Extract only new/updated information
- Merge incrementally with existing summary
- Detect contradictions and resolve them

**Versioning:**
- Keep last N summary versions for rollback
- Track what changed between versions
- Archive old versions with timestamps

### 4. Memory Entropy & Staleness Detection (Dynamic Affective Memory)

**Bayesian-inspired updates:**
- Track memory "entropy" (information value)
- Low entropy = stale/redundant information
- High entropy = new/important information
- Automatically refresh high-entropy stale memories

**Staleness indicators:**
- Age threshold (e.g., 30 days)
- No references in recent conversations
- Contradicted by newer information
- Low importance score

### 5. Graph-Based Memory (Mem0 Graph)

**Relational structure:**
- Link related memories (tasks, decisions, preferences)
- Track dependencies between memories
- Enable multi-hop reasoning
- Identify memory clusters for consolidation

### 6. Sleep-Time Compute (Letta)

**Background memory refinement:**
- During idle periods, analyze and compress memory
- Pre-compute likely context needs
- Consolidate redundant information
- Update importance scores

## Recommended Implementation Plan

### Phase 1: Immediate Improvements (Low Risk)

1. **Summary Versioning**
   - Keep last 5 summary versions in `context/summaries/`
   - Add version metadata (timestamp, size, change summary)
   - Enable rollback if summary quality degrades

2. **High-Level Memory Compression**
   - Archive entries older than 90 days to `context/archive/`
   - Keep only last 50 entries in active memory
   - Add compression summary for archived entries

3. **Incremental Summary Updates**
   - Compare new chat to previous summary
   - Only regenerate changed sections
   - Track what's new vs. what's unchanged

4. **Memory Size Monitoring**
   - Track growth rates of memory files
   - Alert when approaching limits
   - Auto-compress when thresholds exceeded

### Phase 2: Enhanced Memory Management (Medium Risk)

5. **Three-Tier Memory System**
   - Implement short/mid/long-term separation
   - Create CORE_MEMORY.md for persistent facts
   - Migrate important info from summaries to core

6. **Importance Scoring**
   - Score memory entries by recency, frequency, relevance
   - Use scores to prioritize what to keep/compress
   - Implement exponential decay for old entries

7. **Staleness Detection**
   - Track last access time for memory entries
   - Mark stale entries (>30 days, no references)
   - Auto-archive or compress stale entries

8. **Memory Consolidation**
   - Detect duplicate/redundant information
   - Merge related entries
   - Remove contradictions (keep newer)

### Phase 3: Advanced Features (Higher Risk)

9. **Semantic Retrieval**
   - Embed memory entries for semantic search
   - Retrieve only relevant context for queries
   - Dynamic context window based on query needs

10. **Graph Memory Structure**
    - Link related memories
    - Track dependencies
    - Enable multi-hop reasoning

11. **Sleep-Time Compute**
    - Background memory refinement during idle
    - Pre-compute likely context needs
    - Continuous importance score updates

## Specific Code Changes

### 1. Summary Versioning (`tau/__init__.py`)

```python
SUMMARY_VERSIONS_DIR = os.path.join(WORKSPACE, "context", "summaries")
MAX_SUMMARY_VERSIONS = 5

def save_summary_with_version(summary: str):
    """Save summary with versioning."""
    os.makedirs(SUMMARY_VERSIONS_DIR, exist_ok=True)
    
    # Save new version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_file = os.path.join(SUMMARY_VERSIONS_DIR, f"summary_{timestamp}.md")
    with open(version_file, "w") as f:
        f.write(summary)
    
    # Update main summary
    with open(CHAT_SUMMARY_FILE, "w") as f:
        f.write(summary)
    
    # Clean old versions
    versions = sorted(Path(SUMMARY_VERSIONS_DIR).glob("summary_*.md"))
    for old_version in versions[:-MAX_SUMMARY_VERSIONS]:
        old_version.unlink()
```

### 2. Incremental Summary Updates

```python
def generate_incremental_summary(old_summary: str, new_chat: str) -> str:
    """Generate summary incrementally, only updating changed parts."""
    prompt = f"""Compare the old summary and new chat history.
    
OLD SUMMARY:
{old_summary}

NEW CHAT HISTORY (since last summary):
{new_chat}

Create an UPDATED summary that:
1. Preserves unchanged information from old summary
2. Adds new important information from new chat
3. Updates information that has changed
4. Removes information that is no longer relevant

Output ONLY the updated summary."""
    
    # Use agent to generate incremental update
    return run_cursor(prompt)
```

### 3. High-Level Memory Compression

```python
def compress_high_level_memory():
    """Archive old entries and compress active memory."""
    if not os.path.exists(MEMORY_FILE):
        return
    
    content = read_file(MEMORY_FILE)
    lines = content.split("\n")
    
    # Find entries older than 90 days
    cutoff_date = datetime.now() - timedelta(days=90)
    archive_lines = []
    active_lines = []
    
    current_entry = []
    in_entry = False
    
    for line in lines:
        if line.startswith("### "):
            # Parse timestamp
            timestamp_str = line[4:].strip()
            try:
                entry_date = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M")
                if entry_date < cutoff_date:
                    # Archive this entry
                    if current_entry:
                        archive_lines.extend(current_entry)
                    current_entry = [line]
                    in_entry = True
                else:
                    # Keep active
                    if current_entry:
                        active_lines.extend(current_entry)
                    current_entry = [line]
                    in_entry = True
            except:
                if in_entry:
                    current_entry.append(line)
        elif in_entry:
            current_entry.append(line)
    
    # Archive old entries
    if archive_lines:
        archive_dir = os.path.join(WORKSPACE, "context", "archive")
        os.makedirs(archive_dir, exist_ok=True)
        archive_file = os.path.join(archive_dir, f"memory_{datetime.now().strftime('%Y%m')}.md")
        with open(archive_file, "a") as f:
            f.write("\n".join(archive_lines))
    
    # Keep only last 50 active entries
    active_entries = []
    current_entry = []
    for line in active_lines:
        if line.startswith("### "):
            if current_entry:
                active_entries.append("\n".join(current_entry))
            current_entry = [line]
        else:
            current_entry.append(line)
    if current_entry:
        active_entries.append("\n".join(current_entry))
    
    # Write compressed memory
    with open(MEMORY_FILE, "w") as f:
        f.write("# Memory\n\n<!-- High-level summaries only. Detailed memory is in context/tasks/*/memory.md -->\n")
        for entry in active_entries[-50:]:
            f.write(entry + "\n\n")
```

### 4. Staleness Detection

```python
def detect_stale_memory_entries() -> list:
    """Detect memory entries that are stale and should be archived."""
    stale_entries = []
    
    # Check high-level memory
    if os.path.exists(MEMORY_FILE):
        content = read_file(MEMORY_FILE)
        entries = parse_memory_entries(content)
        
        for entry in entries:
            age_days = (datetime.now() - entry['timestamp']).days
            
            # Stale if: >30 days old AND not referenced in recent chat
            if age_days > 30:
                recent_chat = get_chat_history(max_lines=200)
                if entry['content'].lower() not in recent_chat.lower():
                    stale_entries.append(entry)
    
    return stale_entries
```

## Metrics to Track

1. **Memory Growth Rate**: Bytes/day for each memory file
2. **Summary Quality**: User feedback, coherence scores
3. **Context Window Usage**: Average tokens used per agent call
4. **Memory Access Patterns**: Which memories are accessed most
5. **Staleness Rate**: % of memory entries marked stale
6. **Compression Ratio**: Size reduction from compression

## Testing Strategy

1. **Unit Tests**: Memory compression, versioning, staleness detection
2. **Integration Tests**: End-to-end summary generation with versioning
3. **Load Tests**: Memory growth over time, compression effectiveness
4. **Quality Tests**: Summary coherence, information retention

## Rollout Plan

1. **Week 1**: Implement versioning and basic compression
2. **Week 2**: Add incremental summary updates
3. **Week 3**: Implement staleness detection and archival
4. **Week 4**: Monitor metrics and tune thresholds
5. **Month 2**: Implement three-tier system if metrics show benefit
6. **Month 3**: Add semantic retrieval if needed

## Risk Mitigation

- **Backup before changes**: Git commit before major changes
- **Feature flags**: Enable/disable new features via config
- **Gradual rollout**: Test on subset of memory first
- **Rollback plan**: Keep old code paths available
- **Monitoring**: Alert on memory growth anomalies
