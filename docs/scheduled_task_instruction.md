# Antigravity Memory Scheduled Worker Instruction

This document defines the execution protocol, operational context, reference links, and Definition of Done (DOD) for the AGY scheduled memory worker task.

## 1. System Context & Architecture

The Antigravity memory engine captures conversational insights and stores them in a multi-layer cognitive database (~/.gemini/memory.db) without blocking interactive sessions.

### Ingestion Pipeline

```
+---------------------+
| AGY Session Activity|
+---------------------+
           |
           v (PostInvocation / Stop Hook: 0 LLM tokens, ~5ms)
+-------------------------------------------------------------+
| scripts/auto_sync_hook.py -> ~/.gemini/turn_queue.db        |
+-------------------------------------------------------------+
           |
           v (Periodic AGY Scheduled Task / Cron)
+-------------------------------------------------------------+
| scripts/queue_cli.py status (evaluates 5m idle / 15m wait)  |
+-------------------------------------------------------------+
           |
           v (Claim batch)
+-------------------------------------------------------------+
| scripts/queue_cli.py claim --batch-size 25                  |
+-------------------------------------------------------------+
           |
           v (AGY Native LLM Cognitive Extraction)
+-------------------------------------------------------------+
| Facts, Episodes, Learnings, Entity Links                    |
+-------------------------------------------------------------+
           |
           v (Atomic Commit & Acknowledge)
+-------------------------------------------------------------+
| scripts/queue_cli.py commit OR FastMCP memory tools         |
| -> Updates ~/.gemini/memory.db (FTS5 + Vector Embeddings)   |
| -> Acknowledges ~/.gemini/turn_queue.db                     |
+-------------------------------------------------------------+
```

### Key Paths & Storage
- Memory Database: /Users/__blitzzz/.gemini/memory.db
- Turn Queue Database: /Users/__blitzzz/.gemini/turn_queue.db
- Engine Root: /Users/__blitzzz/Documents/GitHub/agy-memory-engine
- Python Virtualenv: /Users/__blitzzz/Documents/GitHub/agy-memory-engine/.venv/bin/python
- Hook Config: /Users/__blitzzz/.gemini/config/hooks.json

## 2. Code References

- Hook ingestion: [scripts/auto_sync_hook.py:45](file:///Users/__blitzzz/Documents/GitHub/agy-memory-engine/scripts/auto_sync_hook.py#L45)
- Queue CLI interface: [scripts/queue_cli.py:32](file:///Users/__blitzzz/Documents/GitHub/agy-memory-engine/scripts/queue_cli.py#L32)
- Queue manager core: [queue_manager.py:207](file:///Users/__blitzzz/Documents/GitHub/agy-memory-engine/queue_manager.py#L207)
- Memory persistence and upsert: [agy_memory.py:609](file:///Users/__blitzzz/Documents/GitHub/agy-memory-engine/agy_memory.py#L609)
- Cognitive layer definition & extraction: [agy_memory.py:870](file:///Users/__blitzzz/Documents/GitHub/agy-memory-engine/agy_memory.py#L870)
- Canonical taxonomy and category validation: [taxonomy.py:10](file:///Users/__blitzzz/Documents/GitHub/agy-memory-engine/taxonomy.py#L10)
- MCP tool interface: [agy_memory_mcp.py:243](file:///Users/__blitzzz/Documents/GitHub/agy-memory-engine/agy_memory_mcp.py#L243)
- Hook configuration entry: [hooks.json:121](file:///Users/__blitzzz/.gemini/config/hooks.json#L121)

## 3. Cognitive Extraction Standards

When processing conversation turns, extract only durable, reusable knowledge across four layers:

### Layer 1: Atomic Facts (facts)
- Description: Technical specs, IP addresses, server configs, personal preferences, account identifiers, definite appointments, medication/health data.
- Allowed categories: config, credential, device, finance, health, identity, infra, network, personal, project, service, tool, general.
- Rule: Keep facts concise and atomic. Never dump raw chat transcripts or transient bash commands into facts.

### Layer 2: Narrative Episodes (episodes)
- Description: Multi-turn project chronicles, relationship dynamics, background context, ongoing decisions, client stances.
- Statuses: active, cooling, historic, resolved.
- Allowed topics: family, health, travel, finance, home, dev, infra, insurance, music, work, realestate, trading, general.
- Rule: Write rich 2 to 4 sentence summaries capturing background context and user attitude.

### Layer 3: Experiential Learnings (learnings)
- Description: Personal heuristics, tested rules of thumb, behavioral patterns, operational insights.
- Allowed categories: architecture, automation, communication, finance, general, hardware, health, safety, security, shopping, travel, workflow.
- Critical filter: Pass the test "Would this insight help in a similar future situation?"
- Do NOT store one-time bug fixes, routine syntax corrections, git commit hashes, or code snippets as learnings.

### Layer 4: Entity Links (entity_links)
- Description: Directed relationships connecting memory keys.
- Canonical relations: hosted_on, runs_on, depends_on, part_of, member_of, monitors, uses, stores, related_to.

## 4. Execution Workflow for the Scheduled Task

Every scheduled task invocation executes the following sequence:

### Step 1: Pre-flight & Debounce Evaluation
Run:
`/Users/__blitzzz/Documents/GitHub/agy-memory-engine/.venv/bin/python /Users/__blitzzz/Documents/GitHub/agy-memory-engine/scripts/queue_cli.py status`

- If `count` is 0: queue is empty, exit immediately.
- If `can_process` is false: chat is actively in progress (<5m idle and <15m wait), exit immediately to preserve calm memory.

### Step 2: Claim Batch
Run:
`/Users/__blitzzz/Documents/GitHub/agy-memory-engine/.venv/bin/python /Users/__blitzzz/Documents/GitHub/agy-memory-engine/scripts/queue_cli.py claim --batch-size 25`

Receive the JSON payload containing `batch_id`, `lease_token`, `source`, `chat_id`, and `turns`.

### Step 3: Cognitive Extraction
Read all turns in the batch:
- If all turns are trivial chit-chat, routine commands, or temporary status inquiries without persistent value:
  Run:
  `/Users/__blitzzz/Documents/GitHub/agy-memory-engine/.venv/bin/python /Users/__blitzzz/Documents/GitHub/agy-memory-engine/scripts/queue_cli.py skip --batch-id <batch_id> --lease-token <lease_token> --summary "No persistent entities found"`
- If turns contain valuable persistent knowledge:
  Structure the JSON extraction payload according to Section 3:
  ```json
  {
    "facts": [{"id": "...", "category": "...", "fact": "...", "keywords": "..."}],
    "episodes": [{"id": "...", "topic": "...", "title": "...", "narrative": "...", "period": "...", "status": "active", "entities": "...", "stance": "...", "keywords": "..."}],
    "learnings": [{"id": "...", "category": "...", "insight": "...", "context": "...", "keywords": "..."}],
    "entity_links": [{"source": "...", "target": "...", "relation": "..."}]
  }
  ```

### Step 4: Atomic Commit & Queue Acknowledgment
Run:
`/Users/__blitzzz/Documents/GitHub/agy-memory-engine/.venv/bin/python /Users/__blitzzz/Documents/GitHub/agy-memory-engine/scripts/queue_cli.py commit --batch-id <batch_id> --lease-token <lease_token> --data '<json_payload>'`

Alternatively, if committing via native FastMCP tools (store_memory, record_learning, record_episode, link_entities_mcp), call the tools and then run:
`/Users/__blitzzz/Documents/GitHub/agy-memory-engine/.venv/bin/python /Users/__blitzzz/Documents/GitHub/agy-memory-engine/scripts/queue_cli.py ack --batch-id <batch_id> --lease-token <lease_token> --summary "<summary>"`

If an error occurs during extraction, release the batch so it can be retried:
`/Users/__blitzzz/Documents/GitHub/agy-memory-engine/.venv/bin/python /Users/__blitzzz/Documents/GitHub/agy-memory-engine/scripts/queue_cli.py release --batch-id <batch_id> --lease-token <lease_token> --error "<error_message>"`

### Step 5: Housekeeping
Prune processed turns older than 7 days:
`/Users/__blitzzz/Documents/GitHub/agy-memory-engine/.venv/bin/python /Users/__blitzzz/Documents/GitHub/agy-memory-engine/scripts/queue_cli.py prune --days 7`

## 5. Definition of Done (DOD)

A scheduled task execution is complete and successful when all of the following conditions are met:

1. Debounce Compliance: No turns are processed while an interactive session is active within the 5-minute silence window, unless the 15-minute maximum wait threshold is exceeded.
2. Partitioned Lease Isolation: Every claimed batch is identified by a unique `batch_id` and `lease_token`. No concurrent worker can double-process or overwrite the claimed turns.
3. Strict Taxonomy Validation: All committed facts, episodes, and learnings strictly match canonical categories and statuses. Invalid categories are rejected or normalized.
4. Clean Queue State: Every processed turn transitions from `status='claimed'` to `status='processed'` (or `'skipped'`), with `lease_token` cleared to `NULL` and `processed_at` timestamp recorded.
5. Vector & FTS Index Integrity: All committed facts, episodes, and learnings have synchronized FTS5 entries and 1:1 vector embeddings in SQLite tables `vec_memories`, `vec_episodes`, and `vec_learnings`.
6. Zero Orphaned Locks: The turn queue database and main memory database have zero stuck WAL frames, zero hung transactions, and zero locked rows after completion.
7. Informative Summary: The task execution outputs a concise, factual summary of turns claimed, entities extracted, and queue count remaining.
