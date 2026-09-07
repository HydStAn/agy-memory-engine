# AGY Memory Engine (v2.2.0)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 49/49 Passing](https://img.shields.io/badge/Tests-49%2F49%20Passed-brightgreen.svg)]()

> Lightweight, high-performance, standalone dynamic cognitive memory layer for Google Antigravity (`agy`) and autonomous agent frameworks.

Inspired by Hermes Agent's multi-pillar memory architecture, using SQLite FTS5 for ultra-fast local retrieval (<2ms), multilingual compound sub-token decomposition & morphological stemming (DE, EN, FR, IT, ES, NL, SV), relational entity linking, and autonomous background queue workers with calm-memory session debouncing.

---

## 📑 Table of Contents

- [The Big Picture: Autonomous Omni-Channel Stack](#-the-big-picture-autonomous-omni-channel-stack)
- [The 4-Layer Cognitive Memory Model](#️-the-4-layer-cognitive-memory-model)
- [Hybrid Multilingual Tokenizer vs. Vector Databases](#-hybrid-multilingual-tokenizer-vs-vector-databases)
- [5-Minute Quickstart Guide for Newbies](#-5-minute-quickstart-guide-for-newbies)
- [CLI Reference & Quick Commands](#-cli-reference--quick-commands)
- [Model Context Protocol (MCP) Server](#-model-context-protocol-mcp-server)
- [Seamless Integration with Antigravity Telegram Bot](#-seamless-integration-with-antigravity-telegram-bot)
- [Configuration (`.env`)](#️-configuration-env)
- [Real-Time Debug Web Dashboard](#-real-time-debug-web-dashboard)
- [Autonomous Background Pipeline (Cron & Lifecycle Hooks)](#-autonomous-background-pipeline-cron--lifecycle-hooks)
- [Testing](#-testing)
- [Release Notes](#-release-notes)
- [Roadmap](#️-roadmap)
- [License](#-license)

---

## 🧩 The Big Picture: Autonomous Omni-Channel Stack

`agy-memory-engine` acts as the persistent semantic backbone across all client interfaces (Telegram, Terminal CLI, Web Cockpit, IDE):

```text
                  ┌────────────────────────────────────────────────────────┐
                  │    Omni-Channel Interfaces (Telegram, CLI, Web, IDE)   │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                                              ▼
                    ┌────────────────────────────────────────────────────┐
                    │       Global AGY Stop-Hook (hooks.json)            │
                    │   • Enqueues turn in < 1ms to turn_queue.db        │
                    │   • Zero latency impact on active conversations    │
                    └─────────────────────────┬──────────────────────────┘
                                              │
                                              ▼ (5m Idle OR 15m Timeout)
                    ┌────────────────────────────────────────────────────┐
                    │      Calm Memory Worker (memory_worker.py)         │
                    │   • Batches full conversation into 1 LLM pass      │
                    │   • Single consolidated Telegram status update     │
                    │   • Loop prevention (AGY_INTERNAL_INVOCATION=1)    │
                    └─────────────────────────┬──────────────────────────┘
                                              │
                                              ▼
                    ┌────────────────────────────────────────────────────┐
                    │     4-Layer Cognitive Memory (~/.gemini/memory.db) │
                    │   • Layer 1: Atomic Facts (memories)               │
                    │   • Layer 2: Narrative Episodes (episodes)         │
                    │   • Layer 3: Experiential Learnings (learnings)    │
                    │   • Layer 4: Relational Entity Graph (links)       │
                    └────────────────────────────────────────────────────┘
```

---

## 🏛️ The 4-Layer Cognitive Memory Model

```text
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    ~/.gemini/memory.db                                      │
├───────────────────┬────────────────────────────┬────────────────────────────┬───────────────┤
│ Layer 1: Facts    │ Layer 2: Narrative Dossiers│ Layer 3: Learnings         │ Layer 4: Graph│
│   (`memories`)    │        (`episodes`)        │       (`learnings`)        │(`entity_links`)
├───────────────────┼────────────────────────────┼────────────────────────────┼───────────────┤
│ - Server IPs/ports│ - Background histories     │ - Rules of thumb, lessons  │ - Directional │
│ - Hardware specs  │ - Stances & sentiment      │ - Tested heuristics        │   relations   │
│ - Master data     │ - Lifecycle status decay   │ - Contextual guidelines    │   (hosted_on, │
│ - Exact BM25 match│   (active->cooling->past)  │ - Decision rationale       │    owns, etc.)│
└───────────────────┴────────────────────────────┴────────────────────────────┴───────────────┘
```

---

## 🔍 Evolution: Vom reinen Tokenizer zur In-Process Hybrid Search

In Version 2.0/2.1 verzichtete `agy-memory-engine` bewusst auf externe, ressourcenhungrige Vektor-Datenbanken (wie Chroma, Milvus oder Qdrant) und PyTorch-Bloat (~500MB RAM, 150ms Latenz), um schlank zu bleiben.

**Die Limitation der reinen Keyword-/FTS5-Suche:**
Reines FTS5 scheitert am sogenannten *Vocabulary Mismatch*: Fragt der Nutzer in freier Alltagssprache nach *"Verhinderung von Datenverlust bei Aufräumarbeiten"*, während in der Datenbank die Heuristik *"Löschungen jeglicher Art erfordern explizite Genehmigung (Löschschutz)"* hinterlegt ist, findet eine reine FTS5-Suche exakt **0 Treffer**.

**Die Lösung ab v2.2.0: In-Process Hybrid Search (sqlite-vec + FastEmbed)**
Statt ein schweres Backend einzuführen, kombiniert v2.2.0 das Beste aus beiden Welten:
1. **Ultra-schneller CLI-Prefetch (< 2ms):** Der synchrone Pre-Invocation Hook bleibt unverändert bei reinem FTS5 + Trigram-Zerlegung für deutsche Komposita – 0 ms spürbare Latenz im Chat.
2. **In-Process Vector Extension (`sqlite-vec`):** Läuft nativ in C direkt im SQLite-Prozess (SIMD-beschleunigt, kein Daemon, kein Netzwerk-Hop).
3. **Schlankes lokales Embedding (`fastembed`):** Nutzt ONNX Runtime mit `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 Dimensionen, ~9ms Inferenzzeit, hervorragende DE/EN-Semantik).
4. **Reciprocal Rank Fusion (RRF):** Führt BM25-Ergebnisse (100% Präzision bei IPs, IDs, exakten Namen) und Vektordistanz (Verständnis vager Konzepte und Synonyme) mathematisch optimal zusammen.
5. **Automatische Synchronisation:** Hintergrund-Worker (`memory_worker.py`) und MCP-Store-Tools indizieren Vektoren transparent im Hintergrund; SQLite-Löschtrigger kaskadieren automatisch.

---

## ⚡ 5-Minute Quickstart Guide for Newbies

Get up and running from zero to autonomous memory in 5 minutes.

### 1. Prerequisites & Installation

Clone the repository and install the lightweight Python dependencies (no PyTorch, no heavyweight vector DBs needed):

```bash
git clone https://github.com/sbolten/agy-memory-engine.git
cd agy-memory-engine

# Optional: set up your environment configuration
cp .env.example .env
```

The database (`~/.gemini/memory.db`) will be automatically initialized with all FTS5 virtual tables on first run!

---

### 2. See Instant Results in the Web Dashboard

The easiest way to see what's happening and test queries is the built-in live dashboard:

```bash
# Start the web UI on port 8085
python3 agy_memory.py ui --port 8085
```
Open **`http://localhost:8085`** in your browser. You can:
* Test hybrid multilingual searches in real-time with sub-millisecond metrics.
* Visually browse Facts, Episodes, Learnings, and Graph Links.
* Monitor pending debounced background tasks.

---

### 3. Add Your First Memories via CLI

You can seed your agent's memory directly from the terminal:

```bash
# 1. Add an atomic fact (Layer 1)
python3 agy_memory.py add --id "infra.server.ip" --category "infra" --fact "Home server IP is 192.168.1.100" --keywords "home server ip host"

# 2. Add a personal learning / heuristic (Layer 3)
python3 agy_memory.py add-learning --id "workflow.email.style" --category "communication" --insight "Keep email replies strictly under 3 bullet points." --keywords "email communication reply rule"

# 3. Test sub-millisecond prefetch (< 2ms)
python3 agy_memory.py prefetch "server"
```

---

### 4. Hook it up to your Agent (AGY / Claude Code)

#### Option A: Native MCP Integration (Works with Claude Code, Cursor, AGY)
Add the memory engine as an MCP tool server so your agent can actively search and store knowledge:

```bash
# For Claude Code:
claude mcp add memory python3 $(pwd)/agy_memory_mcp.py

# Or in your MCP config JSON (~/.gemini/antigravity-cli/mcp_config.json or Claude Desktop):
{
  "mcpServers": {
    "memory": {
      "command": "python3",
      "args": ["/path/to/agy-memory-engine/agy_memory_mcp.py"]
    }
  }
}
```

#### Option B: Configure Agent Prompt / Instructions (`GEMINI.md` or `CLAUDE.md`)
Add a simple memory rule to your global agent instructions (e.g. `~/.gemini/config/GEMINI.md`, `CLAUDE.md`, or your project's system prompt) so your agent knows when to query memory:

```markdown
## Long-Term Memory
- Before answering questions regarding personal preferences, server IPs, hardware, or past projects, call `search_memory` or run `agy_memory.py prefetch "<topic>"`.
- When the user states a new permanent fact or personal rule, persist it using `store_memory` or `record_learning`.
```

#### Option C: Autonomous Background Sync (Zero-friction)
To have the memory engine automatically learn from your conversations without you lifting a finger:

1. Register the sub-millisecond turn hook in `~/.gemini/config/hooks.json` (see [Autonomous Background Pipeline](#-autonomous-background-pipeline-cron--lifecycle-hooks)).
2. Add the debounced background worker to your crontab (`*/5 * * * * python3 /path/to/agy-memory-engine/memory_worker.py`).

---

## 🚀 CLI Reference & Quick Commands

```bash
# Multi-Layer Prefetch (< 2ms)
python3 agy_memory.py prefetch "Hundeversicherung"

# Add Layer 1 Fact
python3 agy_memory.py add --id "infra.beelink.ip" --category "infra" --fact "Beelink Host IP is 100.114.118.47" --keywords "beelink host server ip"

# Add Layer 2 Episode
python3 agy_memory.py add-episode --id "travel.iceland2027" --topic "travel" --title "Laugavegur Trekking" --narrative "Hut booking watchdog active for July 2027." --status "active" --keywords "island laugavegur"

# Add Layer 3 Learning
python3 agy_memory.py add-learning --id "travel.flights.cdp" --category "travel" --insight "Use CDP browser for Google Flights to avoid bot-blocking." --keywords "google flights bot cdp"

# Link Entities in Graph
python3 agy_memory.py link --source "service.immich" --target "infra.beelink.ip" --relation "hosted_on"

# Optimize & Decay Maintenance
python3 agy_memory.py optimize --apply

# Migrate Database from v2.0 to v2.1 (Taxonomies, Relations, Graph Pruning)
python3 agy_memory.py migrate --dry-run
python3 agy_memory.py migrate
```

## 🔌 Model Context Protocol (MCP) Server

The engine includes a native FastMCP server (`agy_memory_mcp.py`) that equips autonomous AI agents (Antigravity, Claude, Cursor, OpenCode) with explicit memory reading and writing capabilities:

### Available MCP Tools

| Tool | Description | Key Parameters |
| :--- | :--- | :--- |
| `search_memory` | Hybrid multilingual search across Facts, Episodes, Learnings & Graph relations | `query` *(str)*, `limit` *(int, default: 5)* |
| `store_memory` | Store or update an atomic configuration parameter or fact (Layer 1) | `id`, `fact`, `category`, `keywords` |
| `record_episode` | Record a rich narrative chronicle, ongoing topic, or history (Layer 2) | `id`, `topic`, `title`, `narrative`, `status`, `stance` |
| `record_learning` | Record a practical heuristic, tested rule of thumb, or stance (Layer 3) | `id`, `category`, `insight`, `context`, `keywords` |
| `link_entities_mcp` | Create directional knowledge graph links between memory entities (Layer 4) | `source_id`, `target_id`, `relation` |
| `list_memories` | Full multi-layer inventory export of all stored memories | — |
| `migrate_memory` | Migrate database to canonical v2.1 taxonomies, map relations, prune orphans | `dry_run` *(bool, default: True)* |
| `optimize_memory` | Rebuild FTS indexes, execute episode aging decay, prune links, and VACUUM | `apply_changes` *(bool, default: True)*, `consolidate` *(bool)* |

### MCP Configuration

Add to your MCP settings file (e.g. `~/.gemini/antigravity-cli/mcp_config.json` or Claude/Cursor config):

```json
{
  "mcpServers": {
    "memory": {
      "command": "python3",
      "args": ["/opt/agy-memory-engine/agy_memory_mcp.py"],
      "env": {
        "AGY_MEMORY_DB": "~/.gemini/memory.db"
      }
    }
  }
}
```

---

## 📱 Seamless Integration with Antigravity Telegram Bot
 
 `agy-memory-engine` is designed to work in synergy with the [Antigravity Telegram Bot (`antigravity-cli-telegram-bot`)](https://github.com/ardiannurcahya/antigravity-cli-telegram-bot) to form a completely autonomous, mobile memory pipeline:

```text
  📱 Mobile User in Telegram (Voice, Text, Photos, Topics)
            │
            ▼
  🤖 AGY Telegram Bot (/opt/agy-telegram-bot)
            │  (Executes standard agy prompt with --add-dir)
            ▼
  ⚡ AGY Global Stop-Hook (~/.gemini/config/hooks.json -> scripts/auto_sync_hook.py)
            │  (Enqueues turn in <1ms to turn_queue.db, resolves Telegram topic/chat ID)
            ▼
  🧠 Calm Memory Worker (memory_worker.py)
            │  (Debounces 5m idle / 15m timeout, batches conversation into 1 LLM pass)
            ▼
  💾 SQLite FTS5 Memory Store (~/.gemini/memory.db)
            │
            ▼
   📲 Instant Status Notification back to Telegram Topic / Chat
      "🧠 Autonomous memory updated (1 fact, 1 learning)
       • ➕ Beelink Host IP is 100.114.118.47
       • ➕ Use CDP browser for Google Flights"
```

### Key Synergy Highlights:
1. **Zero Chat Latency:** The global stop-hook returns in `< 1ms`, ensuring the Telegram Bot responds instantly without waiting for memory extraction.
2. **Context-Aware Topic Routing:** The worker automatically preserves the originating Telegram `chat_id` and `message_thread_id`, routing notifications directly back into the relevant topic thread.
3. **Loop Prevention:** Ingestion runs under `AGY_INTERNAL_INVOCATION=1` with prompt marker guards to prevent recursive agent loops.

---

## ⚙️ Configuration (`.env`)

All engine parameters, database locations, LLM model choice, and debounce thresholds can be configured via `.env` (or environment variables). A ready-to-use template is provided in [`.env.example`](file:///opt/agy-memory-engine/.env.example):

```bash
# Copy template to .env
cp .env.example .env
```

```ini
# ==============================================================================
# AGY Memory Engine - Configuration File
# ==============================================================================

# LLM model used for background memory extraction & consolidation
AGY_MEMORY_MODEL=gemini-3.7-flash-low

# SQLite Database Storage Paths
AGY_MEMORY_DB=~/.gemini/memory.db
AGY_TURN_QUEUE_DB=~/.gemini/turn_queue.db

# Calm-Memory Debounce Settings (in seconds)
AGY_MEMORY_INACTIVITY_SECONDS=300   # 5 minutes idle threshold
AGY_MEMORY_MAX_WAIT_SECONDS=900     # 15 minutes max timeout

# Telegram Notification Recipient (optional: your numeric Telegram user or group ID)
AGY_MEMORY_TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# Path to Antigravity CLI binary
AGY_BIN=agy

# Real-Time Debug Dashboard (Web UI)
AGY_MEMORY_DEBUG_DASHBOARD=true
AGY_MEMORY_DASHBOARD_PORT=8085
AGY_MEMORY_DASHBOARD_HOST=0.0.0.0
```

---

## 📊 Real-Time Debug Web Dashboard

A zero-dependency, standalone live web dashboard is included to inspect, search, and monitor memory state in real time:

* **Live FTS5 Search Sandbox:** Test hybrid multilingual queries with sub-millisecond latency metrics.
* **Turn Queue & Debounce Monitor:** Visual countdown bar for active conversation debouncing (5m idle / 15m timeout) with an instant *"Batch jetzt verarbeiten"* trigger.
* **4-Layer Visualizer:** Browse Facts (Layer 1), Thematic Episodes with status badges (Layer 2), Experiential Learnings (Layer 3), and Knowledge Graph Entity Links (Layer 4).
* **Consolidation Audit Log:** Review automated background merges, deduplications, and semantic rationale.

### Starting the Dashboard

```bash
# Via CLI command
python3 agy_memory.py ui --port 8085

# Or directly via standalone runner
python3 dashboard.py --port 8085

# Or via systemd background user service
systemctl --user start agy-memory-dashboard.service
```

Access in your browser at `http://localhost:8085` (or over Tailscale at `http://<tailscale-ip>:8085`).

---

## ⏰ Autonomous Background Pipeline (Cron & Lifecycle Hooks)

To enable 100% autonomous background learning without manual intervention, configure the **AGY Lifecycle Hook** and the **Linux Crontab**:

### 1. Global Lifecycle Hook (`~/.gemini/config/hooks.json`)

Registers the sub-millisecond transcript collector on every agent turn stop:

```json
{
  "memory-auto-sync": {
    "enabled": true,
    "Stop": [
      {
        "type": "command",
        "command": "python3 /opt/agy-memory-engine/scripts/auto_sync_hook.py",
        "timeout": 15
      }
    ]
  }
}
```

### 2. Crontab Configuration (`crontab -e`)

```bash
# Process pending memory queue every 5 minutes (debounced)
*/5 * * * * python3 /opt/agy-memory-engine/memory_worker.py >/dev/null 2>&1

# Nightly memory decay aging, duplicate consolidation & VACUUM (04:30)
30 4 * * * python3 /opt/agy-memory-engine/agy_memory.py optimize --apply >/dev/null 2>&1
```

---

## 🧪 Testing

```bash
python3 -m unittest discover tests/ -v
# Ran 48 tests in 2.9s (OK)
```

---

## 🚀 Release Notes

### v2.2.0 (2026-09-07)
- **Semantic Recall & In-Process Hybrid Search (FTS5 + `sqlite-vec`)**:
  - In-process vector extension via `sqlite-vec` (C-extension, SIMD-accelerated, zero external daemon).
  - Dense 384-dimensional multilingual embeddings via `fastembed` with `paraphrase-multilingual-MiniLM-L12-v2`.
  - Reciprocal Rank Fusion (RRF) combining BM25 lexical precision with semantic cosine similarity in `search_memory`.
  - Zero latency impact on CLI prefetch: Pre-invocation prefetch remains strictly < 2ms (FTS5 + Trigram).
  - Background vector indexing of 175 facts, 37 episodes, and 92 learnings in `~/.gemini/memory.db`.
  - Automatic cascade deletion triggers from SQLite parent tables to virtual vector tables (`vec_memories`, `vec_episodes`, `vec_learnings`).
  - Standalone reindexing utility `scripts/reindex_vectors.py`.

### v2.1.0 (2026-09-03)
- **Quality-First Extraction & Consolidation Pipeline**:
  - Strict litmust test and exclusion rules for experiential learnings (no transient bug fixes, UI tweaks or code-internal details; strictly reusable heuristics and behavioral insights).
  - Closed canonical relationship taxonomy (26 canonical link types) preventing relationship fragmentation.
  - Closed canonical categories across all layers (Facts, Learnings, Episodes) with automated runtime alias normalization (`_normalize_category`).
  - Automated orphan link pruning (`prune_orphan_links`) removing severed graph relationships when entities are deleted or consolidated.
  - Category normalization enforcement during semantic LLM fact consolidation.
- **Unified Maintenance & Web Dashboard**:
  - Full end-to-end integration of batch normalization and graph pruning into `optimize_db` (`compact`) CLI command and Web Dashboard `/api/optimize` endpoint.
- **Automated Database Migration Tool (`v2.0 -> v2.1`)**:
  - Dedicated CLI migration command `agy_memory.py migrate` and standalone runner `scripts/migrate_v2_to_v2_1.py`.
  - Automatic safety snapshot backups in `~/.gemini/archive/` before applying modifications.
  - Semantic relation remapping and directional inversion (e.g. `hosts` -> `hosted_on`, `monitored_by` -> `monitors`).
  - Strict episode status normalization (`monitoring` -> `active`), topic mapping, and orphan link pruning.
  - FTS5 virtual table rebuilds and database vacuuming with `PRAGMA user_version = 210`.

---

## 🗺️ Roadmap

- [x] **Semantic Recall & Hybrid Search (FTS5 + `sqlite-vec`)**:
  - In-process vector extension via `sqlite-vec` alongside FTS5.
  - Multilingual embedding model (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) via `fastembed` for vague natural language queries.
  - Reciprocal Rank Fusion (RRF) to combine BM25 keyword precision with semantic vector similarity for on-demand queries (`search_memory`).
  - Keep pre-invocation prefetch strictly < 2ms (FTS5 + Trigram).
- [x] **Hermes-Style 3-Tier Memory Architecture**:
  - **Tier 1 (Profile/Preferences)**: Lean, fixed identity facts injected directly into agent system prompt (0ms latency, ~150-250 tokens max).
  - **Tier 2 (Episodic & Semantic Store)**: 4-Layer SQLite `memory.db` with Hybrid Search on-demand.
  - **Tier 3 (Working Memory)**: Active session context & scratchpad.
- [ ] **Pluggable LLM Backends for Background Worker**:
  - Abstract extraction & consolidation caller (`call_llm`) beyond `agy --print`.
  - **Claude Code Compatibility**: Support Claude CLI (`claude -p`) as background extraction engine.
  - **Direct Provider APIs**: Native lightweight connectors for Anthropic (e.g. Claude 3.5 Haiku) and OpenAI-compatible endpoints without external CLI dependencies.
- [ ] **Multi-Agent Session Routing**: Dynamic extraction profile tagging per client/agent session.
- [ ] **Selective Synced Subgraphs**: Export and sync filtered memory subsets across distributed nodes.

---

## 📄 License

MIT License © 2026 HydStAn
