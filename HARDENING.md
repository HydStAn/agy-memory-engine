# Memory engine hardening

This branch changes repository behavior. It does not install host hooks, start a scheduler, configure an inference provider, migrate the production store, or replay the existing backlog.

## Runtime configuration

Automatic extraction and semantic consolidation use `memory_inference.py`, a bounded subprocess calling a trusted OpenAI-compatible chat-completions endpoint. Set `AGY_MEMORY_INFERENCE_URL` to its full HTTPS endpoint, `AGY_MEMORY_INFERENCE_MODEL` to an available provider model, and `AGY_MEMORY_INFERENCE_KEY` if authentication is required. Loopback HTTP is supported for local servers. No tools are offered or executed, and tool-call responses are rejected. There is deliberately no unrestricted agent fallback. Without endpoint configuration, extraction fails and turns remain pending with diagnostic errors.

Database schema version is 211, independent of package version. Bootstrap and migration are serialized; FTS mirrors use base-table rowids. Restart all clients together before deploying this branch. Current engine connections take a shared maintenance lock; restore takes its exclusive counterpart and waits for active clients. Older binaries and arbitrary SQLite clients do not honor that advisory lock and must be stopped before restore. SQLite's backup API supplies database-level consistency, including committed WAL frames. Every restore creates a verified safety backup first; corrupt targets require offline recovery.

Snapshots for the default database remain in `~/.gemini/archive`. Other database paths receive separate archive namespaces. `AGY_MEMORY_ARCHIVE`, `AGY_MEMORY_CACHE`, and `AGY_MEMORY_SYNC_LOCK` provide explicit overrides. Existing custom-database snapshots in the old shared archive need deliberate relocation to their correct namespace.

Generate reviewable service configuration with:

```sh
.venv/bin/python scripts/render_memory_service.py --output-dir /absolute/service-output
```

The result contains a separate Stop hook fragment and a macOS LaunchAgent running the worker every 60 seconds with notifications disabled. Merge the fragment into the host hook configuration without replacing other hooks. Install exactly one scheduler after configuring and validating inference. The repository task does not install these artifacts. Before a live rollout, use a disposable database, queue and transcript to validate one event, one claim, one durable receipt and one acknowledgement. Reconcile historical pending and processed records separately; the old processed status is not evidence of a receipt.

The dashboard defaults to `127.0.0.1`. Mutations require the local bearer token and trusted origin/host. Keep its token file private. CLI optimization previews by default and semantic consolidation is explicitly opt-in. MCP maintenance runs in one bounded worker, leaving the event loop responsive. Cancelling an MCP request detaches the caller and does not imply rollback; the maintenance slot stays occupied until the actual worker finishes. Dashboard subprocess errors and timeouts are failures, never unconditional success; a timeout can follow already committed maintenance stages, so inspect state before retrying.

## Contracts and coverage

| Audit finding | Repository remediation and principal evidence |
| --- | --- |
| F01 | Online WAL snapshot/restore, integrity/schema checks, safety backup, bounded busy retries, coordinated current clients. `test_core_hardening`, `test_remaining_hardening`. |
| F02 | Typed errors, pending retry, durable receipt replay and lease acknowledgement. `test_queue_receipts`, commit/ack crash regression. |
| F03 | Whole extraction transaction, revision conflict checks including tombstones, preserved omitted fields. `test_remaining_hardening`. |
| F04 | Collision-safe source/chat/event identity, partitioned deterministic batches and stable recovered membership. Queue suites. |
| F05 | Same-category consolidation, protected and stale-target rejection, graph rewrites and audit preimages. Core suite. |
| F06 | Stored fact, learning and episode category protection for automatic updates. Core suite. |
| F07 | Library diagnostics on stderr and real FastMCP stdio test. No process-global stdout redirection in background threads. |
| F08 | Loopback default, authenticated mutations, origin/host checks, escaped metadata and bounded payloads. Dashboard suites. |
| F09 | Read-only optimization preview with no initialization, snapshots, inference, queue pruning or maintenance writes. Core suite. |
| F10 | Tool-free inference subprocess and internal invocation flags; no AGY agent launch. Inference regression. |
| F11 | Separate Stop-hook/service artifacts supplied. Host installation and live canary remain deployment work. |
| F12 | Explicit or multi-root transcript resolution, latest USER_INPUT boundary and stable event index. Hook suite. External shared-skill search helpers are outside this repository. |
| F13 | Shared taxonomy, strict new writes, preservation of unmapped legacy categories/statuses, architecture/workflow categories retained. Core, migration and MCP suites. Untracked external import scripts were preserved, not adopted. |
| F14 | Shared relation mapping for link/unlink/MCP/migration, correct prescription inverse, ambiguous legacy relations retained. New core/MCP graph endpoints require unique content-backed identities. |
| F15 | Transactional rowid FTS triggers, base-table repair and PK/update/delete transitions. Core and remaining suites. |
| F16 | Serialized queue migration and no DDL in steady-state claims/enqueue. Queue and remaining suites. |
| F17 | Deterministic connection cleanup, database identity/version checks and classified busy errors. Core and queue suites. |
| F18 | Vocabulary cache with 60-second lifetime and local write invalidation. Core suite. |
| F19 | Short technical tokens, linked learnings, bounded search limits and serialized context byte budget. Retrieval regressions. |
| F20 | MCP tool errors, bounded search envelope, offloaded maintenance, actual acknowledged worker counts, child return-code checks and separate notification errors. Surface and queue suites. |
| F21 | Neutral owner prompts, consistent explicit consolidation, configured model/cache, database-specific locks/archives and loopback example config. |
| F22 | Temporary test environment, unswallowed assertions, protocol tests, WAL recovery, claims and commit/ack crash checks. Hermetic actual-engine benchmark reports failures in its denominator and exits nonzero on failure. |

The legacy `test_authoritative_verification.py` invokes 22 existing tests again. Its test count is not 22 independent demonstrations and its original finding names do not cover every audit requirement. The table above and actual regression assertions define the evidence. Test success does not establish live service wiring, production data parity, provider availability, browser rendering, power-loss recovery, or a production soak test.

Run:

```sh
.venv/bin/python -m unittest discover -s tests
.venv/bin/python tests/benchmark_load_test.py
```
