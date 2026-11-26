"""Tests for anti-blur / lossless memory guardrails:
1. Read-before-write existing context injection in sync_turn
2. Deterministic anti-wipe protection on destructive fact/episode updates
3. Lossless merge guardrail rejecting detail-dropping consolidations
4. Extended protected categories (family, contacts, *.profile)
5. Learning purge brake during automated optimize_db
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_environment  # noqa: F401

import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import agy_memory as memory
import schema


class LosslessGuardrailsTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.home = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.db = str(self.home / "memory.db")
        self.stack.enter_context(patch.dict(os.environ, {"HOME": str(self.home)}))
        self.stack.enter_context(patch.object(schema, "DB_PATH", self.db))
        self.stack.enter_context(patch.object(memory.tempfile, "gettempdir", return_value=str(self.home)))
        self.stack.enter_context(patch.object(memory, "CACHE_PATH", str(self.home / "model.txt")))
        with schema.db_session():
            pass
        memory._VOCABULARY_CACHE.clear()
        self.addCleanup(schema._SCHEMA_INITIALIZED.discard, self.db)

    def test_protected_categories_include_family_contacts_and_profile_keys(self):
        self.assertTrue(memory._is_protected_key("family.member.data"))
        self.assertTrue(memory._is_protected_key("contacts.vendor.primary"))
        self.assertTrue(memory._is_protected_key("custom.entity.profile"))
        self.assertTrue(memory._is_protected_key("opaque.key", {"category": "family"}))
        self.assertTrue(memory._is_protected_key("opaque.key", {"category": "contacts"}))
        self.assertFalse(memory._is_protected_key("infra.service.web", {"category": "infra"}))

    def test_lossless_merge_guard_rejects_detail_dropping_consolidation(self):
        memory.upsert_fact(
            "infra.ssh.timeout",
            "infra",
            "Global SSH config uses ConnectTimeout 3 and ConnectionAttempts 1 in /etc/ssh/config.",
            "ssh timeout",
        )
        memory.upsert_fact(
            "infra.ssh.kex",
            "infra",
            "SSH hosts node-a (10.0.0.10) and node-b (10.0.0.20) require curve25519-sha256.",
            "ssh kex",
        )
        snapshot = memory.export_consolidation_snapshot(category="infra")

        lossy_proposal = {
            "merges": [{
                "target_id": "infra.ssh.config",
                "category": "infra",
                "fact": "SSH configuration is managed centrally for all hosts.",
                "keywords": "ssh config",
                "merged_ids": ["infra.ssh.timeout", "infra.ssh.kex"],
                "rationale": "Combined SSH settings",
            }]
        }
        rejected = memory.consolidate_memories(dry_run=False, proposals=lossy_proposal, snapshot=snapshot)
        self.assertEqual(rejected, [])
        with schema.db_session() as conn:
            ids = {r[0] for r in conn.execute("SELECT id FROM memories")}
            self.assertEqual(ids, {"infra.ssh.timeout", "infra.ssh.kex"})

        lossless_proposal = {
            "merges": [{
                "target_id": "infra.ssh.config",
                "category": "infra",
                "fact": (
                    "Global SSH config (/etc/ssh/config) sets ConnectTimeout 3 and ConnectionAttempts 1. "
                    "SSH hosts node-a (10.0.0.10) and node-b (10.0.0.20) require curve25519-sha256."
                ),
                "keywords": "ssh timeout kex",
                "merged_ids": ["infra.ssh.timeout", "infra.ssh.kex"],
                "rationale": "Lossless merge of SSH settings",
            }]
        }
        accepted = memory.consolidate_memories(dry_run=False, proposals=lossless_proposal, snapshot=snapshot)
        self.assertEqual(len(accepted), 1)
        with schema.db_session() as conn:
            row = conn.execute("SELECT fact FROM memories WHERE id='infra.ssh.config'").fetchone()
            self.assertIsNotNone(row)
            self.assertIn("10.0.0.10", row[0])
            self.assertIn("10.0.0.20", row[0])

    def test_sync_turn_injects_existing_content_and_guards_against_wipe(self):
        original_fact = (
            "Primary build server node-alpha runs at 10.10.1.50 via /srv/docker/builder/docker-compose.yml "
            "on port 8080 (https://builder.example.com/api) with 32 GB RAM and NVMe storage."
        )
        memory.upsert_fact("infra.service.builder", "infra", original_fact, "builder docker server")

        captured_prompts = []

        def fake_infer(prompt, timeout, model=None):
            captured_prompts.append(prompt)
            return json.dumps({
                "facts": [{
                    "id": "infra.service.builder",
                    "category": "infra",
                    "fact": "Build server runs a post-build cleanup script every night.",
                    "keywords": "builder cleanup",
                }],
                "episodes": [],
                "learnings": [],
                "entity_links": [],
            })

        with patch.object(memory, "_infer_json", side_effect=fake_infer):
            memory.sync_turn(
                "Configure the builder server post-build cleanup script.",
                "Configured the cleanup script on the builder server.",
            )

        self.assertEqual(len(captured_prompts), 1)
        self.assertIn("Existing Relevant Entries", captured_prompts[0])
        self.assertIn("10.10.1.50", captured_prompts[0])

        with schema.db_session() as conn:
            updated = conn.execute("SELECT fact FROM memories WHERE id='infra.service.builder'").fetchone()[0]
        self.assertIn("10.10.1.50", updated)
        self.assertIn("/srv/docker/builder/docker-compose.yml", updated)
        self.assertIn("post-build cleanup script", updated)

    def test_sync_turn_episode_anti_wipe_preserves_prior_arc(self):
        original_arc = (
            "Initial server outage on 2026-09-10 caused by 8 GB swap exhaustion and kernel OOM killer on "
            "node-beta (10.20.0.5), requiring manual power cycle and swap expansion to 16 GB."
        )
        memory.upsert_episode(
            "infra.incident.node_beta",
            "infra",
            "Node Beta Incidents",
            original_arc,
            status="active",
            keywords="node-beta outage swap",
        )

        def fake_infer(prompt, timeout, model=None):
            return json.dumps({
                "facts": [],
                "episodes": [{
                    "id": "infra.incident.node_beta",
                    "topic": "infra",
                    "title": "Node Beta Incidents",
                    "status": "active",
                    "narrative": "In November 2026 a display scaling bug in the web console was resolved by pinning 1920x1080.",
                    "keywords": "node-beta console scaling",
                }],
                "learnings": [],
                "entity_links": [],
            })

        with patch.object(memory, "_infer_json", side_effect=fake_infer):
            memory.sync_turn("We fixed the node-beta display scaling issue.", "Pinned 1920x1080.")

        with schema.db_session() as conn:
            nar = conn.execute("SELECT narrative FROM episodes WHERE id='infra.incident.node_beta'").fetchone()[0]
        self.assertIn("10.20.0.5", nar)
        self.assertIn("2026-09-10", nar)
        self.assertIn("1920x1080", nar)

    def test_consolidate_learnings_allow_purges_false_blocks_deletions(self):
        memory.upsert_learning("workflow.rule.keep_me", "workflow", "Always use explicit staging deployments.", "ops", "deploy")
        snapshot = memory.export_consolidation_snapshot(entity_type="learnings")
        proposals = {
            "merges": [],
            "purges": [{"id": "workflow.rule.keep_me", "rationale": "Overly specific"}],
        }
        res = memory.consolidate_learnings(dry_run=False, proposals=proposals, snapshot=snapshot, allow_purges=False)
        self.assertEqual(res, [])
        with schema.db_session() as conn:
            self.assertIsNotNone(conn.execute("SELECT 1 FROM learnings WHERE id='workflow.rule.keep_me'").fetchone())


if __name__ == "__main__":
    unittest.main()
