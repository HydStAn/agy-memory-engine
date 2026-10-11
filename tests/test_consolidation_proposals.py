"""Agent-supplied consolidation: export a snapshot, apply merge proposals without an LLM call."""

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


def _no_llm(*args, **kwargs):
    raise AssertionError("proposal mode must not call an LLM")


class ConsolidationProposalTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.home = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.db = str(self.home / 'memory.db')
        self.stack.enter_context(patch.dict(os.environ, {'HOME': str(self.home)}))
        self.stack.enter_context(patch.object(schema, 'DB_PATH', self.db))
        self.stack.enter_context(patch.object(memory.tempfile, 'gettempdir', return_value=str(self.home)))
        self.stack.enter_context(patch.object(memory, 'CACHE_PATH', str(self.home / 'model.txt')))
        self.stack.enter_context(patch.object(memory, '_infer_json', side_effect=_no_llm))
        self.stack.enter_context(patch.object(memory.subprocess, 'run', side_effect=_no_llm))
        with schema.db_session():
            pass
        memory._VOCABULARY_CACHE.clear()
        self.addCleanup(schema._SCHEMA_INITIALIZED.discard, self.db)
        memory.upsert_fact('a', 'infra', 'server A runs nginx', 'nginx')
        memory.upsert_fact('b', 'infra', 'server A uses nginx', 'nginx web')
        memory.upsert_fact('solo', 'software', 'only fact in its category')

    @staticmethod
    def proposal(target='a', merged=('b',), category='infra'):
        return {'merges': [{'target_id': target, 'category': category, 'fact': 'server A runs nginx',
                            'keywords': 'nginx web', 'merged_ids': list(merged), 'rationale': 'duplicate'}]}

    def test_export_lists_only_categories_with_two_or_more_facts(self):
        snapshot = memory.export_consolidation_snapshot()
        self.assertEqual(sorted(snapshot['categories']), ['infra'])
        self.assertEqual({f['id'] for f in snapshot['categories']['infra']}, {'a', 'b'})
        self.assertEqual(set(snapshot['revisions']), {'a', 'b'})
        self.assertIn('generation', snapshot)

    def test_export_filters_by_category(self):
        self.assertEqual(memory.export_consolidation_snapshot(category='software')['categories'], {})
        self.assertEqual(sorted(memory.export_consolidation_snapshot(category='infra')['categories']), ['infra'])

    def test_apply_proposals_merges_without_llm(self):
        snapshot = memory.export_consolidation_snapshot()
        result = memory.consolidate_memories(proposals=self.proposal(), snapshot=snapshot)
        self.assertEqual([m['merged_ids'] for m in result], [['b']])
        with schema.db_session() as conn:
            self.assertEqual({r[0] for r in conn.execute('SELECT id FROM memories')}, {'a', 'solo'})
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM consolidation_log').fetchone()[0], 1)

    def test_dry_run_proposals_write_nothing(self):
        snapshot = memory.export_consolidation_snapshot()
        result = memory.consolidate_memories(dry_run=True, proposals=self.proposal(), snapshot=snapshot)
        self.assertEqual(len(result), 1)
        with schema.db_session() as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM memories').fetchone()[0], 3)
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM consolidation_log').fetchone()[0], 0)

    def test_fact_changed_after_export_is_not_merged(self):
        snapshot = memory.export_consolidation_snapshot()
        memory.upsert_fact('b', 'infra', 'server A moved to caddy', 'caddy')
        self.assertEqual(memory.consolidate_memories(proposals=self.proposal(), snapshot=snapshot), [])
        with schema.db_session() as conn:
            self.assertEqual(conn.execute("SELECT fact FROM memories WHERE id='b'").fetchone()[0], 'server A moved to caddy')

    def test_proposal_cannot_reach_facts_outside_the_snapshot(self):
        snapshot = memory.export_consolidation_snapshot(category='infra')
        self.assertEqual(memory.consolidate_memories(
            proposals=self.proposal(target='a', merged=('solo',)), snapshot=snapshot), [])
        with schema.db_session() as conn:
            self.assertIsNotNone(conn.execute("SELECT 1 FROM memories WHERE id='solo'").fetchone())

    def test_cli_export_then_apply_takes_backup_first(self):
        snap_file = self.home / 'snapshot.json'
        prop_file = self.home / 'proposals.json'
        with patch.object(sys, 'argv', ['agy_memory.py', 'consolidate', '--export-file', str(snap_file)]):
            memory.main()
        prop_file.write_text(json.dumps(self.proposal()), encoding='utf-8')
        argv = ['agy_memory.py', 'consolidate', '--apply', '--proposals-file', str(prop_file),
                '--snapshot-file', str(snap_file)]
        with patch.object(sys, 'argv', argv), \
                patch.object(memory, 'create_snapshot', wraps=memory.create_snapshot) as backup:
            memory.main()
        backup.assert_called_once_with(tag='consolidate')
        with schema.db_session() as conn:
            self.assertIsNone(conn.execute("SELECT 1 FROM memories WHERE id='b'").fetchone())

    def test_cli_proposals_require_snapshot_file(self):
        prop_file = self.home / 'proposals.json'
        prop_file.write_text(json.dumps(self.proposal()), encoding='utf-8')
        argv = ['agy_memory.py', 'consolidate', '--apply', '--proposals-file', str(prop_file)]
        with patch.object(sys, 'argv', argv), self.assertRaises(SystemExit):
            memory.main()


    def test_learnings_export_and_apply_merges_and_purges(self):
        memory.upsert_learning('l1', 'architecture', 'heuristic 1', 'ctx1', 'kw1')
        memory.upsert_learning('l2', 'architecture', 'heuristic 1 overlapping', 'ctx2', 'kw2')
        memory.upsert_learning('l3_obsolete', 'dev', 'one-time bugfix in test script', 'ctx3', 'kw3')
        memory.link_entities('l2', 'a', 'depends_on')
        memory.link_entities('l3_obsolete', 'solo', 'depends_on')

        snapshot = memory.export_consolidation_snapshot(entity_type='learnings')
        self.assertIn('architecture', snapshot['categories'])
        self.assertIn('dev', snapshot['categories'])
        self.assertIn('l1', snapshot['revisions'])
        self.assertIn('l2', snapshot['revisions'])
        self.assertIn('l3_obsolete', snapshot['revisions'])

        proposals = {
            'merges': [{
                'target_id': 'l1',
                'category': 'architecture',
                'insight': 'canonical heuristic 1',
                'context': 'ctx merged',
                'keywords': 'kw1 kw2',
                'merged_ids': ['l2'],
                'rationale': 'duplicate'
            }],
            'purges': [{
                'id': 'l3_obsolete',
                'rationale': 'obsolete bugfix'
            }]
        }

        # Dry run
        dry_res = memory.consolidate_learnings(dry_run=True, proposals=proposals, snapshot=snapshot)
        self.assertEqual(len(dry_res), 2)
        with schema.db_session() as conn:
            self.assertIsNotNone(conn.execute("SELECT 1 FROM learnings WHERE id='l2'").fetchone())
            self.assertIsNotNone(conn.execute("SELECT 1 FROM learnings WHERE id='l3_obsolete'").fetchone())

        # Apply
        res = memory.consolidate_learnings(dry_run=False, proposals=proposals, snapshot=snapshot)
        self.assertEqual(len(res), 2)
        with schema.db_session() as conn:
            self.assertIsNone(conn.execute("SELECT 1 FROM learnings WHERE id='l2'").fetchone())
            self.assertIsNone(conn.execute("SELECT 1 FROM learnings WHERE id='l3_obsolete'").fetchone())
            self.assertIsNotNone(conn.execute("SELECT 1 FROM learnings WHERE id='l1'").fetchone())
            # entity_links rewired for l2 -> l1
            self.assertIsNotNone(conn.execute("SELECT 1 FROM entity_links WHERE source_id='l1' AND target_id='a'").fetchone())
            # entity_links purged for l3_obsolete
            self.assertIsNone(conn.execute("SELECT 1 FROM entity_links WHERE source_id='l3_obsolete'").fetchone())
            # consolidation log recorded
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM consolidation_log WHERE action='merge'").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM consolidation_log WHERE action='purge'").fetchone()[0], 1)


if __name__ == '__main__':
    unittest.main()
