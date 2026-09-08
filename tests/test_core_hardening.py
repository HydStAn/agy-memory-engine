"""Regression evidence for storage, extraction, consolidation and retrieval defects."""

# Set temporary paths before importing modules that capture configuration defaults.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_environment  # noqa: F401

import io
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from unittest.mock import patch, Mock

import agy_memory as memory
import schema


class CoreHardeningTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.home = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.db = str(self.home / 'memory.db')
        self.stack.enter_context(patch.dict(os.environ, {'HOME': str(self.home)}))
        self.stack.enter_context(patch.object(schema, 'DB_PATH', self.db))
        self.stack.enter_context(patch.object(memory.tempfile, 'gettempdir', return_value=str(self.home)))
        self.stack.enter_context(patch.object(memory, 'CACHE_PATH', str(self.home / 'model.txt')))
        self.stack.enter_context(patch('queue_manager.prune_processed_turns'))
        with schema.db_session() as conn:
            pass
        memory._VOCABULARY_CACHE.clear()
        self.addCleanup(schema._SCHEMA_INITIALIZED.discard, self.db)

    def extract(self, data):
        response = Mock(stdout=json.dumps(data), returncode=0)
        with patch.object(memory.subprocess, 'run', return_value=response) as run:
            result = memory.sync_turn('Remember the deployment database and its operational configuration.', 'Recorded.')
        return result, run

    def test_wal_snapshot_restore_with_open_connection(self):
        with sqlite3.connect(self.db) as live:
            live.execute('PRAGMA wal_autocheckpoint=0')
            live.execute("INSERT INTO memories(id, category, fact) VALUES ('wal', 'infra', 'committed in WAL')")
            live.commit()
            self.assertGreater(Path(self.db + '-wal').stat().st_size, 0)
            snap = memory.create_snapshot()
            with sqlite3.connect(memory.archive_path(self.db) / snap['filename']) as copy:
                self.assertEqual(copy.execute('SELECT fact FROM memories').fetchone()[0], 'committed in WAL')
            live.execute("UPDATE memories SET fact='later'")
            live.commit()
            restored = memory.restore_snapshot(snap['filename'])
            self.assertEqual(live.execute('SELECT fact FROM memories').fetchone()[0], 'committed in WAL')
            with sqlite3.connect(memory.archive_path(self.db) / restored['safety_backup']) as backup:
                self.assertEqual(backup.execute('SELECT fact FROM memories').fetchone()[0], 'later')

    def test_backup_contention_is_bounded(self):
        destination = str(self.home / 'destination.db')
        with sqlite3.connect(destination) as locked, sqlite3.connect(destination, timeout=0.01) as dst, sqlite3.connect(self.db) as src:
            locked.execute('CREATE TABLE marker(value)')
            locked.commit()
            locked.execute('BEGIN IMMEDIATE')
            with self.assertRaises(TimeoutError):
                memory._backup_connections(src, dst, timeout=0.05)
            locked.rollback()
            self.assertIsNotNone(dst.execute("SELECT name FROM sqlite_master WHERE name='marker'").fetchone())

    def test_corrupt_target_restore_fails_closed(self):
        snapshot = memory.create_snapshot()
        Path(self.db).write_bytes(b'corrupt database header')
        original = Path(self.db).read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'offline recovery'):
            memory.restore_snapshot(snapshot['filename'])
        self.assertEqual(Path(self.db).read_bytes(), original)

    def test_fresh_preview_does_not_initialize_database(self):
        fresh = str(self.home / 'fresh.db')
        with patch.object(schema, 'DB_PATH', fresh):
            result = memory.optimize_db(False)
            self.assertEqual(result['memories'], 0)
            self.assertFalse(Path(fresh).exists())
            self.assertFalse((memory.archive_path(self.db)).exists())
            result = memory.optimize_db(True)
            self.assertTrue(result['applied'])
            self.assertTrue(Path(fresh).exists())
        schema._SCHEMA_INITIALIZED.discard(fresh)

    def test_corrupt_restore_preserves_target_and_archive(self):
        memory.upsert_fact('safe', 'infra', 'original')
        archive = memory.archive_path(self.db)
        archive.mkdir(parents=True)
        (archive / 'memory_db_backup_bad.bak').write_bytes(b'not sqlite')
        before = set(archive.iterdir())
        with self.assertRaises(sqlite3.DatabaseError):
            memory.restore_snapshot('memory_db_backup_bad.bak')
        self.assertEqual(set(archive.iterdir()), before)
        with schema.db_session() as conn:
            self.assertEqual(conn.execute('SELECT fact FROM memories').fetchone()[0], 'original')

    def test_atomic_rollback_after_fact_write(self):
        with patch.object(memory, 'upsert_learning', side_effect=RuntimeError('injected write failure')):
            with self.assertRaises(memory.SyncExtractionError):
                self.extract({'facts': [{'id': 'new', 'fact': 'must roll back'}],
                              'learnings': [{'id': 'lesson', 'insight': 'failure'}]})
        with schema.db_session() as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM memories').fetchone()[0], 0)
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM memories_fts').fetchone()[0], 0)

    def test_successful_transaction_and_child_environment(self):
        result, run = self.extract({'facts': [{'id': 'node', 'fact': 'new fact'}],
                                    'learnings': [{'id': 'lesson', 'insight': 'new learning'}],
                                    'entity_links': [{'source': 'node', 'target': 'lesson', 'relation': 'uses'}]})
        self.assertEqual(len(result['entity_links']), 1)
        self.assertEqual(run.call_args.kwargs['env']['AGY_INTERNAL_INVOCATION'], '1')
        self.assertEqual(run.call_args.kwargs['env']['AGY_SAGE_DISABLED'], '1')

    def test_extraction_normalizes_graph_endpoint_whitespace(self):
        result, _ = self.extract({'facts': [{'id': ' node ', 'fact': 'node'}],
                                 'learnings': [{'id': ' lesson ', 'insight': 'insight'}],
                                 'entity_links': [{'source': ' node ', 'target': ' lesson ', 'relation': 'uses'}]})
        self.assertEqual(result['facts'][0]['id'], 'node')
        with schema.db_session() as conn:
            self.assertEqual(conn.execute('SELECT * FROM entity_links').fetchall(), [('node', 'lesson', 'uses')])
            self.assertEqual(memory.prune_orphan_links(dry_run=True), 0)

    def test_invalid_extraction_is_not_empty_success(self):
        for data in ({}, {'facts': None}, {'facts': [{}]}, {'facts': [{'id': 'x', 'fact': 3}]}):
            with self.subTest(data=data), self.assertRaises(memory.SyncExtractionError):
                self.extract(data)
        with patch.object(memory.subprocess, 'run', return_value=Mock(stdout='garbage', returncode=0)):
            with self.assertRaises(memory.SyncExtractionError):
                memory.sync_turn('Remember the database hostname for deployments.', 'yes')
        result, _ = self.extract({'facts': [], 'episodes': [], 'learnings': [], 'entity_links': []})
        self.assertFalse(any(result.values()))

    def test_busy_and_model_errors_are_typed(self):
        import fcntl
        with patch.object(fcntl, 'flock', side_effect=BlockingIOError):
            with self.assertRaises(memory.SyncBusyError):
                memory.sync_turn('Save the deployment hostname.', 'yes')
        with patch.object(memory.subprocess, 'run', side_effect=RuntimeError('model crash')), patch.object(memory, 'discover_and_cache_latest_flash_low_model', return_value='test'):
            with self.assertRaises(memory.SyncExtractionError):
                memory.sync_turn('Save the deployment hostname.', 'yes')

    def test_protected_categories_with_opaque_ids(self):
        memory.upsert_fact('opaque', 'preferences', 'keep this')
        memory.upsert_learning('opaque_lesson', 'health', 'keep this too')
        result, _ = self.extract({'facts': [{'id': 'opaque', 'category': 'infra', 'fact': 'overwrite'}],
                                 'learnings': [{'id': 'opaque_lesson', 'category': 'general', 'insight': 'overwrite'}]})
        self.assertFalse(result['facts'])
        self.assertFalse(result['learnings'])
        self.assertTrue(memory._is_protected_key('opaque'))
        with schema.db_session() as conn:
            self.assertEqual(conn.execute('SELECT insight FROM learnings').fetchone()[0], 'keep this too')

    def test_unknown_graph_endpoint_rolls_back(self):
        with self.assertRaises(memory.SyncExtractionError):
            self.extract({'facts': [{'id': 'node', 'fact': 'new'}], 'entity_links': [
                {'source': 'node', 'target': 'missing', 'relation': 'uses'}]})
        with schema.db_session() as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM memories').fetchone()[0], 0)

    def merge(self, target='a', category='infra'):
        proposal = {'merges': [{'target_id': target, 'category': category, 'fact': 'merged',
                                'keywords': 'merge', 'merged_ids': ['b'], 'rationale': 'duplicate'}]}
        with patch.object(memory.subprocess, 'run', return_value=Mock(stdout=json.dumps(proposal), returncode=0)) as run:
            result = memory.consolidate_memories()
        self.assertEqual(run.call_args.kwargs['env']['AGY_SAGE_DISABLED'], '1')
        return result

    def test_consolidation_rewrites_both_graph_directions_and_logs_preimages(self):
        for fid in ('a', 'b', 'outside'):
            memory.upsert_fact(fid, 'infra', fid)
        memory.link_entities('outside', 'a', 'uses')
        memory.link_entities('outside', 'b', 'uses')
        memory.link_entities('b', 'outside', 'depends_on')
        self.assertEqual(len(self.merge()), 1)
        with schema.db_session() as conn:
            links = set(conn.execute('SELECT * FROM entity_links'))
            self.assertEqual(links, {('outside', 'a', 'uses'), ('a', 'outside', 'depends_on')})
            preimage = json.loads(conn.execute('SELECT diff_summary FROM consolidation_log').fetchone()[0])
            self.assertEqual(len(preimage['entity_links']), 2)
            self.assertEqual(preimage['target'][2], 'a')
            self.assertEqual(preimage['facts'][0][0], 'b')

    def test_consolidation_removes_collapsed_self_links(self):
        for fid in ('a', 'b'):
            memory.upsert_fact(fid, 'infra', fid)
        memory.link_entities('a', 'b', 'depends_on')
        self.assertEqual(len(self.merge()), 1)
        with schema.db_session() as conn:
            self.assertEqual(conn.execute('SELECT * FROM entity_links').fetchall(), [])
            preimage = json.loads(conn.execute('SELECT diff_summary FROM consolidation_log').fetchone()[0])
            self.assertEqual(preimage['entity_links'], [['a', 'b', 'depends_on']])

    def test_consolidation_rejects_other_category_and_protected_targets(self):
        for fid in ('a', 'b'):
            memory.upsert_fact(fid, 'infra', fid)
        for category in ('software', 'health', 'preferences'):
            memory.upsert_fact('victim', category, 'protected original')
            self.assertEqual(self.merge('victim'), [])
        with schema.db_session() as conn:
            self.assertEqual(conn.execute("SELECT fact FROM memories WHERE id='victim'").fetchone()[0], 'protected original')
            self.assertIsNotNone(conn.execute("SELECT id FROM memories WHERE id='b'").fetchone())

    def test_preview_has_no_writes_and_apply_repairs_fts(self):
        memory.upsert_fact('node', 'infra', 'needle', 'needle')
        memory.upsert_episode('old', 'infra', 'old', 'narrative')
        with schema.db_session() as conn:
            conn.execute("UPDATE episodes SET updated_at='2000-01-01'")
            conn.execute('DELETE FROM memories_fts')
            conn.commit()
        before = {p.name: p.read_bytes() for p in self.home.rglob('*') if p.is_file() and not p.name.endswith(('-wal', '-shm'))}
        output = io.StringIO()
        with redirect_stdout(output):
            result = memory.optimize_db(apply_changes=False)
        self.assertFalse(result['applied'])
        self.assertEqual(output.getvalue(), '')
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.home.rglob('*') if p.is_file() and not p.name.endswith(('-wal', '-shm'))})
        memory.optimize_db(apply_changes=True)
        with schema.db_session() as conn:
            self.assertEqual(conn.execute("SELECT id FROM memories_fts WHERE memories_fts MATCH 'needle'").fetchone()[0], 'node')
            self.assertEqual(conn.execute('SELECT status FROM episodes').fetchone()[0], 'historic')

    def test_vocabulary_cache_and_linked_learning_short_tokens(self):
        memory.upsert_fact('host', 'infra', 'IP for deployment', 'IP')
        memory.upsert_learning('lesson', 'architecture', 'Use immutable backups')
        memory.link_entities('host', 'lesson', 'related_to')
        with schema.db_session() as conn:
            statements = []
            conn.set_trace_callback(statements.append)
            first = memory.get_all_vocabulary(conn.cursor())
            statements.clear()
            self.assertEqual(memory.get_all_vocabulary(conn.cursor()), first)
            self.assertFalse(any('FROM memories' in sql for sql in statements))
        self.assertTrue({'ip', 'ai', 'db', 'r2', 'ci'} <= set(memory.extract_multilingual_tokens('IP AI DB R2 CI')))
        result = memory.prefetch('IP', quiet=True)
        self.assertTrue(any('Use immutable backups' in s for s in result['linked_context']))
        memory.upsert_fact('new', 'infra', 'freshvocabulary')
        with schema.db_session() as conn:
            self.assertIn('freshvocabulary', memory.get_all_vocabulary(conn.cursor()))

    def test_canonical_writers_and_prescription_inverse(self):
        from scripts.migrate_v2_to_v2_1 import map_relation
        self.assertEqual(map_relation('med', 'doctor', 'prescribed_by'), ('doctor', 'med', 'prescribes'))
        memory.upsert_fact('f', 'infrastructure', 'fact')
        with self.assertRaises(ValueError):
            memory.upsert_learning('l', 'nonsense', 'insight')
        memory.upsert_learning('l', 'general', 'insight')
        with schema.db_session() as conn:
            self.assertEqual(conn.execute('SELECT category FROM memories').fetchone()[0], 'infra')
            self.assertEqual(conn.execute('SELECT category FROM learnings').fetchone()[0], 'general')

    def test_mcp_stdio_preview_transport(self):
        import subprocess
        import selectors
        import time
        env = dict(os.environ, AGY_MEMORY_DB=self.db,
                   AGY_TURN_QUEUE_DB=str(self.home / 'queue.db'))
        process = subprocess.Popen([sys.executable, '-u', str(Path(memory.__file__).with_name('agy_memory_mcp.py'))],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, env=env)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)

        def request(message, response_id):
            process.stdin.write(json.dumps(message) + '\n')
            process.stdin.flush()
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if not selector.select(timeout=1):
                    continue
                line = process.stdout.readline()
                self.assertTrue(line, 'MCP server closed stdout')
                response = json.loads(line)  # Any stray diagnostic fails here.
                self.assertEqual(response.get('jsonrpc'), '2.0')
                if response.get('id') == response_id:
                    return response
            self.fail('MCP response timeout')

        try:
            initialized = request({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {
                'protocolVersion': '2024-11-05', 'capabilities': {},
                'clientInfo': {'name': 'hardening-tests', 'version': '1'}}}, 1)
            self.assertIn('result', initialized)
            process.stdin.write(json.dumps({'jsonrpc': '2.0', 'method': 'notifications/initialized'}) + '\n')
            process.stdin.flush()
            response = request({'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call', 'params': {
                'name': 'optimize_memory', 'arguments': {'apply_changes': False, 'consolidate': False}}}, 2)
            self.assertNotIn('error', response)
            result = json.loads(response['result']['content'][0]['text'])
            self.assertEqual(result['status'], 'success')
            self.assertFalse(result['stats']['applied'])
            self.assertFalse((memory.archive_path(self.db)).exists())
        finally:
            selector.close()
            process.terminate()
            process.communicate(timeout=5)

    def test_mcp_optimization_stdout_guard(self):
        from agy_memory_mcp import optimize_memory
        output = io.StringIO()
        with redirect_stdout(output):
            result = json.loads(optimize_memory(False))
        self.assertEqual(result['status'], 'success')
        self.assertEqual(output.getvalue(), '')
