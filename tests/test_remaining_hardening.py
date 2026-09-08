"""Failure-boundary regressions added after the initial hardening pass."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_environment
import json
import sqlite3
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch, Mock
import agy_memory as memory
import schema

class RemainingHardeningTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.db = str(self.root / 'memory.db')
        self.stack.enter_context(patch.object(schema, 'DB_PATH', self.db))
        self.stack.enter_context(patch.object(memory, 'CACHE_PATH', str(self.root / 'model')))
        self.stack.enter_context(patch.object(memory.tempfile, 'gettempdir', return_value=str(self.root)))
        with schema.db_session():
            pass

    def response(self, data):
        return Mock(returncode=0, stdout=json.dumps(data))

    def test_committed_receipt_replay_does_not_extract_again(self):
        data = {'facts': [{'id':'server','category':'infra','fact':'database server'}]}
        with patch.object(memory.subprocess, 'run', return_value=self.response(data)) as model:
            first = memory.sync_turn('Remember this database server configuration', 'saved', batch_id='batch-1')
            second = memory.sync_turn('Remember this database server configuration', 'saved', batch_id='batch-1')
        self.assertEqual(first, second)
        self.assertEqual(model.call_count, 1)
        with schema.db_session() as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM batch_receipts').fetchone()[0], 1)

    def test_stale_extraction_rejects_entire_transaction(self):
        memory.upsert_fact('server', 'infra', 'original')
        def concurrent_edit(*args, **kwargs):
            memory.upsert_fact('server', 'infra', 'manual newer edit')
            return self.response({'facts':[{'id':'new','category':'infra','fact':'must roll back'},
                                           {'id':'server','category':'infra','fact':'stale model'}]})
        with patch.object(memory.subprocess, 'run', side_effect=concurrent_edit):
            with self.assertRaises(memory.SyncExtractionError):
                memory.sync_turn('Remember this database server configuration', 'saved')
        with schema.db_session() as conn:
            self.assertEqual(conn.execute('SELECT id,fact FROM memories').fetchall(), [('server','manual newer edit')])

    def test_omitted_update_fields_are_preserved(self):
        memory.upsert_fact('server','infra','original','postgres sql')
        with patch.object(memory.subprocess,'run',return_value=self.response({'facts':[{'id':'server','fact':'new content'}]})):
            memory.sync_turn('Remember this database server configuration', 'saved')
        with schema.db_session() as conn:
            self.assertEqual(conn.execute('SELECT category,keywords FROM memories').fetchone(), ('infra','postgres sql'))

    def test_invalid_taxonomy_rejected_without_flattening(self):
        with self.assertRaises(ValueError):
            memory.upsert_fact('bad','invented-category','some content')
        memory.upsert_fact('architecture','architecture','preserve useful distinction')
        with schema.db_session() as conn:
            conn.execute("INSERT INTO memories(id, category, fact) VALUES ('legacy','legacy-unknown','preserve provenance')")
            conn.commit()
        memory.normalize_existing_categories()
        with schema.db_session() as conn:
            self.assertEqual(conn.execute("SELECT category FROM memories WHERE id='legacy'").fetchone()[0], 'legacy-unknown')

    def test_link_and_unlink_share_inverse_mapping(self):
        memory.upsert_fact('host','infra','server')
        memory.upsert_fact('service','software','web server')
        memory.link_entities('host','service',' HOSTS ')
        self.assertEqual(memory.list_entity_links(), [('service','host','hosted_on')])
        memory.unlink_entities('host','service','hosts')
        self.assertEqual(memory.list_entity_links(), [])

    def test_schema_reinitializes_replaced_database(self):
        Path(self.db).unlink()
        with schema.db_session() as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM memories').fetchone()[0], 0)
            self.assertGreaterEqual(conn.execute('PRAGMA user_version').fetchone()[0], 211)

    def test_commit_ack_crash_replay_keeps_original_batch_membership(self):
        import queue_manager as queue
        import memory_worker as worker
        queue_db = str(self.root / 'queue.db')
        queue.enqueue_turn('Remember server one configuration', 'saved', chat_id='chat', db_path=queue_db)
        response = self.response({'facts':[{'id':'server','category':'infra','fact':'saved configuration'}]})
        with patch.object(memory.subprocess,'run',return_value=response) as model, patch.object(worker,'acknowledge_batch',side_effect=RuntimeError('ack crash')):
            self.assertEqual(worker.process_queue(notify=False,db_path=queue_db), 0)
        with sqlite3.connect(queue_db) as conn:
            first_id = conn.execute('SELECT batch_id FROM turn_queue').fetchone()[0]
            conn.execute("UPDATE turn_queue SET processed_at=datetime('now','-120 seconds')")
        queue.enqueue_turn('Remember server two configuration', 'saved', chat_id='chat',db_path=queue_db)
        with patch.object(memory.subprocess,'run',side_effect=AssertionError('replay must not infer')):
            self.assertEqual(worker.process_queue(batch_size=1,notify=False,db_path=queue_db),1)
        with sqlite3.connect(queue_db) as conn:
            rows=conn.execute('SELECT batch_id,status FROM turn_queue ORDER BY id').fetchall()
        self.assertEqual(rows[0], (first_id,'processed'))
        self.assertEqual(rows[1], (None,'pending'))

    def test_claim_has_no_hot_schema_statements(self):
        import queue_manager as queue
        queue_db = str(self.root / 'queue.db')
        queue.enqueue_turn('Remember database configuration', 'saved',db_path=queue_db)
        statements=[]
        connect=sqlite3.connect
        def traced(*args,**kwargs):
            conn=connect(*args,**kwargs)
            conn.set_trace_callback(statements.append)
            return conn
        with patch.object(queue.sqlite3,'connect',side_effect=traced):
            queue.claim_batch(db_path=queue_db)
        self.assertFalse(any('CREATE ' in sql.upper() or 'TABLE_INFO' in sql.upper() for sql in statements))

    def test_prefetch_context_has_byte_budget(self):
        for index in range(10):
            memory.upsert_fact(f'rule{index}','preferences','remember unicode 文 '+('字'*100))
        result=memory.prefetch('Remember my preferences',quiet=True,max_context_bytes=512)
        self.assertLessEqual(len(json.dumps(result,ensure_ascii=False).encode()),512)

    def test_tool_free_inference_rejects_tools(self):
        import memory_inference
        response=Mock()
        response.__enter__=Mock(return_value=response)
        response.__exit__=Mock(return_value=False)
        response.read.return_value=json.dumps({'choices':[{'message':{'content':'{}','tool_calls':[{}]}}]}).encode()
        with patch.object(memory_inference,'get_config',side_effect=lambda key, default='': 'http://127.0.0.1/completions' if key=='AGY_MEMORY_INFERENCE_URL' else default), patch.object(memory_inference,'urlopen',return_value=response) as request:
            with self.assertRaisesRegex(ValueError,'Tool calls'):
                memory_inference.infer('remember configuration','test-model')
            body=json.loads(request.call_args.args[0].data)
            self.assertNotIn('tools',body)

    def test_fts_primary_key_and_delete_transitions(self):
        memory.upsert_fact('old','infra','searchable sql')
        with schema.db_session() as conn, conn:
            conn.execute("UPDATE memories SET id='new' WHERE id='old'")
            self.assertEqual(conn.execute("SELECT id FROM memories_fts WHERE memories_fts MATCH 'sql'").fetchall(), [('new',)])
            conn.execute("DELETE FROM memories WHERE id='new'")
            self.assertEqual(conn.execute('SELECT count(*) FROM memories_fts').fetchone()[0],0)

    def test_absolute_paths_are_not_trivial_commands(self):
        self.assertFalse(memory.is_trivial_prompt('/etc/hosts contains production server aliases'))
        self.assertTrue(memory.is_trivial_prompt('/help'))

    def test_manual_status_cannot_steal_claimed_turn(self):
        import queue_manager as queue
        queue_db=str(self.root/'queue.db')
        queue.enqueue_turn('Remember configuration for server', 'saved', db_path=queue_db)
        claim=queue.claim_batch(db_path=queue_db)
        queue.mark_turn_status([claim.turns[0]['id']], 'processed', db_path=queue_db)
        with sqlite3.connect(queue_db) as conn:
            self.assertEqual(conn.execute('SELECT status FROM turn_queue').fetchone()[0], 'claimed')

    def test_restore_waits_for_engine_clients(self):
        with schema.db_session():
            with self.assertRaises(TimeoutError):
                with schema.maintenance_lock(self.db,exclusive=True,timeout=0.01):
                    pass
