"""Jev-scored duplicate candidates: pairing, chunked scoring, export wiring.

Every test patches the Jev transport (`jev_dedupe._call_jev_deadlined`) and the
key, so nothing here can reach the network even when AGY_JEV_API_KEY is set in
the environment.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_environment  # noqa: F401

import io
import json
import math
import os
import re
import sqlite3
import subprocess
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from unittest.mock import Mock, patch

import numpy as np

import agy_memory as memory
import jev_dedupe
import schema
import vector_index


def vec(*vals):
    # Model-dimension vector (embedding shape contract), with `vals` in the
    # leading slots; zero padding keeps the cosine geometry of small fixtures.
    arr = np.zeros(jev_dedupe.EMBEDDING_DIM, dtype=np.float32)
    arr[:len(vals)] = vals
    return arr


def boolean(p):
    return {"type": "boolean", "probability": p}


def answers_for(n, dup=0.9, con=0.1):
    out = {}
    for i in range(1, n + 1):
        out["d%d" % i] = boolean(dup)
        out["c%d" % i] = boolean(con)
    return {"answers": out}


def make_pair(a, b, category="dev", sim=0.9):
    return {"category": category, "a": a, "b": b, "similarity": sim}


class JevedupeTestBase(unittest.TestCase):
    """Patched key and transport in every test (spec item 11)."""

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(jev_dedupe, "JEV_GATE_API_KEY", "test-key"))
        self.stack.enter_context(patch.object(jev_dedupe, "JEV_DEDUPE_ENABLED", True))
        self.stack.enter_context(patch.object(
            jev_dedupe, "JEV_DEDUPE_EXCLUDE", frozenset(schema.PROTECTED_CATEGORIES)))
        self.stack.enter_context(patch.object(jev_dedupe, "JEV_DEDUPE_FLOOR", 0.5))
        self.stack.enter_context(patch.object(jev_dedupe, "JEV_DEDUPE_MIN_SIM", 0.5))
        self.stack.enter_context(patch.object(jev_dedupe, "JEV_DEDUPE_TOP_K", 1))
        self.stack.enter_context(patch.object(jev_dedupe, "JEV_DEDUPE_MAX_PAIRS", 400))
        self.stack.enter_context(patch.object(jev_dedupe, "JEV_DEDUPE_PAIRS_PER_CALL", 20))
        self.stack.enter_context(patch.object(jev_dedupe, "JEV_DEDUPE_BUDGET", 180.0))
        self.stack.enter_context(patch.object(jev_dedupe, "JEV_DEDUPE_REQUEST_BYTES", 48000))
        self.transport = Mock(side_effect=lambda state, questions, timeout: answers_for(len(questions) // 2))
        self.stack.enter_context(patch.object(jev_dedupe, "_call_jev_deadlined", self.transport))


class CandidatePairsTests(JevedupeTestBase):
    """Spec 5.1: top-K, MIN_SIM, unordered dedup, sort, truncation, since filter."""

    @staticmethod
    def triple():
        # a close to b, b close to c; only angles differ on one axis.
        rad = math.radians
        facts = [{"id": fid, "category": "dev", "fact": fid} for fid in ("a", "b", "c")]
        vectors = {
            "a": vec(math.cos(rad(0)), math.sin(rad(0))),
            "b": vec(math.cos(rad(5)), math.sin(rad(5))),
            "c": vec(math.cos(rad(10)), math.sin(rad(10))),
        }
        return facts, vectors

    @staticmethod
    def orthogonal_pairs():
        # Three parallel pairs on disjoint axes: cross similarities are 0.
        rad = math.radians
        facts, vectors = [], {}
        for i, (fid1, fid2, ang) in enumerate((("f1", "f2", 5), ("f3", "f4", 18), ("f5", "f6", 25))):
            base = vec(*([0.0] * (2 * i) + [1.0, 0.0] + [0.0] * (6 - 2 * i - 2)))
            tilt = vec(*([0.0] * (2 * i) + [math.cos(rad(ang)), math.sin(rad(ang))] + [0.0] * (6 - 2 * i - 2)))
            for fid, v in ((fid1, base), (fid2, tilt)):
                facts.append({"id": fid, "category": "dev", "fact": fid})
                vectors[fid] = v
        return facts, vectors

    def test_top_k_and_min_sim(self):
        facts, vectors = self.triple()
        # TOP_K=1: (a,c) at cosine 0.985 never enters, even though >= MIN_SIM.
        pairs = jev_dedupe.candidate_pairs(facts, vectors)
        self.assertEqual([(p["a"], p["b"]) for p in pairs], [("a", "b"), ("b", "c")])
        self.stack.enter_context(patch.object(jev_dedupe, "JEV_DEDUPE_TOP_K", 2))
        pairs = jev_dedupe.candidate_pairs(facts, vectors)
        self.assertEqual([(p["a"], p["b"]) for p in pairs], [("a", "b"), ("b", "c"), ("a", "c")])
        self.stack.enter_context(patch.object(jev_dedupe, "JEV_DEDUPE_MIN_SIM", 0.99))
        pairs = jev_dedupe.candidate_pairs(facts, vectors)
        self.assertEqual([(p["a"], p["b"]) for p in pairs], [("a", "b"), ("b", "c")])

    def test_unordered_dedup_and_sort_and_truncation(self):
        facts, vectors = self.triple()
        self.stack.enter_context(patch.object(jev_dedupe, "JEV_DEDUPE_TOP_K", 2))
        pairs = jev_dedupe.candidate_pairs(list(reversed(facts)), vectors)
        keys = [(p["a"], p["b"]) for p in pairs]
        self.assertEqual(keys, [("a", "b"), ("b", "c"), ("a", "c")])
        self.assertEqual(len(keys), len(set(keys)), "each unordered pair appears once")
        self.assertTrue(all(a < b for a, b in keys), "pairs are normalized to a < b")
        self.assertEqual([p["similarity"] for p in pairs],
                         sorted((p["similarity"] for p in pairs), reverse=True))
        self.stack.enter_context(patch.object(jev_dedupe, "JEV_DEDUPE_MAX_PAIRS", 1))
        self.assertEqual(len(jev_dedupe.candidate_pairs(facts, vectors)), 1)

    def test_since_filter_keeps_one_fresh_side(self):
        facts, vectors = self.orthogonal_pairs()
        updated = {"f1": "2026-09-26 10:00:00", "f2": "2026-09-26 09:00:00",
                   "f3": "2026-01-01 00:00:00", "f4": "2026-01-01 00:00:00"}
        pairs = jev_dedupe.candidate_pairs(facts, vectors, updated, "2026-09-26 09:30:00")
        self.assertEqual([(p["a"], p["b"]) for p in pairs], [("f1", "f2")])
        pairs = jev_dedupe.candidate_pairs(facts, vectors, updated, "2026-09-27 00:00:00")
        self.assertEqual(pairs, [])

    def test_facts_without_vectors_are_skipped(self):
        facts, vectors = self.triple()
        del vectors["b"]
        pairs = jev_dedupe.candidate_pairs(facts, vectors)
        self.assertEqual([(p["a"], p["b"]) for p in pairs], [("a", "c")])


class LoadVectorsTests(JevedupeTestBase):
    """Spec 5.2/5.8b/5.8c at the stored-vector layer."""

    @staticmethod
    def raw_conn():
        # Plain stand-in tables let tests insert malformed blobs that vec0
        # would reject at INSERT time.
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE vec_memories (id TEXT PRIMARY KEY, embedding BLOB)")
        conn.execute("CREATE TABLE vector_index_state (entity_type TEXT, entity_id TEXT,"
                     " indexed_revision INTEGER, generation TEXT, model_fingerprint TEXT,"
                     " text_hash TEXT, indexed_at TIMESTAMP)")
        return conn

    @staticmethod
    def seed_state(conn, entity_id, indexed_revision, generation, fingerprint):
        conn.execute("INSERT INTO vector_index_state VALUES ('memories', ?, ?, ?, ?, NULL, NULL)",
                     (entity_id, indexed_revision, generation, fingerprint))

    def test_only_vectors_fresh_for_the_snapshot_revision_are_used(self):
        conn = self.raw_conn()
        generation = schema.get_db_generation(conn)
        fingerprint = vector_index.get_active_model_fingerprint(conn)
        arr = vec(0.5, 0.5)
        conn.execute("INSERT INTO vec_memories VALUES ('a', ?)", (arr.tobytes(),))
        self.seed_state(conn, "a", 1, generation, fingerprint)
        # b's stored vector was indexed for a different revision than the snapshot.
        conn.execute("INSERT INTO vec_memories VALUES ('b', ?)", (arr.tobytes(),))
        self.seed_state(conn, "b", 2, generation, fingerprint)
        # c's stored vector is indexed fresh but under a stale generation.
        conn.execute("INSERT INTO vec_memories VALUES ('c', ?)", (arr.tobytes(),))
        self.seed_state(conn, "c", 1, "other-generation", fingerprint)
        out = jev_dedupe.load_vectors(conn, {"a": 1, "b": 1, "c": 1})
        self.assertEqual(set(out), {"a"})
        np.testing.assert_allclose(out["a"], arr)

    def test_malformed_stored_vectors_are_treated_as_missing(self):
        conn = self.raw_conn()
        generation = schema.get_db_generation(conn)
        fingerprint = vector_index.get_active_model_fingerprint(conn)
        good = vec(0.5, 0.5)
        conn.execute("INSERT INTO vec_memories VALUES ('good', ?)", (good.tobytes(),))
        self.seed_state(conn, "good", 1, generation, fingerprint)
        wrong_dim = np.zeros(10, dtype=np.float32).tobytes()
        conn.execute("INSERT INTO vec_memories VALUES ('dim', ?)", (wrong_dim,))
        self.seed_state(conn, "dim", 1, generation, fingerprint)
        nan_blob = np.full(jev_dedupe.EMBEDDING_DIM, np.nan, dtype=np.float32).tobytes()
        conn.execute("INSERT INTO vec_memories VALUES ('nan', ?)", (nan_blob,))
        self.seed_state(conn, "nan", 1, generation, fingerprint)
        conn.execute("INSERT INTO vec_memories VALUES ('junk', ?)", (b"not a vector",))
        self.seed_state(conn, "junk", 1, generation, fingerprint)
        out = jev_dedupe.load_vectors(conn, {"good": 1, "dim": 1, "nan": 1, "junk": 1})
        self.assertEqual(set(out), {"good"})


class ScorePairsTests(JevedupeTestBase):
    """Spec 5.3-5.7: chunking, validation, failures, exclusion, caps, budget."""

    @staticmethod
    def pairs(n):
        return [make_pair("a%d" % i, "b%d" % i) for i in range(n)]

    @staticmethod
    def texts_for(pairs):
        texts = {}
        for p in pairs:
            texts[p["a"]] = "fact about " + p["a"]
            texts[p["b"]] = "fact about " + p["b"]
        return texts

    def test_chunking_question_ids_and_criterion(self):
        self.stack.enter_context(patch.object(jev_dedupe, "JEV_DEDUPE_PAIRS_PER_CALL", 20))
        pairs = self.pairs(45)

        def mapping_transport(state, questions, timeout):
            # Distinct values per pair, derived from the ids in the criterion
            # lines: a swapped question-to-pair mapping fails the asserts below.
            answers = {}
            for line in state["criterion"].splitlines():
                if not line.startswith("[p"):
                    continue  # criterion preamble
                m = re.match(r"\[p(\d+)\] A \((a\d+)\):.*\|\| B \((b\d+)\):", line)
                assert m is not None, line
                n, idx = int(m.group(1)), int(m.group(2)[1:])
                answers["d%d" % n] = boolean(round(0.01 * idx, 3))
                answers["c%d" % n] = boolean(round(0.5 + 0.01 * idx, 3))
            return {"answers": answers}

        self.transport.side_effect = mapping_transport
        results, meta = jev_dedupe.score_pairs(pairs, self.texts_for(pairs))
        self.assertEqual(self.transport.call_count, 3)
        for call, n_pairs in zip(self.transport.call_args_list, (20, 20, 5)):
            state, questions, _timeout = call.args
            self.assertEqual(sorted(questions),
                             sorted(["d%d" % i for i in range(1, n_pairs + 1)]
                                    + ["c%d" % i for i in range(1, n_pairs + 1)]))
            for i in range(n_pairs):
                self.assertIn("duplicates as defined", questions["d%d" % (i + 1)]["instructions"])
                self.assertIn("conflict as defined", questions["c%d" % (i + 1)]["instructions"])
        state = self.transport.call_args_list[0].args[0]
        for p in pairs[:20]:
            self.assertIn(p["a"], state["criterion"])
            self.assertIn(p["b"], state["criterion"])
        self.assertTrue(all(r is not None for r in results))
        for i, res in enumerate(results):
            self.assertEqual(res, {"duplicate": round(0.01 * i, 3),
                                   "conflict": round(0.5 + 0.01 * i, 3)},
                             "pair %d must carry its own duplicate/conflict" % i)
        self.assertEqual(len({r["duplicate"] for r in results if r is not None}), 45)
        self.assertEqual(len({r["conflict"] for r in results if r is not None}), 45)
        self.assertEqual(meta["calls"], 3)
        self.assertEqual(meta["scored_pairs"], 45)
        self.assertEqual(meta["status"], "scored")

    def test_one_bad_answer_poisons_only_its_chunk(self):
        self.stack.enter_context(patch.object(jev_dedupe, "JEV_DEDUPE_PAIRS_PER_CALL", 2))
        pairs = self.pairs(4)
        bad = {"answers": {"d1": boolean(0.9), "c1": boolean(0.9),
                           "d2": boolean(0.9), "c2": {"type": "boolean", "probability": "oops"}}}
        self.transport.side_effect = [answers_for(2), bad]
        results, meta = jev_dedupe.score_pairs(pairs, self.texts_for(pairs))
        self.assertIsNotNone(results[0])
        self.assertIsNotNone(results[1])
        self.assertIsNone(results[2])
        self.assertIsNone(results[3])
        self.assertEqual(meta["status"], "partial")
        self.assertEqual(meta["scored_pairs"], 2)

    def test_failing_call_leaves_chunk_unscored(self):
        pairs = self.pairs(2)
        self.transport.side_effect = OSError("down")
        results, meta = jev_dedupe.score_pairs(pairs, self.texts_for(pairs))
        self.assertEqual(results, [None, None])
        self.assertEqual(meta["status"], "unavailable")
        self.assertEqual(meta["calls"], 1)
        # One failing and one working call is partial.
        self.transport.side_effect = [OSError("down"), answers_for(1)]
        self.stack.enter_context(patch.object(jev_dedupe, "JEV_DEDUPE_PAIRS_PER_CALL", 1))
        results, meta = jev_dedupe.score_pairs(pairs, self.texts_for(pairs))
        self.assertEqual(results, [None, {"duplicate": 0.9, "conflict": 0.1}])
        self.assertEqual(meta["status"], "partial")

    def test_disabled_or_empty_key_makes_zero_calls(self):
        pairs = self.pairs(3)
        self.stack.enter_context(patch.object(jev_dedupe, "JEV_DEDUPE_ENABLED", False))
        results, meta = jev_dedupe.score_pairs(pairs, self.texts_for(pairs))
        self.assertEqual(meta["status"], "disabled")
        self.assertEqual(meta["calls"], 0)
        self.assertEqual(results, [None, None, None])
        self.stack.enter_context(patch.object(jev_dedupe, "JEV_DEDUPE_ENABLED", True))
        self.stack.enter_context(patch.object(jev_dedupe, "JEV_GATE_API_KEY", ""))
        results, meta = jev_dedupe.score_pairs(pairs, self.texts_for(pairs))
        self.assertEqual(meta["status"], "disabled")
        self.assertEqual(meta["calls"], 0)

    def test_excluded_category_never_enters_a_request(self):
        # Effective exclusion adds configured categories to the protected set:
        # dev (configured), health and finance (protected) are never sent.
        self.stack.enter_context(patch.object(
            jev_dedupe, "JEV_DEDUPE_EXCLUDE",
            frozenset(schema.PROTECTED_CATEGORIES) | {"dev"}))
        pairs = [make_pair("dv1", "dv2"), make_pair("fx1", "fx2", category="finance"),
                 make_pair("dv3", "dv4"), make_pair("hp1", "hp2", category="health"),
                 make_pair("m1", "m2", category="misc"), make_pair("m3", "m4", category="misc")]
        self.stack.enter_context(patch.object(jev_dedupe, "JEV_DEDUPE_PAIRS_PER_CALL", 3))
        self.transport.side_effect = lambda state, questions, timeout: answers_for(len(questions) // 2)
        results, meta = jev_dedupe.score_pairs(pairs, self.texts_for(pairs))
        self.assertEqual(meta["excluded_pairs"], 4)
        for i in (0, 1, 2, 3):
            self.assertIsNone(results[i])
        self.assertEqual(meta["scored_pairs"], 2)
        for call in self.transport.call_args_list:
            state, questions, _timeout = call.args
            body = state["criterion"]
            self.assertNotIn("dv1", body)
            self.assertNotIn("dv2", body)
            self.assertNotIn("dv3", body)
            self.assertNotIn("dv4", body)
            self.assertNotIn("fx1", body)
            self.assertNotIn("fx2", body)
            self.assertNotIn("hp1", body)
            self.assertNotIn("hp2", body)
            self.assertIn("m1", body, "non-excluded pairs still go out")
            self.assertEqual(len(questions), 4, "only sendable pairs consume question slots")

    def test_request_byte_cap_sends_only_small_bodies_and_splits(self):
        pairs = self.pairs(4)
        texts = {pid: "x" * 5000 for p in pairs for pid in (p["a"], p["b"])}
        chunk = [(i, p) for i, p in enumerate(pairs)]
        size_4 = len(json.dumps(jev_dedupe._build_body(chunk, texts, jev_dedupe.PAIR_TEXT_MIN)).encode("utf-8"))
        size_2 = len(json.dumps(jev_dedupe._build_body(chunk[:2], texts, jev_dedupe.PAIR_TEXT_MIN)).encode("utf-8"))
        self.assertGreater(size_4, size_2)
        cap = (size_4 + size_2) // 2
        self.stack.enter_context(patch.object(jev_dedupe, "JEV_DEDUPE_REQUEST_BYTES", cap))
        self.stack.enter_context(patch.object(jev_dedupe, "JEV_DEDUPE_PAIRS_PER_CALL", 4))
        results, meta = jev_dedupe.score_pairs(pairs, texts)
        for call in self.transport.call_args_list:
            state, questions, _timeout = call.args
            body = json.dumps({"state": state, "questions": questions, "providerOptions": {}})
            self.assertLessEqual(len(body.encode("utf-8")), cap)
        self.assertGreater(meta["calls"], 1, "the chunk must be split")
        self.assertLessEqual(meta["max_request_bytes"], cap)
        self.assertTrue(all(r is not None for r in results))
        self.assertEqual(meta["status"], "scored")

    def test_single_pair_that_cannot_fit_stays_unscored(self):
        pair = make_pair("a1", "b1")
        texts = {"a1": "x" * 5000, "b1": "y" * 5000}
        chunk = [(0, pair)]
        size_1 = len(json.dumps(jev_dedupe._build_body(chunk, texts, jev_dedupe.PAIR_TEXT_MIN)).encode("utf-8"))
        self.stack.enter_context(patch.object(jev_dedupe, "JEV_DEDUPE_REQUEST_BYTES", size_1 - 1))
        results, meta = jev_dedupe.score_pairs([pair], texts)
        self.assertEqual(results, [None])
        self.assertEqual(meta["calls"], 0)
        self.assertEqual(meta["status"], "unavailable")

    def test_budget_stops_later_chunks(self):
        pairs = self.pairs(3)
        self.stack.enter_context(patch.object(jev_dedupe, "JEV_DEDUPE_PAIRS_PER_CALL", 1))
        ticks = [0.0, 5.0]
        self.stack.enter_context(patch.object(
            jev_dedupe, "_now", lambda: ticks.pop(0) if ticks else 10000.0))
        results, meta = jev_dedupe.score_pairs(pairs, self.texts_for(pairs))
        self.assertEqual(self.transport.call_count, 1)
        self.assertIsNotNone(results[0])
        self.assertIsNone(results[1])
        self.assertIsNone(results[2])
        self.assertEqual(meta["status"], "partial")
        self.assertEqual(meta["calls"], 1)


class FindCandidatesTests(JevedupeTestBase):
    """Spec 5.2/5.8/5.8b: rank-only ordering, unembedded, snapshot consistency."""

    def setUp(self):
        super().setUp()
        self.home = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.db = str(self.home / "memory.db")
        self.stack.enter_context(patch.object(schema, "DB_PATH", self.db))
        with schema.db_session(self.db) as conn:
            conn.commit()
        self.addCleanup(schema._SCHEMA_INITIALIZED.discard, self.db)
        self.stack.enter_context(patch.object(vector_index, "embed_text", return_value=None))
        self.embed = self.stack.enter_context(patch("embedder.embed_texts", return_value=None))

    def seed(self, rows):
        """rows: (id, category, fact, updated_at, live_revision)."""
        with schema.db_session(self.db) as conn, conn:
            for fid, cat, fact, ts, rev in rows:
                conn.execute("INSERT INTO memories (id, category, fact, keywords, updated_at)"
                             " VALUES (?, ?, ?, '', ?)", (fid, cat, fact, ts))
                conn.execute("UPDATE entity_revisions SET revision = ?"
                             " WHERE entity_type = 'memories' AND entity_id = ?", (rev, fid))
                if conn.execute("SELECT 1 FROM entity_revisions WHERE entity_type='memories'"
                                " AND entity_id = ?", (fid,)).fetchone() is None:
                    conn.execute("INSERT INTO entity_revisions VALUES ('memories', ?, ?)", (fid, rev))

    def snapshot(self, facts, revisions=None):
        # A real export snapshot carries the live generation at export time.
        revisions = revisions or {f["id"]: 1 for f in facts}
        with schema.db_session(self.db) as conn, conn:
            generation = schema.get_db_generation(conn)
        return {"generation": generation, "revisions": revisions,
                "categories": {"dev": facts}}

    @staticmethod
    def parallel_vectors(ids):
        # Three near-parallel pairs on disjoint axis pairs, with distinct
        # similarities (cos 5deg, cos 18deg, cos 25deg) and ~0 cross similarity.
        rad = math.radians
        out = {}
        angles = (5, 18, 25)
        for i in range(3):
            base = [0.0] * (2 * i) + [1.0, 0.0] + [0.0] * (6 - 2 * i - 2)
            tilt = [0.0] * (2 * i) + [math.cos(rad(angles[i])), math.sin(rad(angles[i]))] + [0.0] * (6 - 2 * i - 2)
            out[ids[2 * i]] = vec(*base)
            out[ids[2 * i + 1]] = vec(*tilt)
        return out

    def test_unembedded_reported_and_pairs_skip_missing(self):
        facts = [{"id": fid, "category": "dev", "fact": "fact " + fid} for fid in ("f1", "f2", "f3")]
        self.seed([(f["id"], "dev", f["fact"], "2026-09-26 10:00:00", 1) for f in facts])
        self.embed.side_effect = [[vec(1.0, 0.0), None, vec(0.99, 0.1)]]
        items, meta = jev_dedupe.find_candidates(self.snapshot(facts))
        self.assertEqual([(it["a"], it["b"]) for it in items], [("f1", "f3")])
        self.assertEqual(meta["unembedded"], {"dev": ["f2"]})
        self.assertEqual(meta["pairs_considered"], 1)
        self.assertEqual(meta["jev"], "scored")
        self.assertEqual(meta["stale_ids"], [])
        self.assertIsNotNone(meta["generation"])
        self.assertIsNotNone(meta["model_fingerprint"])

    def test_no_vectors_reports_no_embeddings(self):
        facts = [{"id": "f1", "category": "dev", "fact": "one"},
                 {"id": "f2", "category": "dev", "fact": "two"}]
        self.seed([(f["id"], "dev", f["fact"], "2026-09-26 10:00:00", 1) for f in facts])
        self.embed.return_value = None
        items, meta = jev_dedupe.find_candidates(self.snapshot(facts))
        self.assertEqual(items, [])
        self.assertEqual(meta["unembedded"], {"dev": ["f1", "f2"]})
        self.assertEqual(meta["jev"], "no_embeddings")
        self.assertEqual(meta["calls"], 0)

    def test_rank_only_order_and_meta_counts(self):
        facts = [{"id": fid, "category": "dev", "fact": "fact " + fid}
                 for fid in ("f1", "f2", "f3", "f4", "f5", "f6")]
        self.seed([(f["id"], "dev", f["fact"], "2026-09-26 10:00:00", 1) for f in facts])
        vectors = self.parallel_vectors([f["id"] for f in facts])
        self.embed.side_effect = [[vectors[f["id"]] for f in facts]]
        self.stack.enter_context(patch.object(jev_dedupe, "JEV_DEDUPE_PAIRS_PER_CALL", 2))
        self.transport.side_effect = [
            {"answers": {"d1": boolean(0.9), "c1": boolean(0.1),
                         "d2": boolean(0.1), "c2": boolean(0.2)}},
            OSError("down"),
        ]
        items, meta = jev_dedupe.find_candidates(self.snapshot(facts))
        self.assertEqual(len(items), 3, "rank-only: nothing is dropped")
        self.assertEqual([(it["a"], it["b"]) for it in items],
                         [("f1", "f2"), ("f5", "f6"), ("f3", "f4")])
        self.assertEqual(items[0]["below_floor"], False)
        self.assertEqual(items[0]["duplicate"], 0.9)
        self.assertIsNone(items[1]["duplicate"])
        self.assertIsNone(items[1]["conflict"])
        self.assertFalse(items[1]["below_floor"])
        self.assertTrue(items[2]["below_floor"])
        self.assertEqual(meta["pairs_considered"], 3)
        self.assertEqual(meta["above_floor"], 1)
        self.assertEqual(meta["unscored"], 1)
        self.assertEqual(meta["below_floor"], 1)
        self.assertEqual(meta["jev"], "partial")
        self.assertEqual(meta["calls"], 2)
        self.assertEqual(meta["floor"], 0.5)

    def test_snapshot_consistency_stale_ids_and_reembed(self):
        facts = [{"id": "a", "category": "dev", "fact": "alpha"},
                 {"id": "b", "category": "dev", "fact": "beta"}]
        self.seed([(f["id"], "dev", f["fact"], "2026-09-26 10:00:00", 1) for f in facts])
        with schema.db_session(self.db) as conn, conn:
            generation = schema.get_db_generation(conn)
            fingerprint = vector_index.get_active_model_fingerprint(conn)
            arr = vec(0.5, 0.5)
            for fid in ("a", "b"):
                conn.execute("INSERT INTO vec_memories (id, embedding) VALUES (?, ?)",
                             (fid, arr.tobytes()))
                conn.execute("INSERT INTO vector_index_state"
                             " (entity_type, entity_id, indexed_revision, generation, model_fingerprint)"
                             " VALUES ('memories', ?, 1, ?, ?)", (fid, generation, fingerprint))
        # Snapshot says revision 9 for b while the live row is still revision 1:
        # b's stored vector is not fresh for the snapshot and must be re-embedded
        # from the snapshot text; b is also reported stale.
        self.embed.side_effect = [[vec(0.5, 0.5)]]
        items, meta = jev_dedupe.find_candidates(self.snapshot(facts, {"a": 1, "b": 9}))
        self.embed.assert_called_once()
        embedded_text = self.embed.call_args[0][0][0]
        self.assertIn("beta", embedded_text, "b is re-embedded from its snapshot text")
        self.assertEqual(meta["stale_ids"], ["b"])
        self.assertEqual(meta["unembedded"], {})
        self.assertFalse(meta["generation_mismatch"])
        self.assertEqual([(it["a"], it["b"]) for it in items], [("a", "b")])

    def test_generation_mismatch_ignores_stored_vectors(self):
        facts = [{"id": "a", "category": "dev", "fact": "alpha"},
                 {"id": "b", "category": "dev", "fact": "beta"}]
        self.seed([(f["id"], "dev", f["fact"], "2026-09-26 10:00:00", 1) for f in facts])
        snapshot = self.snapshot(facts)
        snapshot["generation"] = "other-generation"
        with patch.object(jev_dedupe, "load_vectors",
                          return_value={"a": vec(1.0, 0.0), "b": vec(0.99, 0.1)}) as load:
            items, meta = jev_dedupe.find_candidates(snapshot)
        load.assert_not_called()
        self.embed.assert_called_once()
        self.assertTrue(meta["generation_mismatch"])
        self.assertEqual(meta["unembedded"], {"dev": ["a", "b"]})
        self.assertEqual(items, [])

    def test_internal_error_keeps_earlier_pairs_unscored(self):
        aaa = [{"id": "a1", "category": "aaa", "fact": "one"},
               {"id": "a2", "category": "aaa", "fact": "two"}]
        zzz = [{"id": "z1", "category": "zzz", "fact": "one"},
               {"id": "z2", "category": "zzz", "fact": "two"}]
        self.seed([(f["id"], f["category"], f["fact"], "2026-09-26 10:00:00", 1)
                   for f in aaa + zzz])
        with schema.db_session(self.db) as conn, conn:
            generation = schema.get_db_generation(conn)
        snapshot = {"generation": generation,
                    "revisions": {f["id"]: 1 for f in aaa + zzz},
                    "categories": {"aaa": aaa, "zzz": zzz}}
        with patch.object(jev_dedupe, "embed_missing",
                          side_effect=[{"a1": vec(1.0, 0.0), "a2": vec(0.99, 0.1)},
                                       RuntimeError("boom")]):
            items, meta = jev_dedupe.find_candidates(snapshot)
        self.assertEqual([(it["a"], it["b"]) for it in items], [("a1", "a2")],
                         "pairs generated before the failure are kept, unscored")
        self.assertIsNone(items[0]["duplicate"])
        self.assertIsNone(items[0]["conflict"])
        self.assertFalse(items[0]["below_floor"])
        self.assertEqual(meta["jev"], "error")
        self.assertEqual(meta["error"], "RuntimeError")
        self.assertEqual(meta["pairs_considered"], 1)
        self.assertEqual(meta["above_floor"], 0)
        self.assertEqual(meta["unscored"], 1)
        self.assertEqual(meta["below_floor"], 0)


class CliTests(JevedupeTestBase):
    """Spec 5.9: consolidate CLI wiring around --candidates / --since-epoch."""

    def setUp(self):
        super().setUp()
        self.home = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.db = str(self.home / "memory.db")
        self.stack.enter_context(patch.dict(os.environ, {"HOME": str(self.home)}))
        self.stack.enter_context(patch.object(schema, "DB_PATH", self.db))
        self.stack.enter_context(patch.object(memory.tempfile, "gettempdir", return_value=str(self.home)))
        self.stack.enter_context(patch.object(memory, "CACHE_PATH", str(self.home / "model.txt")))
        self.stack.enter_context(patch.object(memory, "_infer_json", side_effect=AssertionError("no LLM")))
        self.stack.enter_context(patch.object(memory.subprocess, "run", side_effect=AssertionError("no LLM")))
        with schema.db_session():
            pass
        memory._VOCABULARY_CACHE.clear()
        self.addCleanup(schema._SCHEMA_INITIALIZED.discard, self.db)
        memory.upsert_fact("a", "infra", "server A runs nginx", "nginx")
        memory.upsert_fact("b", "infra", "server A uses nginx", "nginx web")
        memory.upsert_fact("solo", "software", "only fact in its category")

    @staticmethod
    def fake_candidates():
        items = []
        for i in range(70):
            items.append({"category": "infra", "a": "a%d" % i, "b": "b%d" % i,
                          "similarity": 0.9, "duplicate": 0.9, "conflict": 0.1,
                          "below_floor": False})
        for i in range(5):
            items.append({"category": "infra", "a": "x%d" % i, "b": "y%d" % i,
                          "similarity": 0.2, "duplicate": 0.1, "conflict": 0.1,
                          "below_floor": True})
        meta = {"pairs_considered": 75, "above_floor": 70, "unscored": 0, "below_floor": 5,
                "jev": "disabled", "calls": 0, "floor": 0.5, "excluded_categories": [],
                "excluded_pairs": 0, "unembedded": {}, "stale_ids": [], "since_utc": None,
                "generation": "g", "model_fingerprint": "f", "max_request_bytes": 0,
                "elapsed_s": 0.0}
        return items, meta

    def run_cli(self, argv):
        out = io.StringIO()
        with patch.object(sys, "argv", ["agy_memory.py"] + argv), redirect_stdout(out):
            memory.main()
        return out.getvalue()

    def test_export_with_candidates_writes_file_and_filters_stdout(self):
        items, meta = self.fake_candidates()
        self.stack.enter_context(patch.object(jev_dedupe, "find_candidates",
                                              Mock(return_value=(items, meta))))
        snap_file = self.home / "snapshot.json"
        out = self.run_cli(["consolidate", "--export-file", str(snap_file), "--candidates"])
        payload = json.loads(snap_file.read_text(encoding="utf-8"))
        self.assertEqual(payload["candidates"], items, "the file keeps every generated pair")
        self.assertEqual(payload["candidates_meta"], meta)
        self.assertEqual(set(payload["categories"]), {"infra"},
                         "only categories with 2+ facts are exported")
        self.assertIn("generation", payload)
        self.assertIn("revisions", payload)
        printed = json.loads(out)
        self.assertEqual(printed["candidates_meta"], meta, "degraded states are loud")
        self.assertEqual(len(printed["candidates"]), 60, "at most 60 items")
        for it in printed["candidates"]:
            self.assertEqual(set(it), {"category", "a", "b", "similarity", "duplicate", "conflict"})
        self.assertEqual([it["a"] for it in printed["candidates"]], ["a%d" % i for i in range(60)],
                         "below-floor items are never listed")

    def test_export_without_candidates_is_unchanged(self):
        snap_file = self.home / "snapshot.json"
        out = self.run_cli(["consolidate", "--export-file", str(snap_file)])
        printed = json.loads(out)
        self.assertEqual(set(printed), {"exported", "categories"})
        payload = json.loads(snap_file.read_text(encoding="utf-8"))
        self.assertNotIn("candidates", payload)
        self.assertNotIn("candidates_meta", payload)

    def test_candidates_without_export_file_errors(self):
        with self.assertRaises(SystemExit):
            self.run_cli(["consolidate", "--candidates"])
        with self.assertRaises(SystemExit):
            self.run_cli(["consolidate", "--since-epoch", "5"])

    def test_since_epoch_requires_candidates(self):
        snap_file = self.home / "snapshot.json"
        with self.assertRaises(SystemExit):
            self.run_cli(["consolidate", "--export-file", str(snap_file), "--since-epoch", "5"])

    def test_export_with_candidates_still_applies_as_snapshot_file(self):
        self.stack.enter_context(patch.object(jev_dedupe, "find_candidates",
                                              Mock(return_value=self.fake_candidates())))
        snap_file = self.home / "snapshot.json"
        self.run_cli(["consolidate", "--export-file", str(snap_file), "--candidates"])
        prop_file = self.home / "proposals.json"
        prop_file.write_text(json.dumps({"merges": [{
            "target_id": "a", "category": "infra", "fact": "server A runs nginx",
            "keywords": "nginx web", "merged_ids": ["b"], "rationale": "duplicate"}]}),
            encoding="utf-8")
        out = self.run_cli(["consolidate", "--proposals-file", str(prop_file),
                            "--snapshot-file", str(snap_file)])
        self.assertIn('"applied": false', out)
        with schema.db_session() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0], 3)

    def test_find_candidates_receives_since_epoch(self):
        fake = Mock(return_value=([], {"jev": "disabled"}))
        self.stack.enter_context(patch.object(jev_dedupe, "find_candidates", fake))
        snap_file = self.home / "snapshot.json"
        self.run_cli(["consolidate", "--export-file", str(snap_file),
                      "--candidates", "--since-epoch", "123"])
        fake.assert_called_once()
        args, _kwargs = fake.call_args
        self.assertEqual(args[1], 123)


class ConfigTests(JevedupeTestBase):
    """Spec 5.10: env knobs parse in a fresh interpreter; invalid values disable."""

    def check(self, extra):
        env = dict(os.environ)
        env.pop("AGY_MEMORY_JEV_DEDUPE_EXCLUDE", None)
        env.update(extra)
        expr = ("import config; print(config.JEV_DEDUPE_ENABLED, config.JEV_DEDUPE_TOP_K,"
                " config.JEV_DEDUPE_EXCLUDE is None)")
        return subprocess.run([sys.executable, "-c", expr],
                              capture_output=True, text=True, env=env, timeout=30)

    def test_enable_and_disable(self):
        res = self.check({"AGY_MEMORY_JEV_DEDUPE": "true"})
        self.assertEqual(res.stdout.split()[0], "True")
        self.assertEqual(res.stdout.split()[2], "True", "unset exclude defers to the protected default")
        res = self.check({"AGY_MEMORY_JEV_DEDUPE": "false"})
        self.assertEqual(res.stdout.split()[0], "False")

    def test_invalid_numeric_knob_disables_scoring(self):
        res = self.check({"AGY_MEMORY_JEV_DEDUPE": "true", "AGY_MEMORY_JEV_DEDUPE_TOP_K": "oops"})
        self.assertEqual(res.stdout.split()[0], "False")
        self.assertEqual(res.stdout.split()[1], "5", "fall back to the default value")
        res = self.check({"AGY_MEMORY_JEV_DEDUPE": "true", "AGY_MEMORY_JEV_DEDUPE_FLOOR": "nan"})
        self.assertEqual(res.stdout.split()[0], "False")

    def test_exclude_list_parses(self):
        res = self.check({"AGY_MEMORY_JEV_DEDUPE_EXCLUDE": "Finance, health"})
        self.assertEqual(res.stdout.split()[2], "False")
        expr = "import config; print(sorted(config.JEV_DEDUPE_EXCLUDE))"
        env = dict(os.environ)
        env["AGY_MEMORY_JEV_DEDUPE_EXCLUDE"] = "Finance, health"
        res = subprocess.run([sys.executable, "-c", expr], capture_output=True, text=True,
                             env=env, timeout=30)
        self.assertEqual(res.stdout.strip(), "['finance', 'health']")

    def test_exclude_adds_to_protected_categories(self):
        # R1: a configured list must never replace the protected set.
        env = dict(os.environ)
        env["AGY_MEMORY_JEV_DEDUPE_EXCLUDE"] = "dev"
        expr = "import jev_dedupe; print(','.join(sorted(jev_dedupe.JEV_DEDUPE_EXCLUDE)))"
        res = subprocess.run([sys.executable, "-c", expr], capture_output=True, text=True,
                             env=env, timeout=30)
        self.assertEqual(res.returncode, 0, res.stderr)
        exclude = set(res.stdout.strip().split(","))
        self.assertIn("dev", exclude)
        self.assertIn("health", exclude)
        self.assertIn("finance", exclude)

    def test_non_integral_integer_knob_disables_scoring(self):
        # TOP_K/MAX_PAIRS/PAIRS_PER_CALL/REQUEST_BYTES must be whole numbers;
        # "1.9" is in range for the first three, and 48000.5 is in range for
        # REQUEST_BYTES (its range floor is 4000, so 1.9 would only test that).
        for name, value in (("TOP_K", "1.9"), ("MAX_PAIRS", "1.9"),
                            ("PAIRS_PER_CALL", "1.9"), ("REQUEST_BYTES", "48000.5")):
            res = self.check({"AGY_MEMORY_JEV_DEDUPE": "true",
                              "AGY_MEMORY_JEV_DEDUPE_" + name: value})
            self.assertEqual(res.stdout.split()[0], "False", name)


if __name__ == "__main__":
    unittest.main()
