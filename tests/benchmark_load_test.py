#!/usr/bin/env python3
"""Hermetic real-engine retrieval benchmark; no production database or HTTP calls."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_environment
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
import agy_memory as memory
import schema


def benchmark(size=1000, requests=200, seed=1729):
    randomizer = random.Random(seed)
    for index in range(size):
        memory.upsert_fact(f'host.{index}', 'infra', f'PostgreSQL server region{index % 20} IP SQL configuration', 'database backup')
    queries = [f'PostgreSQL region{randomizer.randrange(20)}' for _ in range(requests)]
    def search(query):
        start = time.perf_counter()
        try:
            result = memory.prefetch(query, quiet=True)
            if not result.get('facts'):
                raise AssertionError('Expected ranked facts')
            return time.perf_counter() - start, None
        except Exception as error:
            return time.perf_counter() - start, type(error).__name__
    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(search, queries))
    latencies = sorted(elapsed * 1000 for elapsed, error in outcomes)
    errors = [error for elapsed, error in outcomes if error]
    with schema.db_session() as conn:
        base_count = conn.execute('SELECT count(*) FROM memories').fetchone()[0]
        fts_count = conn.execute('SELECT count(*) FROM memories_fts').fetchone()[0]
    parity = base_count == fts_count == size
    report = {'seed': seed, 'corpus_size': size, 'attempted': requests,
              'failed': len(errors), 'errors': errors, 'content_count_parity': parity,
              **{f'p{p}_ms': latencies[min(len(latencies)-1, int(len(latencies)*p/100))] for p in (50,95,99)}}
    print(json.dumps(report, indent=2))
    return not errors and parity


if __name__ == '__main__':
    sys.exit(0 if benchmark() else 1)
