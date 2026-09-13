"""Isolate import-time defaults for unittest discovery and standalone test runs."""
import atexit
import os
import tempfile

_sandbox = tempfile.TemporaryDirectory(prefix='agy-memory-tests-')
_values = {
    'HOME': _sandbox.name,
    'AGY_MEMORY_DB': os.path.join(_sandbox.name, 'memory.db'),
    'AGY_TURN_QUEUE_DB': os.path.join(_sandbox.name, 'queue.db'),
    'AGY_MEMORY_CACHE': os.path.join(_sandbox.name, 'model.txt'),
    'AGY_MEMORY_DASHBOARD_TOKEN_PATH': os.path.join(_sandbox.name, 'dashboard.token'),
    'AGY_MEMORY_DASHBOARD_HOST': '127.0.0.1',
    'AGY_MEMORY_DASHBOARD_ALLOW_PRIVATE_NETWORKS': '',
    'AGY_MEMORY_DASHBOARD_ALLOWED_HOSTS': '',
    'AGY_MEMORY_STRICT_GRAPH': 'true',
}
_original = {key: os.environ.get(key) for key in _values}
os.environ.update(_values)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
config._FILE_VARS.clear()
config.DASHBOARD_ALLOW_PRIVATE_NETWORKS = None
config.DASHBOARD_ALLOWED_HOSTS = []
config.STRICT_GRAPH = True

def _cleanup():
    for key, value in _original.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    _sandbox.cleanup()


atexit.register(_cleanup)
