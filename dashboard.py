#!/usr/bin/env python3
"""
AGY Memory Engine - Real-Time Debug Dashboard
Zero-dependency, standalone live web interface for inspecting and testing
multi-layer memories, knowledge graph relations, and turn queue state.
"""

import os
import sys
import json
import time
import sqlite3
import secrets
import hmac
import fcntl
import urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import (
    DB_PATH,
    QUEUE_DB_PATH,
    MODEL_NAME,
    DASHBOARD_PORT,
    DASHBOARD_HOST,
    DASHBOARD_TOKEN,
    DASHBOARD_TOKEN_PATH,
    INACTIVITY_THRESHOLD_SECONDS,
    MAX_WAIT_THRESHOLD_SECONDS
)
from schema import db_session
from agy_memory import (
    extract_multilingual_tokens,
    get_all_vocabulary,
    list_snapshots,
    create_snapshot,
    restore_snapshot
)
from queue_manager import (
    get_pending_stats,
    get_pending_turns,
    get_recent_turns,
    prune_processed_turns
)


def get_or_create_dashboard_token(token_path: Optional[str] = None) -> str:
    """Load or create the authentication token for mutating dashboard endpoints."""
    env_token = os.environ.get("AGY_MEMORY_DASHBOARD_TOKEN") or DASHBOARD_TOKEN
    if env_token and env_token.strip():
        return env_token.strip()

    target_path = Path(token_path or os.environ.get("AGY_MEMORY_DASHBOARD_TOKEN_PATH") or DASHBOARD_TOKEN_PATH)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize concurrent first requests and fail closed on persistence errors.
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(str(target_path), flags, 0o600)
    with os.fdopen(fd, "r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        os.fchmod(handle.fileno(), 0o600)
        token = handle.read().strip()
        if not token:
            token = secrets.token_urlsafe(32)
            handle.seek(0)
            handle.write(token)
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())
        return token


def is_local_origin(origin: str) -> bool:
    """Check if an Origin header refers to a local host / loopback address."""
    if not origin:
        return False
    try:
        parsed = urllib.parse.urlparse(origin)
        if parsed.scheme not in ("http", "https"):
            return False
        hostname = (parsed.hostname or "").lower()
        return hostname in ("127.0.0.1", "localhost", "::1")
    except Exception:
        return False



HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AGY Memory Engine • Live Debug Dashboard</title>
  <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🧠</text></svg>">
  <link rel="apple-touch-icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🧠</text></svg>">
  <meta name="apple-mobile-web-app-title" content="AGY Memory">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
  <meta name="theme-color" content="#0d1117">
  <style>
    :root {
      --bg: #0d1117;
      --card-bg: #161b22;
      --card-border: #30363d;
      --text: #c9d1d9;
      --text-bright: #f0f6fc;
      --text-muted: #8b949e;
      --accent: #58a6ff;
      --accent-glow: rgba(88, 166, 255, 0.15);
      --success: #3fb950;
      --warning: #d29922;
      --danger: #f85149;
      --purple: #bc8cff;
      --cyan: #39c5cf;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 20px;
      max-width: 1400px;
      margin: 0 auto;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 15px;
      padding-bottom: 18px;
      border-bottom: 1px solid var(--card-border);
      margin-bottom: 20px;
    }
    .header-title {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .header-title h1 {
      font-size: 1.35rem;
      color: var(--text-bright);
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .pulse-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 3px 9px;
      background: rgba(63, 185, 80, 0.12);
      border: 1px solid rgba(63, 185, 80, 0.35);
      color: var(--success);
      font-size: 0.75rem;
      border-radius: 20px;
      font-weight: 500;
    }
    .pulse-dot {
      width: 7px;
      height: 7px;
      background: var(--success);
      border-radius: 50%;
      box-shadow: 0 0 8px var(--success);
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.8); }
      100% { opacity: 1; transform: scale(1); }
    }
    .header-meta {
      display: flex;
      gap: 10px;
      align-items: center;
      font-size: 0.85rem;
      flex-wrap: wrap;
    }
    .badge {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      padding: 4px 10px;
      border-radius: 6px;
      color: var(--text-muted);
      font-family: monospace;
    }
    .badge b { color: var(--accent); }

    /* Clean 5-Metric Primary Navigation Bar */
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 12px;
      margin-bottom: 20px;
    }
    @media (max-width: 900px) {
      .stats-grid {
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      }
    }
    .stat-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 12px 14px;
      position: relative;
      cursor: pointer;
      user-select: none;
      transition: all 0.15s ease;
      text-align: left;
    }
    .stat-card:hover {
      border-color: #58a6ff99;
      transform: translateY(-2px);
    }
    .stat-card.active {
      border-color: var(--accent);
      background: rgba(88, 166, 255, 0.08);
      box-shadow: 0 0 12px var(--accent-glow);
    }
    .stat-card::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0; height: 3px;
      background: var(--card-border);
      border-radius: 8px 8px 0 0;
    }
    .stat-card.c-facts::before { background: #58a6ff; }
    .stat-card.c-episodes::before { background: var(--purple); }
    .stat-card.c-learnings::before { background: var(--cyan); }
    .stat-card.c-links::before { background: var(--success); }
    .stat-card.c-queue::before { background: var(--warning); }

    .stat-label {
      font-size: 0.75rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.6px;
      font-weight: 600;
      white-space: nowrap;
    }
    .stat-val {
      font-size: 1.7rem;
      font-weight: 700;
      color: var(--text-bright);
      line-height: 1.2;
      margin: 3px 0 2px;
    }
    .stat-layer {
      font-size: 0.75rem;
      color: var(--text-muted);
      white-space: nowrap;
    }

    /* Prominent Search Bar */
    .search-box {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 20px;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      padding: 10px 14px;
      border-radius: 8px;
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    .search-box:focus-within {
      border-color: var(--accent);
      box-shadow: 0 0 12px var(--accent-glow);
    }
    .search-box input {
      flex: 1;
      background: none;
      border: none;
      color: var(--text-bright);
      font-size: 1rem;
      outline: none;
    }
    .search-meta {
      display: flex;
      align-items: center;
      font-size: 0.8rem;
      color: var(--text-muted);
      font-family: monospace;
    }

    .tab-content { display: none; }
    .tab-content.active { display: block; }

    /* Cards & Lists */
    .item-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 15px;
      margin-bottom: 12px;
      transition: border-color 0.2s;
    }
    .item-card:hover { border-color: #58a6ff66; }
    .item-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }
    .item-id {
      font-family: monospace;
      font-weight: 600;
      color: var(--accent);
      font-size: 0.9rem;
    }
    .pill {
      font-size: 0.75rem;
      padding: 2px 8px;
      border-radius: 12px;
      font-weight: 500;
      border: 1px solid var(--card-border);
      background: transparent;
      color: var(--text-muted);
    }
    .pill.active { background: rgba(63, 185, 80, 0.15); color: var(--success); border-color: rgba(63, 185, 80, 0.4); }
    .pill.cooling { background: rgba(210, 153, 34, 0.15); color: var(--warning); border-color: rgba(210, 153, 34, 0.4); }
    .pill.historic { background: rgba(139, 148, 158, 0.15); color: var(--text-muted); border-color: rgba(139, 148, 158, 0.4); }
    .pill.resolved { background: rgba(57, 197, 207, 0.15); color: var(--cyan); border-color: rgba(57, 197, 207, 0.4); }
    .pill.danger { background: rgba(248, 81, 73, 0.15); color: var(--danger); border-color: rgba(248, 81, 73, 0.4); }
    .pill.category { background: rgba(88, 166, 255, 0.1); color: var(--accent); }

    .item-body {
      color: var(--text-bright);
      font-size: 0.95rem;
      margin-bottom: 8px;
      white-space: pre-wrap;
    }
    .item-meta {
      font-size: 0.8rem;
      color: var(--text-muted);
      display: flex;
      gap: 15px;
      flex-wrap: wrap;
    }
    .item-meta span b { color: var(--text); }

    /* Modern Grouped Entity Graph */
    .entity-group-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
      gap: 15px;
    }
    @media (max-width: 600px) {
      .entity-group-grid {
        grid-template-columns: 1fr;
      }
    }
    .entity-subject-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 14px;
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    .entity-subject-card:hover {
      border-color: rgba(63, 185, 80, 0.4);
      box-shadow: 0 0 12px rgba(63, 185, 80, 0.08);
    }
    .entity-subject-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
      padding-bottom: 8px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .entity-subject-title {
      display: flex;
      align-items: center;
      gap: 8px;
      font-family: monospace;
      font-weight: 600;
      color: var(--accent);
      font-size: 0.9rem;
      word-break: break-all;
    }
    .entity-links-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .entity-link-row {
      display: flex;
      align-items: center;
      gap: 10px;
      background: rgba(255,255,255,0.03);
      padding: 7px 10px;
      border-radius: 6px;
      border: 1px solid rgba(255,255,255,0.04);
    }
    .entity-relation-pill {
      font-size: 0.7rem;
      font-family: monospace;
      font-weight: 600;
      background: rgba(63, 185, 80, 0.15);
      color: var(--success);
      border: 1px solid rgba(63, 185, 80, 0.35);
      padding: 2px 7px;
      border-radius: 4px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      white-space: nowrap;
      flex-shrink: 0;
    }
    .entity-arrow {
      color: var(--text-muted);
      font-size: 0.8rem;
      flex-shrink: 0;
    }
    .entity-target-node {
      font-family: monospace;
      font-size: 0.85rem;
      color: var(--text-bright);
      word-break: break-all;
    }

    /* Table View for Graph */
    .relation-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
    }
    .relation-table th {
      text-align: left;
      padding: 10px 14px;
      border-bottom: 2px solid var(--card-border);
      color: var(--text-muted);
      text-transform: uppercase;
      font-size: 0.75rem;
      letter-spacing: 0.5px;
    }
    .relation-table td {
      padding: 10px 14px;
      border-bottom: 1px solid rgba(255,255,255,0.05);
      font-family: monospace;
    }
    .relation-table tr:hover td {
      background: rgba(255,255,255,0.03);
    }

    /* Queue & Debounce */
    .queue-bar-container {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 15px;
      margin-bottom: 20px;
    }
    .queue-bar-header {
      display: flex;
      justify-content: space-between;
      margin-bottom: 8px;
      font-size: 0.85rem;
    }
    .progress-track {
      height: 8px;
      background: rgba(255,255,255,0.1);
      border-radius: 4px;
      overflow: hidden;
      margin-bottom: 10px;
    }
    .progress-fill {
      height: 100%;
      background: var(--warning);
      width: 0%;
      transition: width 0.5s ease;
    }
    .btn {
      background: var(--accent);
      color: #0d1117;
      border: none;
      padding: 6px 14px;
      border-radius: 6px;
      font-weight: 600;
      font-size: 0.85rem;
      cursor: pointer;
      transition: opacity 0.2s;
    }
    .btn:hover { opacity: 0.9; }
    .btn-secondary {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      color: var(--text-bright);
    }
    .btn-danger {
      background: #da3633;
      color: #ffffff;
      border: 1px solid rgba(248, 81, 73, 0.4);
    }
    .btn-danger:hover {
      background: #f85149;
    }

    /* In-App Glassmorphism Modal */
    .modal-backdrop {
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(0, 0, 0, 0.78);
      backdrop-filter: blur(6px);
      -webkit-backdrop-filter: blur(6px);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 9999;
      padding: 16px;
      animation: fadeIn 0.12s ease-out;
    }
    .modal-dialog {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 10px;
      width: 100%;
      max-width: 640px;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.75);
      overflow: hidden;
      animation: modalSlide 0.15s ease-out;
      display: flex;
      flex-direction: column;
      max-height: 88vh;
    }
    .modal-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 14px 18px;
      border-bottom: 1px solid #30363d;
      font-size: 1.05rem;
      font-weight: 600;
      color: var(--text-bright);
    }
    .modal-close-btn {
      background: none;
      border: none;
      color: var(--text-muted);
      font-size: 1.1rem;
      cursor: pointer;
      padding: 4px 8px;
      border-radius: 4px;
      line-height: 1;
    }
    .modal-close-btn:hover {
      color: var(--text-bright);
      background: rgba(255, 255, 255, 0.08);
    }
    .modal-body {
      padding: 18px 20px;
      color: var(--text);
      font-size: 0.9rem;
      line-height: 1.55;
      overflow-y: auto;
    }
    .modal-footer {
      display: flex;
      justify-content: flex-end;
      gap: 10px;
      padding: 12px 18px;
      border-top: 1px solid #30363d;
      background: rgba(0, 0, 0, 0.25);
    }
    .modal-log-box {
      background: #090d13;
      border: 1px solid #30363d;
      border-radius: 6px;
      padding: 12px 14px;
      font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
      font-size: 0.82rem;
      color: #7ee787;
      white-space: pre-wrap;
      word-break: break-all;
      max-height: 300px;
      overflow-y: auto;
      line-height: 1.45;
      margin-top: 10px;
    }
    .modal-checklist {
      list-style: none;
      margin: 10px 0;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .modal-checklist li {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.88rem;
      color: var(--text);
    }

    /* Toast Notifications */
    .toast-container {
      position: fixed;
      bottom: 24px;
      right: 24px;
      z-index: 10000;
      display: flex;
      flex-direction: column;
      gap: 10px;
      pointer-events: none;
      max-width: 420px;
    }
    .toast {
      background: #161b22;
      border: 1px solid #30363d;
      border-left: 4px solid var(--accent);
      color: var(--text-bright);
      padding: 11px 16px;
      border-radius: 6px;
      font-size: 0.88rem;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.55);
      pointer-events: auto;
      animation: toastIn 0.2s ease-out;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .toast.success { border-left-color: var(--success); }
    .toast.error { border-left-color: var(--danger); }
    .toast.warning { border-left-color: var(--warning); }

    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }
    @keyframes modalSlide {
      from { opacity: 0; transform: translateY(-12px) scale(0.98); }
      to { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes toastIn {
      from { opacity: 0; transform: translateY(12px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>

  <header>
    <div class="header-title">
      <h1>🧠 AGY Memory Engine</h1>
      <div class="pulse-badge"><div class="pulse-dot"></div> Live FTS5</div>
    </div>
    <div class="header-meta">
      <div class="badge">Model: <b id="lbl-model">-</b></div>
      <div class="badge">DB: <b id="lbl-db-size">-</b></div>
      <button class="btn btn-secondary" onclick="switchTab('history')">📸 History & Snapshots</button>
      <button class="btn btn-secondary" onclick="fetchData(true)">🔄 Refresh</button>
      <button class="btn btn-secondary" id="btn-optimize" onclick="optimizeDb()" title="Rebuild FTS5 indexes, VACUUM DB, consolidate facts, and age episodes">🧹 Optimize DB</button>
      <button class="btn" onclick="forceWorker()">⚡ Force Queue</button>
    </div>
  </header>

  <!-- 5-Card Navigation Bar -->
  <div class="stats-grid">
    <div class="stat-card c-facts active" id="card-facts" onclick="switchTab('facts')">
      <div class="stat-label">Facts</div>
      <div class="stat-val" id="cnt-facts">0</div>
      <div class="stat-layer">Layer 1 • Master Data</div>
    </div>
    <div class="stat-card c-episodes" id="card-episodes" onclick="switchTab('episodes')">
      <div class="stat-label">Episodes</div>
      <div class="stat-val" id="cnt-episodes">0</div>
      <div class="stat-layer">Layer 2 • Topics</div>
    </div>
    <div class="stat-card c-learnings" id="card-learnings" onclick="switchTab('learnings')">
      <div class="stat-label">Learnings</div>
      <div class="stat-val" id="cnt-learnings">0</div>
      <div class="stat-layer">Layer 3 • Heuristics</div>
    </div>
    <div class="stat-card c-links" id="card-graph" onclick="switchTab('graph')">
      <div class="stat-label">Relations</div>
      <div class="stat-val" id="cnt-links">0</div>
      <div class="stat-layer">Layer 4 • Entity Graph</div>
    </div>
    <div class="stat-card c-queue" id="card-queue" onclick="switchTab('queue')">
      <div class="stat-label">Turn Queue</div>
      <div class="stat-val" id="cnt-queue">0</div>
      <div class="stat-layer" id="lbl-queue-sub">0 pending (idle)</div>
    </div>
  </div>

  <!-- Dedicated Live Search Bar -->
  <div class="search-box">
    <span style="color:var(--text-muted);">🔍</span>
    <input type="text" id="inp-search" placeholder="Type to search across all memory layers (e.g. dog insurance, Tesla, Beelink, Madrid, retirement)..." oninput="debounceSearch()">
    <button id="btn-clear-search" onclick="clearSearch()" style="display:none; background:none; border:none; color:var(--text-muted); cursor:pointer; font-size:1.1rem; padding:0 6px;">✕</button>
    <div class="search-meta" id="search-latency">0 ms</div>
  </div>

  <!-- Live Search Results Container -->
  <div id="search-results" style="display:none; margin-bottom: 25px;"></div>

  <!-- TAB 1: Facts (Default Active) -->
  <div id="tab-facts" class="tab-content active">
    <div id="facts-items"></div>
  </div>

  <!-- TAB 2: Episodes -->
  <div id="tab-episodes" class="tab-content">
    <div id="episodes-items"></div>
  </div>

  <!-- TAB 3: Learnings -->
  <div id="tab-learnings" class="tab-content">
    <div id="learnings-items"></div>
  </div>

  <!-- TAB 4: Relations -->
  <div id="tab-graph" class="tab-content">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px; flex-wrap:wrap; gap:10px;">
      <h3 style="font-size:1.1rem; color:var(--text-bright);">Knowledge Graph Relations (<span id="graph-total-count">0</span>)</h3>
      <div style="display:flex; gap:8px; align-items:center;">
        <input type="text" id="inp-filter-graph" placeholder="Filter relations..." oninput="filterGraphView()" style="background:var(--card-bg); border:1px solid var(--card-border); border-radius:6px; padding:4px 10px; color:var(--text-bright); font-size:0.8rem; width:180px; outline:none;">
        <button class="pill active g-view-btn" style="cursor:pointer;" onclick="setGraphViewMode('grouped', event)">Grouped</button>
        <button class="pill g-view-btn" style="cursor:pointer;" onclick="setGraphViewMode('table', event)">Table</button>
      </div>
    </div>
    <div id="graph-items"></div>
  </div>

  <!-- TAB 5: Queue Monitor -->
  <div id="tab-queue" class="tab-content">
    <div class="queue-bar-container">
      <div class="queue-bar-header">
        <span><b>Calm-Memory Debounce Status</b></span>
        <span id="lbl-debounce-text">Queue is empty</span>
      </div>
      <div class="progress-track">
        <div class="progress-fill" id="progress-debounce"></div>
      </div>
      <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.8rem; color:var(--text-muted);">
        <span>Debounce window: 300s (5m idle) / Max timeout: 900s (15m)</span>
        <button class="btn" style="padding: 4px 10px; font-size: 0.75rem;" onclick="forceWorker()">Process Batch Now</button>
      </div>
    </div>

    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px; flex-wrap:wrap; gap:10px;">
      <h3 style="font-size:1.1rem; color:var(--text-bright);">Conversation Turn Queue Inspector</h3>
      <div class="queue-filters" style="display:flex; gap:6px; align-items:center;">
        <button class="pill active q-filter-btn" style="cursor:pointer;" onclick="filterQueue('all', event)">All</button>
        <button class="pill q-filter-btn" style="cursor:pointer;" onclick="filterQueue('pending', event)">Pending</button>
        <button class="pill q-filter-btn" style="cursor:pointer;" onclick="filterQueue('processed', event)">Processed</button>
        <button class="pill q-filter-btn" style="cursor:pointer;" onclick="filterQueue('skipped', event)">Skipped</button>
        <button class="btn btn-secondary" style="padding: 3px 8px; font-size: 0.7rem; margin-left:6px;" onclick="clearProcessedQueue()" title="Purge processed turns older than 0 days">🗑️ Clear Processed</button>
      </div>
    </div>
    <div id="queue-items"></div>
  </div>

  <!-- TAB 6: History & Snapshots -->
  <div id="tab-history" class="tab-content">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:18px; flex-wrap:wrap; gap:12px;">
      <div>
        <h3 style="font-size:1.15rem; color:var(--text-bright); margin-bottom:4px;">Snapshot History & Maintenance Audit</h3>
        <p style="font-size:0.8rem; color:var(--text-muted);">Inspect automatic optimization snapshots, rollback database to previous versions, and view consolidation logs.</p>
      </div>
      <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
        <button class="pill active h-subtab-btn" id="btn-subtab-snapshots" style="cursor:pointer;" onclick="switchHistorySubTab('snapshots', event)">📸 Snapshots (<span id="cnt-snapshots-tab">0</span>)</button>
        <button class="pill h-subtab-btn" id="btn-subtab-audit" style="cursor:pointer;" onclick="switchHistorySubTab('audit', event)">🧠 Consolidation Audit (<span id="cnt-audit-tab">0</span>)</button>
        <button class="btn btn-secondary" style="font-size:0.75rem; padding:4px 10px;" onclick="createManualSnapshot()" title="Take an immediate backup snapshot of current DB">📸 Take Snapshot</button>
      </div>
    </div>

    <!-- Snapshots Sub-Section -->
    <div id="history-sub-snapshots">
      <div id="snapshot-items"></div>
    </div>

    <!-- Consolidation Audit Log Sub-Section -->
    <div id="history-sub-audit" style="display:none;">
      <div id="audit-items"></div>
    </div>
  </div>

  <!-- Reusable In-App Dark Modal -->
  <div id="modal-backdrop" class="modal-backdrop" style="display:none;" onclick="handleModalBackdropClick(event)">
    <div class="modal-dialog" id="modal-container">
      <div class="modal-header">
        <div style="display:flex; align-items:center; gap:10px;">
          <span id="modal-icon" style="font-size:1.25rem;">✨</span>
          <span id="modal-title">Modal Title</span>
        </div>
        <button class="modal-close-btn" onclick="closeModal()">✕</button>
      </div>
      <div class="modal-body" id="modal-body">
        <!-- Dynamic body content -->
      </div>
      <div class="modal-footer" id="modal-footer">
        <!-- Dynamic footer buttons -->
      </div>
    </div>
  </div>

  <!-- In-App Toast Container -->
  <div id="toast-container" class="toast-container"></div>

  <script>
    const DASHBOARD_TOKEN = {{DASHBOARD_TOKEN}};
    let rawData = null;
    let searchTimer = null;
    const openTurnDetails = new Set();
    let currentQueueFilter = 'all';
    let currentActiveTab = 'facts';
    let currentGraphViewMode = 'grouped';
    let graphFilterQuery = '';

    // State caches to avoid unnecessary DOM re-creation
    let lastRenderedQueueHash = '';
    let lastRenderedFactsHash = '';
    let lastRenderedEpisodesHash = '';
    let lastRenderedLearningsHash = '';
    let lastRenderedGraphHash = '';
    let lastRenderedAuditHash = '';
    let lastRenderedSnapshotsHash = '';
    let currentHistorySubTab = 'snapshots';

    function switchTab(name) {
      if (name === 'audit') {
        name = 'history';
        switchHistorySubTab('audit');
      }
      currentActiveTab = name;
      document.querySelectorAll('.stat-card').forEach(c => c.classList.remove('active'));
      const activeCard = document.getElementById('card-' + name);
      if (activeCard) activeCard.classList.add('active');

      // Clear search when explicitly selecting a layer card
      const inp = document.getElementById('inp-search');
      if (inp) inp.value = '';
      document.getElementById('btn-clear-search').style.display = 'none';
      document.getElementById('search-latency').innerText = '0 ms';
      document.getElementById('search-results').style.display = 'none';

      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      const targetTab = document.getElementById('tab-' + name);
      if (targetTab) targetTab.classList.add('active');
    }

    function switchHistorySubTab(subtab, ev) {
      currentHistorySubTab = subtab;
      document.querySelectorAll('.h-subtab-btn').forEach(b => b.classList.remove('active'));
      const activeBtn = document.getElementById('btn-subtab-' + subtab);
      if (activeBtn) activeBtn.classList.add('active');

      const subSnaps = document.getElementById('history-sub-snapshots');
      const subAudit = document.getElementById('history-sub-audit');
      if (subSnaps) subSnaps.style.display = subtab === 'snapshots' ? 'block' : 'none';
      if (subAudit) subAudit.style.display = subtab === 'audit' ? 'block' : 'none';
    }

    function clearSearch() {
      const inp = document.getElementById('inp-search');
      if (inp) inp.value = '';
      document.getElementById('btn-clear-search').style.display = 'none';
      document.getElementById('search-latency').innerText = '0 ms';
      document.getElementById('search-results').style.display = 'none';
      switchTab(currentActiveTab);
    }

    function onTurnDetailToggle(turnId, isOpen) {
      if (isOpen) {
        openTurnDetails.add(turnId);
      } else {
        openTurnDetails.delete(turnId);
      }
    }

    function filterQueue(status, ev) {
      currentQueueFilter = status;
      document.querySelectorAll('.q-filter-btn').forEach(b => b.classList.remove('active'));
      if (ev && ev.target) ev.target.classList.add('active');
      lastRenderedQueueHash = ''; // force re-render
      if (rawData && rawData.queue) {
        renderQueue(rawData.queue, true);
      }
    }

    function setGraphViewMode(mode, ev) {
      currentGraphViewMode = mode;
      document.querySelectorAll('.g-view-btn').forEach(b => b.classList.remove('active'));
      if (ev && ev.target) ev.target.classList.add('active');
      lastRenderedGraphHash = '';
      if (rawData && rawData.links) renderGraph(rawData.links, true);
    }

    function filterGraphView() {
      const inp = document.getElementById('inp-filter-graph');
      graphFilterQuery = (inp ? inp.value : '').trim().toLowerCase();
      lastRenderedGraphHash = '';
      if (rawData && rawData.links) renderGraph(rawData.links, true);
    }

    async function fetchData(forceDomRefresh = false) {
      try {
        const res = await fetch('/api/stats');
        const data = await res.json();
        rawData = data;

        document.getElementById('lbl-model').innerText = data.model || 'gemini-3.7-flash-low';
        document.getElementById('lbl-db-size').innerText = data.db_size || '-';
        document.getElementById('cnt-facts').innerText = data.counts.facts || 0;
        document.getElementById('cnt-episodes').innerText = data.counts.episodes || 0;
        document.getElementById('cnt-learnings').innerText = data.counts.learnings || 0;
        document.getElementById('cnt-links').innerText = data.counts.links || 0;
        const qCnt = data.counts.queue_pending || 0;
        document.getElementById('cnt-queue').innerText = qCnt;
        document.getElementById('lbl-queue-sub').innerText = qCnt === 0 ? '0 pending (idle)' : `${qCnt} waiting for debounce`;

        renderQueue(data.queue, forceDomRefresh);
        renderFacts(data.facts, forceDomRefresh);
        renderEpisodes(data.episodes, forceDomRefresh);
        renderLearnings(data.learnings, forceDomRefresh);
        renderGraph(data.links, forceDomRefresh);
        renderSnapshots(data.snapshots, forceDomRefresh);
        renderAudit(data.audit, forceDomRefresh);

        const snCnt = (data.snapshots || []).length;
        const auCnt = (data.audit || []).length;
        const cntSnTab = document.getElementById('cnt-snapshots-tab');
        if (cntSnTab) cntSnTab.innerText = snCnt;
        const cntAuTab = document.getElementById('cnt-audit-tab');
        if (cntAuTab) cntAuTab.innerText = auCnt;
      } catch (e) {
        console.error("Failed to load dashboard data", e);
      }
    }

    function renderQueue(queue, force = false) {
      const qCont = document.getElementById('queue-items');
      const debLbl = document.getElementById('lbl-debounce-text');
      const prog = document.getElementById('progress-debounce');

      if (!queue) return;

      const idleSec = queue.newest_age_seconds || 0;
      const pct = Math.min(100, Math.round((idleSec / 300) * 100));
      prog.style.width = pct + '%';
      debLbl.innerText = queue.pending_count > 0
        ? `Last message ${idleSec}s ago (batch triggers at 300s idle or 15m timeout)`
        : 'Queue is idle (0 pending turns)';

      const allTurns = queue.turns || [];
      const filteredTurns = currentQueueFilter === 'all'
        ? allTurns
        : allTurns.filter(t => t.status === currentQueueFilter);

      const currentHash = currentQueueFilter + '::' + JSON.stringify(filteredTurns);
      if (!force && currentHash === lastRenderedQueueHash) {
        return; // Skip rebuilding DOM if data did not change
      }
      lastRenderedQueueHash = currentHash;

      if (filteredTurns.length === 0) {
        qCont.innerHTML = `<p style="color:var(--text-muted); padding:30px; text-align:center;">No ${currentQueueFilter !== 'all' ? currentQueueFilter : ''} turns found in queue.</p>`;
        return;
      }

      // Group turns:
      // - 'pending' turns are rendered individually / as pending queue entries
      // - turns sharing the same 'batch_id' are grouped into a consolidated batch card
      // - legacy turns without 'batch_id' are rendered as standalone items
      const groups = [];
      const batchMap = new Map();

      for (const t of filteredTurns) {
        if (t.status === 'pending') {
          groups.push({ type: 'pending', turn: t });
        } else if (t.batch_id) {
          if (!batchMap.has(t.batch_id)) {
            const batchGroup = {
              type: 'batch',
              batch_id: t.batch_id,
              status: t.status,
              summary: t.extracted_summary,
              error: t.error,
              processed_at: t.processed_at || t.created_at,
              turns: []
            };
            batchMap.set(t.batch_id, batchGroup);
            groups.push(batchGroup);
          }
          batchMap.get(t.batch_id).turns.push(t);
        } else {
          groups.push({ type: 'legacy', turn: t });
        }
      }

      qCont.innerHTML = groups.map(g => {
        if (g.type === 'pending') {
          const t = g.turn;
          const isOpen = openTurnDetails.has(t.id);
          return `
            <div class="item-card" style="border-left: 3px solid var(--warning); margin-bottom:14px;">
              <div class="item-header">
                <span class="item-id">Turn #${escapeHtml(t.id)} <span style="font-weight:normal; color:var(--text-muted)">[${escapeHtml(t.source || 'telegram')}]</span></span>
                <div style="display:flex; gap:8px; align-items:center;">
                  <span class="pill cooling">PENDING</span>
                  <span style="font-size:0.75rem; color:var(--text-muted);">${escapeHtml(t.created_at)}</span>
                </div>
              </div>
              
              <div style="background:rgba(88, 166, 255, 0.08); border-radius:6px; padding:10px 12px; margin-bottom:10px; border-left:3px solid var(--accent);">
                <div style="font-size:0.75rem; text-transform:uppercase; color:var(--accent); font-weight:600; margin-bottom:4px;">👤 User Prompt</div>
                <div style="color:var(--text-bright); font-size:0.9rem; white-space:pre-wrap;">${escapeHtml(t.user_prompt)}</div>
              </div>

              <details ${isOpen ? 'open' : ''} ontoggle="onTurnDetailToggle(parseInt(this.dataset.turnId, 10), this.open)" data-turn-id="${escapeHtml(t.id)}" style="background:rgba(255, 255, 255, 0.03); border-radius:6px; padding:8px 12px; margin-bottom:10px;">
                <summary style="cursor:pointer; font-size:0.8rem; color:var(--text-muted); font-weight:500;">
                  🤖 Assistant Response (${escapeHtml(String((t.assistant_response || '').length))} chars) — Click to view
                </summary>
                <div style="color:var(--text); font-size:0.85rem; margin-top:8px; white-space:pre-wrap; max-height:280px; overflow-y:auto; border-top:1px solid rgba(255,255,255,0.06); padding-top:8px;">
                  ${escapeHtml(t.assistant_response || '')}
                </div>
              </details>

              <div class="item-meta">
                <span><b>Chat ID:</b> ${escapeHtml(t.chat_id || '-')}</span>
                <span><b>Status:</b> Waiting for calm-memory debounce</span>
              </div>
            </div>
          `;
        }

        if (g.type === 'batch') {
          const b = g;
          let statusClass = 'historic';
          let borderCol = 'var(--card-border)';
          if (b.status === 'processed') { statusClass = 'active'; borderCol = 'var(--success)'; }
          if (b.status === 'failed') { statusClass = 'danger'; borderCol = 'var(--danger)'; }

          const turnIdsStr = b.turns.map(t => '#' + escapeHtml(t.id)).join(', ');

          return `
            <div class="item-card" style="border-left: 4px solid ${borderCol}; margin-bottom:18px; background:rgba(22, 27, 34, 0.95); padding:16px;">
              <div class="item-header" style="padding-bottom:10px; border-bottom:1px solid rgba(255,255,255,0.06); margin-bottom:12px;">
                <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
                  <span style="font-weight:700; color:var(--text-bright); font-size:0.95rem; display:flex; align-items:center; gap:6px;">
                    📦 Batch <span style="color:var(--accent); font-family:monospace;">${escapeHtml(b.batch_id)}</span>
                  </span>
                  <span class="pill" style="background:rgba(88, 166, 255, 0.12); color:var(--accent); border-color:rgba(88, 166, 255, 0.3);">
                    ${escapeHtml(String(b.turns.length))} turn${b.turns.length > 1 ? 's' : ''} processed together (${turnIdsStr})
                  </span>
                </div>
                <div style="display:flex; gap:8px; align-items:center;">
                  <span class="pill ${escapeHtml(statusClass)}">${escapeHtml(String(b.status).toUpperCase())}</span>
                  <span style="font-size:0.75rem; color:var(--text-muted); font-family:monospace;">${escapeHtml(b.processed_at)}</span>
                </div>
              </div>

              ${b.summary ? `
                <div style="background:rgba(63, 185, 80, 0.12); border:1px solid rgba(63, 185, 80, 0.25); border-radius:6px; padding:9px 12px; margin-bottom:12px; font-size:0.85rem; color:var(--success);">
                  🧠 <b>Extracted Knowledge (from this batch):</b> ${escapeHtml(b.summary)}
                </div>
              ` : ''}

              ${b.error ? `
                <div style="background:rgba(248, 81, 73, 0.12); border:1px solid rgba(248, 81, 73, 0.25); border-radius:6px; padding:9px 12px; margin-bottom:12px; font-size:0.85rem; color:var(--danger);">
                  ⚠️ <b>Error:</b> ${escapeHtml(b.error)}
                </div>
              ` : ''}

              <div style="display:flex; flex-direction:column; gap:10px;">
                ${b.turns.map(t => {
                  const isOpen = openTurnDetails.has(t.id);
                  return `
                    <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); border-radius:6px; padding:10px 12px;">
                      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <span style="font-family:monospace; font-weight:600; color:var(--accent); font-size:0.85rem;">
                          Turn #${escapeHtml(t.id)} <span style="font-weight:normal; color:var(--text-muted)">[${escapeHtml(t.source || 'telegram')}]</span>
                        </span>
                        <span style="font-size:0.75rem; color:var(--text-muted);">${escapeHtml(t.created_at)}</span>
                      </div>

                      <div style="background:rgba(88, 166, 255, 0.06); border-radius:5px; padding:8px 10px; margin-bottom:6px; border-left:3px solid var(--accent);">
                        <div style="font-size:0.7rem; text-transform:uppercase; color:var(--accent); font-weight:600; margin-bottom:2px;">👤 User Prompt</div>
                        <div style="color:var(--text-bright); font-size:0.85rem; white-space:pre-wrap;">${escapeHtml(t.user_prompt)}</div>
                      </div>

                      <details ${isOpen ? 'open' : ''} ontoggle="onTurnDetailToggle(parseInt(this.dataset.turnId, 10), this.open)" data-turn-id="${escapeHtml(t.id)}" style="background:rgba(255, 255, 255, 0.02); border-radius:5px; padding:6px 10px;">
                        <summary style="cursor:pointer; font-size:0.75rem; color:var(--text-muted); font-weight:500;">
                          🤖 Assistant Response (${escapeHtml(String((t.assistant_response || '').length))} chars) — Click to view
                        </summary>
                        <div style="color:var(--text); font-size:0.8rem; margin-top:6px; white-space:pre-wrap; max-height:240px; overflow-y:auto; border-top:1px solid rgba(255,255,255,0.06); padding-top:6px;">
                          ${escapeHtml(t.assistant_response || '')}
                        </div>
                      </details>
                    </div>
                  `;
                }).join('')}
              </div>
            </div>
          `;
        }

        // Legacy unbatched turn
        const t = g.turn;
        let statusClass = 'historic';
        if (t.status === 'processed') statusClass = 'active';
        if (t.status === 'failed') statusClass = 'danger';
        const isOpen = openTurnDetails.has(t.id);

        return `
          <div class="item-card" style="border-left: 3px solid ${t.status === 'processed' ? 'var(--success)' : 'var(--card-border)'}; margin-bottom:14px;">
            <div class="item-header">
              <span class="item-id">Turn #${escapeHtml(t.id)} <span style="font-weight:normal; color:var(--text-muted)">[${escapeHtml(t.source || 'telegram')}]</span></span>
              <div style="display:flex; gap:8px; align-items:center;">
                <span class="pill ${escapeHtml(statusClass)}">${escapeHtml(String(t.status).toUpperCase())}</span>
                <span style="font-size:0.75rem; color:var(--text-muted);">${escapeHtml(t.created_at)}</span>
              </div>
            </div>
            
            <div style="background:rgba(88, 166, 255, 0.08); border-radius:6px; padding:10px 12px; margin-bottom:10px; border-left:3px solid var(--accent);">
              <div style="font-size:0.75rem; text-transform:uppercase; color:var(--accent); font-weight:600; margin-bottom:4px;">👤 User Prompt</div>
              <div style="color:var(--text-bright); font-size:0.9rem; white-space:pre-wrap;">${escapeHtml(t.user_prompt)}</div>
            </div>

            <details ${isOpen ? 'open' : ''} ontoggle="onTurnDetailToggle(parseInt(this.dataset.turnId, 10), this.open)" data-turn-id="${escapeHtml(t.id)}" style="background:rgba(255, 255, 255, 0.03); border-radius:6px; padding:8px 12px; margin-bottom:10px;">
              <summary style="cursor:pointer; font-size:0.8rem; color:var(--text-muted); font-weight:500;">
                🤖 Assistant Response (${escapeHtml(String((t.assistant_response || '').length))} chars) — Click to view
              </summary>
              <div style="color:var(--text); font-size:0.85rem; margin-top:8px; white-space:pre-wrap; max-height:280px; overflow-y:auto; border-top:1px solid rgba(255,255,255,0.06); padding-top:8px;">
                ${escapeHtml(t.assistant_response || '')}
              </div>
            </details>

            ${t.extracted_summary ? `
              <div style="background:rgba(63, 185, 80, 0.1); border-radius:6px; padding:6px 10px; margin-bottom:8px; font-size:0.8rem; color:var(--success);">
                🧠 <b>Extracted Knowledge:</b> ${escapeHtml(t.extracted_summary)}
              </div>
            ` : ''}

            ${t.error ? `
              <div style="background:rgba(248, 81, 73, 0.1); border-radius:6px; padding:6px 10px; margin-bottom:8px; font-size:0.8rem; color:var(--danger);">
                ⚠️ <b>Error:</b> ${escapeHtml(t.error)}
              </div>
            ` : ''}

            <div class="item-meta">
              <span><b>Chat ID:</b> ${escapeHtml(t.chat_id || '-')}</span>
              <span><b>Status:</b> ${escapeHtml(t.status)}</span>
            </div>
          </div>
        `;
      }).join('');
    }

    function renderFacts(facts, force = false) {
      const cont = document.getElementById('facts-items');
      const hash = JSON.stringify(facts || []);
      if (!force && hash === lastRenderedFactsHash) return;
      lastRenderedFactsHash = hash;

      if (!facts || facts.length === 0) {
        cont.innerHTML = '<p style="color:var(--text-muted); padding:20px; text-align:center;">No facts stored.</p>';
        return;
      }
      cont.innerHTML = facts.map(f => `
        <div class="item-card">
          <div class="item-header">
            <span class="item-id">${escapeHtml(f.id)}</span>
            <span class="pill category">${escapeHtml(f.category)}</span>
          </div>
          <div class="item-body">${escapeHtml(f.fact)}</div>
          <div class="item-meta">
            <span><b>Keywords:</b> ${escapeHtml(f.keywords || '-')}</span>
            <span><b>Updated:</b> ${escapeHtml(f.updated_at)}</span>
          </div>
        </div>
      `).join('');
    }

    function renderEpisodes(episodes, force = false) {
      const cont = document.getElementById('episodes-items');
      const hash = JSON.stringify(episodes || []);
      if (!force && hash === lastRenderedEpisodesHash) return;
      lastRenderedEpisodesHash = hash;

      if (!episodes || episodes.length === 0) {
        cont.innerHTML = '<p style="color:var(--text-muted); padding:20px; text-align:center;">No episodes stored.</p>';
        return;
      }
      cont.innerHTML = episodes.map(e => `
        <div class="item-card">
          <div class="item-header">
            <span class="item-id">${escapeHtml(e.title || e.id)} <span style="font-weight:normal; color:var(--text-muted)">(${escapeHtml(e.topic)})</span></span>
            <span class="pill ${escapeHtml(e.status)}">${escapeHtml(e.status)}</span>
          </div>
          <div class="item-body">${escapeHtml(e.narrative)}</div>
          <div class="item-meta">
            <span><b>Period:</b> ${escapeHtml(e.period || '-')}</span>
            <span><b>Entities:</b> ${escapeHtml(e.entities || '-')}</span>
            <span><b>Stance / Sentiment:</b> ${escapeHtml(e.stance || '-')}</span>
          </div>
        </div>
      `).join('');
    }

    function renderLearnings(learnings, force = false) {
      const cont = document.getElementById('learnings-items');
      const hash = JSON.stringify(learnings || []);
      if (!force && hash === lastRenderedLearningsHash) return;
      lastRenderedLearningsHash = hash;

      if (!learnings || learnings.length === 0) {
        cont.innerHTML = '<p style="color:var(--text-muted); padding:20px; text-align:center;">No learnings stored.</p>';
        return;
      }
      cont.innerHTML = learnings.map(l => `
        <div class="item-card">
          <div class="item-header">
            <span class="item-id">${escapeHtml(l.id)}</span>
            <span class="pill category">${escapeHtml(l.category)}</span>
          </div>
          <div class="item-body">💡 ${escapeHtml(l.insight)}</div>
          <div class="item-meta">
            <span><b>Context:</b> ${escapeHtml(l.context || '-')}</span>
            <span><b>Keywords:</b> ${escapeHtml(l.keywords || '-')}</span>
          </div>
        </div>
      `).join('');
    }

    function renderGraph(links, force = false) {
      const cont = document.getElementById('graph-items');
      const totalCntLbl = document.getElementById('graph-total-count');
      if (totalCntLbl) totalCntLbl.innerText = (links || []).length;

      if (!links || links.length === 0) {
        cont.innerHTML = '<p style="color:var(--text-muted); padding:30px; text-align:center;">No entity relationships stored.</p>';
        return;
      }

      let filtered = links;
      if (graphFilterQuery) {
        filtered = links.filter(l => 
          l.source_id.toLowerCase().includes(graphFilterQuery) ||
          l.relation.toLowerCase().includes(graphFilterQuery) ||
          l.target_id.toLowerCase().includes(graphFilterQuery)
        );
      }

      const hash = currentGraphViewMode + '::' + graphFilterQuery + '::' + JSON.stringify(filtered);
      if (!force && hash === lastRenderedGraphHash) return;
      lastRenderedGraphHash = hash;

      if (filtered.length === 0) {
        cont.innerHTML = `<p style="color:var(--text-muted); padding:30px; text-align:center;">No relations matching "${escapeHtml(graphFilterQuery)}".</p>`;
        return;
      }

      if (currentGraphViewMode === 'table') {
        cont.innerHTML = `
          <div style="background:var(--card-bg); border:1px solid var(--card-border); border-radius:8px; overflow:hidden;">
            <table class="relation-table">
              <thead>
                <tr>
                  <th>Source Entity</th>
                  <th style="text-align:center;">Relation</th>
                  <th>Target Entity</th>
                </tr>
              </thead>
              <tbody>
                ${filtered.map(l => `
                  <tr>
                    <td style="color:var(--accent); font-weight:600;">${escapeHtml(l.source_id)}</td>
                    <td style="text-align:center;">
                      <span class="entity-relation-pill">${escapeHtml(l.relation)}</span>
                    </td>
                    <td style="color:var(--text-bright);">${escapeHtml(l.target_id)}</td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
        `;
        return;
      }

      // Grouped View (Default)
      const grouped = {};
      for (const l of filtered) {
        if (!grouped[l.source_id]) grouped[l.source_id] = [];
        grouped[l.source_id].push(l);
      }

      const sources = Object.keys(grouped).sort();

      cont.innerHTML = `
        <div class="entity-group-grid">
          ${sources.map(src => `
            <div class="entity-subject-card">
              <div class="entity-subject-header">
                <div class="entity-subject-title">
                  <span style="color:var(--success);">🔗</span>
                  <span>${escapeHtml(src)}</span>
                </div>
                <span class="pill active">${grouped[src].length}</span>
              </div>
              <div class="entity-links-list">
                ${grouped[src].map(l => `
                  <div class="entity-link-row">
                    <span class="entity-relation-pill">${escapeHtml(l.relation)}</span>
                    <span class="entity-arrow">➔</span>
                    <span class="entity-target-node">${escapeHtml(l.target_id)}</span>
                  </div>
                `).join('')}
              </div>
            </div>
          `).join('')}
        </div>
      `;
    }

    function renderSnapshots(snapshots, force = false) {
      const cont = document.getElementById('snapshot-items');
      const tabCnt = document.getElementById('cnt-snapshots-tab');
      if (tabCnt) tabCnt.innerText = (snapshots || []).length;

      const hash = JSON.stringify(snapshots || []);
      if (!force && hash === lastRenderedSnapshotsHash) return;
      lastRenderedSnapshotsHash = hash;

      if (!snapshots || snapshots.length === 0) {
        cont.innerHTML = '<p style="color:var(--text-muted); padding:30px; text-align:center;">No backup snapshots found in ~/.gemini/archive.</p>';
        return;
      }

      cont.innerHTML = snapshots.map((s, idx) => {
        const st = s.stats || {};
        let tagClass = 'category';
        if (s.tag === 'manual') tagClass = 'active';
        if (s.tag === 'pre-restore') tagClass = 'cooling';

        return `
          <div class="item-card" style="border-left: 3px solid ${idx === 0 ? 'var(--accent)' : 'var(--card-border)'}; margin-bottom:12px;">
            <div class="item-header">
              <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
                <span class="item-id" style="font-size:0.95rem; color:var(--text-bright);">
                  📸 ${escapeHtml(s.filename)}
                </span>
                <span class="pill ${escapeHtml(tagClass)}">${escapeHtml(String(s.tag).toUpperCase())}</span>
                ${idx === 0 ? '<span class="pill active" style="font-size:0.7rem;">LATEST</span>' : ''}
              </div>
              <div style="display:flex; gap:10px; align-items:center;">
                <span style="font-size:0.8rem; color:var(--text-muted); font-family:monospace;">${escapeHtml(s.created_at)}</span>
                <span class="badge" style="font-size:0.75rem; padding:2px 8px;">${escapeHtml(s.size_kb)}</span>
                <button class="btn btn-secondary" style="font-size:0.75rem; padding:3px 10px; border-color:rgba(248,81,73,0.4); color:var(--danger);" onclick="handleRestoreSnapshotClick(${idx})" title="Rollback database to this snapshot">
                  ⏮️ Restore
                </button>
              </div>
            </div>
            
            <div style="display:flex; gap:12px; margin-top:8px; font-size:0.8rem; color:var(--text-muted); flex-wrap:wrap;">
              <span>🧱 <b>${escapeHtml(String(st.facts || 0))}</b> Facts</span>
              <span>📖 <b>${escapeHtml(String(st.episodes || 0))}</b> Episodes</span>
              <span>💡 <b>${escapeHtml(String(st.learnings || 0))}</b> Learnings</span>
              <span>🔗 <b>${escapeHtml(String(st.links || 0))}</b> Links</span>
            </div>
          </div>
        `;
      }).join('');
    }

    function handleRestoreSnapshotClick(idx) {
      const s = (rawData && rawData.snapshots) ? rawData.snapshots[idx] : null;
      if (!s) return;
      const st = s.stats || {};
      const statsSummary = `${st.facts || 0} Facts, ${st.episodes || 0} Episodes, ${st.learnings || 0} Learnings, ${st.links || 0} Links`;
      promptRestoreSnapshot(s.filename, s.created_at, statsSummary);
    }

    function renderAudit(audit, force = false) {
      const cont = document.getElementById('audit-items');
      const tabCnt = document.getElementById('cnt-audit-tab');
      if (tabCnt) tabCnt.innerText = (audit || []).length;

      const hash = JSON.stringify(audit || []);
      if (!force && hash === lastRenderedAuditHash) return;
      lastRenderedAuditHash = hash;

      if (!audit || audit.length === 0) {
        cont.innerHTML = '<p style="color:var(--text-muted); padding:20px; text-align:center;">No consolidation logs recorded.</p>';
        return;
      }
      cont.innerHTML = audit.map(a => `
        <div class="item-card">
          <div class="item-header">
            <span class="item-id">${escapeHtml(String(a.action).toUpperCase())}: ${escapeHtml(a.target_id)}</span>
            <span class="pill category">${escapeHtml(a.category)}</span>
          </div>
          <div class="item-body"><b>Change:</b> ${escapeHtml(a.diff_summary)}</div>
          <div class="item-meta">
            <span><b>Rationale:</b> ${escapeHtml(a.rationale)}</span>
            <span><b>Timestamp:</b> ${escapeHtml(a.timestamp)}</span>
          </div>
        </div>
      `).join('');
    }

    // Modal & Toast Controller
    function closeModal() {
      const backdrop = document.getElementById('modal-backdrop');
      if (backdrop) backdrop.style.display = 'none';
    }

    function handleModalBackdropClick(e) {
      if (e.target && e.target.id === 'modal-backdrop') {
        closeModal();
      }
    }

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        closeModal();
      }
    });

    function showConfirmModal({ title, icon = '❓', message, bulletPoints = [], confirmText = 'Confirm', confirmStyle = 'btn', onConfirm }) {
      document.getElementById('modal-icon').innerText = icon;
      document.getElementById('modal-title').innerText = title;

      let bodyHtml = `<p style="margin-bottom:10px; color:var(--text-bright); font-size:0.92rem;">${escapeHtml(message)}</p>`;
      if (bulletPoints && bulletPoints.length > 0) {
        bodyHtml += `<ul class="modal-checklist">` + bulletPoints.map(p => `<li><span>${escapeHtml(p)}</span></li>`).join('') + `</ul>`;
      }
      document.getElementById('modal-body').innerHTML = bodyHtml;


      const footer = document.getElementById('modal-footer');
      footer.innerHTML = `
        <button class="btn btn-secondary" onclick="closeModal()">Cancel</button>
        <button class="btn ${confirmStyle}" id="modal-btn-confirm">${escapeHtml(confirmText)}</button>
      `;

      document.getElementById('modal-btn-confirm').onclick = async () => {
        closeModal();
        if (onConfirm) await onConfirm();
      };

      document.getElementById('modal-backdrop').style.display = 'flex';
    }

    function showResultModal({ title, icon = 'ℹ️', message, logOutput = '', isError = false }) {
      document.getElementById('modal-icon').innerText = icon;
      document.getElementById('modal-title').innerText = title;

      let bodyHtml = `<p style="margin-bottom:8px; color:${isError ? 'var(--danger)' : 'var(--text-bright)'}; font-size:0.92rem;">${escapeHtml(message)}</p>`;
      if (logOutput) {
        bodyHtml += `<div class="modal-log-box" id="modal-log-content">${escapeHtml(logOutput)}</div>`;
      }
      document.getElementById('modal-body').innerHTML = bodyHtml;

      const footer = document.getElementById('modal-footer');
      footer.innerHTML = `
        ${logOutput ? `<button class="btn btn-secondary" style="margin-right:auto;" onclick="copyModalLog()">📋 Copy Report</button>` : ''}
        <button class="btn ${isError ? 'btn-secondary' : ''}" onclick="closeModal()">Done</button>
      `;

      document.getElementById('modal-backdrop').style.display = 'flex';
    }

    function copyModalLog() {
      const logBox = document.getElementById('modal-log-content');
      if (logBox) {
        navigator.clipboard.writeText(logBox.innerText);
        showToast('Report copied to clipboard', 'success', 2500);
      }
    }

    function showToast(message, type = 'info', duration = 3500) {
      const container = document.getElementById('toast-container');
      if (!container) return;

      const icons = { info: 'ℹ️', success: '✅', error: '⚠️', warning: '⚡' };
      const toast = document.createElement('div');
      toast.className = `toast ${type}`;
      toast.innerHTML = `
        <span style="font-size:1.1rem;">${icons[type] || 'ℹ️'}</span>
        <span style="flex:1;">${escapeHtml(message)}</span>
      `;
      container.appendChild(toast);

      setTimeout(() => {
        toast.style.transition = 'opacity 0.25s, transform 0.25s';
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(8px)';
        setTimeout(() => toast.remove(), 250);
      }, duration);
    }

    function debounceSearch() {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(runSearch, 150);
    }

    async function runSearch() {
      const q = document.getElementById('inp-search').value.trim();
      const cont = document.getElementById('search-results');
      const latLbl = document.getElementById('search-latency');
      const clearBtn = document.getElementById('btn-clear-search');

      if (!q) {
        clearBtn.style.display = 'none';
        cont.style.display = 'none';
        latLbl.innerText = '0 ms';
        // Restore active tab
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        const target = document.getElementById('tab-' + currentActiveTab);
        if (target) target.classList.add('active');
        const activeCard = document.getElementById('card-' + currentActiveTab);
        if (activeCard) activeCard.classList.add('active');
        return;
      }

      clearBtn.style.display = 'block';
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      document.querySelectorAll('.stat-card').forEach(c => c.classList.remove('active'));
      cont.style.display = 'block';

      const t0 = performance.now();
      try {
        const res = await fetch('/api/search?q=' + encodeURIComponent(q));
        const data = await res.json();
        const t1 = performance.now();
        latLbl.innerText = `${(t1 - t0).toFixed(1)} ms (${data.tokens ? data.tokens.join(', ') : ''})`;

        let html = '';
        if (data.facts && data.facts.length > 0) {
          html += '<h4 style="color:var(--accent); margin:15px 0 10px;">🧱 Matches in Facts (' + escapeHtml(String(data.facts.length)) + ')</h4>' + data.facts.map(f => `
            <div class="item-card">
              <div class="item-header"><span class="item-id">${escapeHtml(f.id)}</span><span class="pill category">${escapeHtml(f.category)}</span></div>
              <div class="item-body">${escapeHtml(f.fact)}</div>
            </div>
          `).join('');
        }
        if (data.episodes && data.episodes.length > 0) {
          html += '<h4 style="color:var(--purple); margin:15px 0 10px;">📖 Matches in Episodes (' + escapeHtml(String(data.episodes.length)) + ')</h4>' + data.episodes.map(e => `
            <div class="item-card">
              <div class="item-header"><span class="item-id">${escapeHtml(e.title || e.id)}</span><span class="pill ${escapeHtml(e.status)}">${escapeHtml(e.status)}</span></div>
              <div class="item-body">${escapeHtml(e.narrative)}</div>
            </div>
          `).join('');
        }
        if (data.learnings && data.learnings.length > 0) {
          html += '<h4 style="color:var(--cyan); margin:15px 0 10px;">💡 Matches in Learnings (' + escapeHtml(String(data.learnings.length)) + ')</h4>' + data.learnings.map(l => `
            <div class="item-card">
              <div class="item-header"><span class="item-id">${escapeHtml(l.id)}</span><span class="pill category">${escapeHtml(l.category)}</span></div>
              <div class="item-body">${escapeHtml(l.insight)}</div>
            </div>
          `).join('');
        }

        if (!html) {
          html = '<p style="color:var(--text-muted); text-align:center; padding: 40px;">No matches found for "' + escapeHtml(q) + '"</p>';
        }
        cont.innerHTML = html;
      } catch (e) {
        cont.innerHTML = '<p style="color:var(--danger); text-align:center; padding: 20px;">Search request failed</p>';
      }
    }

    function forceWorker() {
      showConfirmModal({
        title: '⚡ Process Pending Conversation Queue',
        icon: '⚡',
        message: 'Immediately trigger memory_worker.py to batch-process all pending conversation turns without waiting for the 300s calm-memory debounce window.',
        bulletPoints: [
          '⚡ Groups pending turns into a shared batch',
          '🧠 Runs Gemini LLM multi-layer extraction',
          '📨 Sends Telegram batch summary notification'
        ],
        confirmText: 'Process Batch Now',
        onConfirm: async () => {
          showToast('Processing queue turns...', 'info', 3000);
          try {
            const res = await fetch('/api/force-worker', {
              method: 'POST',
              headers: { 'X-Dashboard-Token': DASHBOARD_TOKEN }
            });
            const data = await res.json();
            if (data.status === 'ok') {
              showResultModal({
                title: '✅ Queue Processed',
                icon: '⚡',
                message: 'Conversation queue was batch-processed successfully.',
                logOutput: data.message
              });
            } else {
              showResultModal({
                title: '⚠️ Worker Error',
                icon: '❌',
                message: 'Worker encountered an issue while processing.',
                logOutput: data.message || data.error,
                isError: true
              });
            }
            lastRenderedQueueHash = '';
            fetchData(true);
          } catch (e) {
            showToast('Worker execution failed: ' + e, 'error', 4000);
          }
        }
      });
    }

    function clearProcessedQueue() {
      showConfirmModal({
        title: '🗑️ Purge Processed Conversation Turns',
        icon: '🗑️',
        message: 'Permanently remove all processed and skipped conversation turns from the queue database (~/.gemini/turn_queue.db).',
        bulletPoints: [
          '🧹 Removes processed and skipped entries',
          '🛡️ Leaves pending queue turns untouched',
          '🗜️ Optimizes queue database storage'
        ],
        confirmText: 'Purge Processed',
        confirmStyle: 'btn-danger',
        onConfirm: async () => {
          try {
            const res = await fetch('/api/clear-processed-queue', {
              method: 'POST',
              headers: { 'X-Dashboard-Token': DASHBOARD_TOKEN }
            });
            const data = await res.json();
            showToast(data.message || 'Processed turns cleared from queue.', 'success', 3500);
            lastRenderedQueueHash = '';
            fetchData(true);
          } catch (e) {
            showToast('Error clearing queue: ' + e, 'error', 4000);
          }
        }
      });
    }

    function optimizeDb() {
      showConfirmModal({
        title: '🧹 Run Full Memory Optimization',
        icon: '🧹',
        message: 'Execute complete cognitive memory engine maintenance, semantic deduplication, and database compacting.',
        bulletPoints: [
          '📸 Create snapshot backup in ~/.gemini/archive (with 20-snapshot retention)',
          '⏳ Episode state decay (active ➔ cooling ➔ historic)',
          '🧠 Semantic LLM fact deduplication & consolidation',
          '🗑️ Automatic queue pruning (> 7 days retention)',
          '🔍 Rebuild all SQLite FTS5 full-text search indexes',
          '🗜️ Execute SQLite VACUUM database compaction'
        ],
        confirmText: 'Start Optimization',
        onConfirm: async () => {
          const btn = document.getElementById('btn-optimize');
          const origText = btn ? btn.innerText : '🧹 Optimize DB';
          if (btn) {
            btn.disabled = true;
            btn.innerText = '⏳ Optimizing...';
          }
          showToast('Database optimization started...', 'info', 4000);

          try {
            const res = await fetch('/api/optimize', {
              method: 'POST',
              headers: { 'X-Dashboard-Token': DASHBOARD_TOKEN }
            });
            const data = await res.json();
            if (data.status === 'ok') {
              showResultModal({
                title: '✅ Optimization Completed',
                icon: '🧹',
                message: 'All memory layers and FTS5 indexes were optimized successfully.',
                logOutput: data.message
              });
              showToast('Memory database optimized successfully!', 'success', 3500);
            } else {
              showResultModal({
                title: '⚠️ Optimization Failed',
                icon: '❌',
                message: 'Optimization encountered an error.',
                logOutput: data.message || data.error,
                isError: true
              });
            }
            lastRenderedFactsHash = '';
            lastRenderedEpisodesHash = '';
            lastRenderedLearningsHash = '';
            lastRenderedGraphHash = '';
            lastRenderedAuditHash = '';
            lastRenderedSnapshotsHash = '';
            lastRenderedQueueHash = '';
            fetchData(true);
          } catch (e) {
            showToast('Optimization failed: ' + e, 'error', 4000);
          } finally {
            if (btn) {
              btn.disabled = false;
              btn.innerText = origText;
            }
          }
        }
      });
    }

    function promptRestoreSnapshot(filename, createdAt, statsSummary) {
      showConfirmModal({
        title: '⚠️ Rollback Memory Database',
        icon: '⏮️',
        message: `Are you sure you want to rollback the memory engine to snapshot "${filename}"?`,
        bulletPoints: [
          `📅 Snapshot Date: ${createdAt}`,
          `📊 Snapshot Content: ${statsSummary}`,
          '🛡️ A safety backup of your CURRENT state will be taken automatically before restoring',
          '🔄 FTS5 indexes and VACUUM will be run automatically after restore'
        ],
        confirmText: 'Confirm Rollback',
        confirmStyle: 'btn-danger',
        onConfirm: async () => {
          showToast(`Restoring snapshot ${filename}...`, 'info', 4000);
          try {
            const res = await fetch('/api/restore-snapshot', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'X-Dashboard-Token': DASHBOARD_TOKEN
              },
              body: JSON.stringify({ filename })
            });
            const data = await res.json();
            if (data.status === 'ok') {
              const st = data.counts || {};
              showResultModal({
                title: '✅ Database Restored',
                icon: '⏮️',
                message: data.message,
                logOutput: `[RESTORE] Successfully rolled back to: ${data.restored_snapshot}\n[SAFETY] Safety backup of previous state: ${data.safety_backup}\n[STATS] Active DB State: ${st.facts || 0} Facts, ${st.episodes || 0} Episodes, ${st.learnings || 0} Learnings, ${st.links || 0} Links (Size: ${data.db_size})`
              });
              showToast('Database rolled back successfully!', 'success', 3500);
            } else {
              showResultModal({
                title: '⚠️ Restore Failed',
                icon: '❌',
                message: data.message || data.error,
                isError: true
              });
            }
            lastRenderedFactsHash = '';
            lastRenderedEpisodesHash = '';
            lastRenderedLearningsHash = '';
            lastRenderedGraphHash = '';
            lastRenderedAuditHash = '';
            lastRenderedSnapshotsHash = '';
            lastRenderedQueueHash = '';
            fetchData(true);
          } catch (e) {
            showToast('Restore request failed: ' + e, 'error', 4000);
          }
        }
      });
    }

    async function createManualSnapshot() {
      showToast('Creating manual snapshot...', 'info', 2000);
      try {
        const res = await fetch('/api/create-snapshot', {
          method: 'POST',
          headers: { 'X-Dashboard-Token': DASHBOARD_TOKEN }
        });
        const data = await res.json();
        if (data.status === 'ok') {
          showToast(`📸 Snapshot created: ${data.filename}`, 'success', 3500);
          lastRenderedSnapshotsHash = '';
          fetchData(true);
        } else {
          showToast(`Snapshot error: ${data.message}`, 'error', 4000);
        }
      } catch (e) {
        showToast(`Failed to create snapshot: ${e}`, 'error', 4000);
      }
    }

    function escapeHtml(text) {
      if (text === null || text === undefined) return '';
      return String(text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }

    // Auto-refresh every 3 seconds for real-time responsiveness
    fetchData();
    setInterval(fetchData, 3000);
  </script>
</body>
</html>
"""


class MemoryDashboardHandler(BaseHTTPRequestHandler):
    """Zero-dependency HTTP request handler for AGY Memory Engine Debug Dashboard."""

    def log_message(self, format, *args):
        # Suppress noisy standard request logging
        pass

    def _trusted_request(self):
        """Reject foreign Host headers (including DNS rebinding) and origins."""
        host = self.headers.get("Host", "")
        parsed = urllib.parse.urlparse("http://" + host)
        allowed = {"localhost", "127.0.0.1", "::1"}
        if DASHBOARD_HOST not in ("0.0.0.0", "::"):
            allowed.add(DASHBOARD_HOST.lower())
        if (parsed.hostname or "").lower() not in allowed:
            return False
        origin = self.headers.get("Origin")
        if origin:
            parsed_origin = urllib.parse.urlparse(origin)
            return parsed_origin.scheme in ("http", "https") and parsed_origin.netloc.lower() == host.lower()
        return True

    def _set_cors_headers(self):
        origin = self.headers.get("Origin")
        if origin and is_local_origin(origin) and self._trusted_request():
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Credentials", "true")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Dashboard-Token")
            self.send_header("Vary", "Origin")

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = 200):
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Length", str(len(body)))
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _verify_auth(self) -> bool:
        expected = get_or_create_dashboard_token()
        if not expected:
            return False

        # Header: X-Dashboard-Token
        token = self.headers.get("X-Dashboard-Token")

        # Header: Authorization: Bearer <token>
        if not token:
            auth = self.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                token = auth[7:].strip()

        # Query param: ?token=<token>
        if not token:
            url = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(url.query)
            token_list = params.get("token")
            if token_list:
                token = token_list[0].strip()

        if token and hmac.compare_digest(token.encode("utf-8"), expected.encode("utf-8")):
            return True

        return False

    def do_OPTIONS(self):
        if not self._trusted_request():
            self._send_json({"error": "Untrusted request origin or host"}, status=403)
            return
        origin = self.headers.get("Origin")
        if origin and not is_local_origin(origin):
            self.send_response(403)
            self.end_headers()
            return
        self.send_response(204)
        self._set_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if not self._trusted_request():
            self._send_json({"error": "Untrusted request origin or host"}, status=403)
            return
        url = urllib.parse.urlparse(self.path)
        path = url.path
        params = urllib.parse.parse_qs(url.query)

        if path == "/" or path == "/index.html":
            token = get_or_create_dashboard_token()
            rendered = HTML_TEMPLATE.replace("{{DASHBOARD_TOKEN}}", json.dumps(token).replace("<", "\\u003c"))
            self._send_html(rendered)
            return

        if path == "/favicon.ico":
            svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">🧠</text></svg>'
            body = svg.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "image/svg+xml")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "public, max-age=86400")
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/api/stats":
            self._handle_stats()
            return

        if path == "/api/search":
            q = params.get("q", [""])[0]
            self._handle_search(q)
            return

        self._send_json({"error": "Not Found"}, status=404)

    def do_POST(self):
        if not self._trusted_request():
            self._send_json({"error": "Untrusted request origin or host"}, status=403)
            return
        if not self._verify_auth():
            self._send_json(
                {"status": "error", "error": "Unauthorized", "message": "Missing or invalid dashboard token."},
                status=401
            )
            return

        url = urllib.parse.urlparse(self.path)
        if url.path == "/api/force-worker":
            import subprocess
            try:
                worker_bin = BASE_DIR / "memory_worker.py"
                res = subprocess.run([sys.executable, str(worker_bin), "--force"], capture_output=True, text=True, timeout=60)
                self._send_json({"status": "ok", "message": res.stdout.strip() or "Queue processed successfully."})
            except Exception as e:
                self._send_json({"status": "error", "message": str(e)}, status=500)
            return

        if url.path == "/api/clear-processed-queue":
            try:
                prune_processed_turns(days=0)
                self._send_json({"status": "ok", "message": "All processed and skipped turns have been purged from the queue."})
            except Exception as e:
                self._send_json({"status": "error", "message": str(e)}, status=500)
            return

        if url.path == "/api/optimize":
            import subprocess
            try:
                main_bin = BASE_DIR / "agy_memory.py"
                res = subprocess.run(
                    [sys.executable, str(main_bin), "optimize", "--apply"],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                output = (res.stdout or "").strip()
                err = (res.stderr or "").strip()
                if res.returncode == 0:
                    self._send_json({"status": "ok", "message": output or "Database optimization completed successfully."})
                else:
                    self._send_json({"status": "error", "message": err or output or f"Process exited with code {res.returncode}"}, status=500)
            except Exception as e:
                self._send_json({"status": "error", "message": str(e)}, status=500)
            return

        if url.path == "/api/create-snapshot":
            try:
                res = create_snapshot(tag="manual")
                self._send_json(res)
            except Exception as e:
                self._send_json({"status": "error", "message": str(e)}, status=500)
            return

        if url.path == "/api/restore-snapshot":
            try:
                length = int(self.headers.get("Content-Length", 0))
                req_data = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
                filename = req_data.get("filename")
                if not filename:
                    self._send_json({"status": "error", "message": "Missing 'filename' in request."}, status=400)
                    return
                res = restore_snapshot(filename)
                self._send_json(res)
            except Exception as e:
                self._send_json({"status": "error", "message": str(e)}, status=500)
            return

        self._send_json({"error": "Not Found"}, status=404)

    def _handle_stats(self):
        try:
            with db_session() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM memories")
                cnt_facts = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM episodes")
                cnt_episodes = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM learnings")
                cnt_learnings = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM entity_links")
                cnt_links = cursor.fetchone()[0]

                # Fetch all facts (convert UTC timestamps to local system timezone)
                cursor.execute("SELECT id, category, fact, keywords, datetime(updated_at, 'localtime') FROM memories ORDER BY updated_at DESC")
                facts = [{"id": r[0], "category": r[1], "fact": r[2], "keywords": r[3], "updated_at": str(r[4])} for r in cursor.fetchall()]

                # Fetch all episodes (convert UTC timestamps to local system timezone)
                cursor.execute("SELECT id, topic, title, period, status, narrative, entities, stance, datetime(updated_at, 'localtime') FROM episodes ORDER BY updated_at DESC")
                episodes = [{"id": r[0], "topic": r[1], "title": r[2], "period": r[3], "status": r[4], "narrative": r[5], "entities": r[6], "stance": r[7], "updated_at": str(r[8])} for r in cursor.fetchall()]

                # Fetch all learnings (convert UTC timestamps to local system timezone)
                cursor.execute("SELECT id, category, insight, context, keywords, datetime(updated_at, 'localtime') FROM learnings ORDER BY updated_at DESC")
                learnings = [{"id": r[0], "category": r[1], "insight": r[2], "context": r[3], "keywords": r[4], "updated_at": str(r[5])} for r in cursor.fetchall()]

                # Fetch all links
                cursor.execute("SELECT source_id, target_id, relation FROM entity_links ORDER BY source_id")
                links = [{"source_id": r[0], "target_id": r[1], "relation": r[2]} for r in cursor.fetchall()]

                # Fetch audit log (convert UTC timestamps to local system timezone)
                cursor.execute("SELECT action, category, target_id, diff_summary, rationale, datetime(timestamp, 'localtime') FROM consolidation_log ORDER BY id DESC LIMIT 50")
                audit = [{"action": r[0], "category": r[1], "target_id": r[2], "diff_summary": r[3], "rationale": r[4], "timestamp": str(r[5])} for r in cursor.fetchall()]

            # Queue stats
            q_stats = get_pending_stats()
            q_turns = get_recent_turns(limit=50)

            # Snapshots
            snapshots = list_snapshots()

            # DB file size
            db_size = "-"
            if os.path.exists(DB_PATH):
                sz = os.path.getsize(DB_PATH)
                db_size = f"{sz / 1024:.1f} KB" if sz < 1024 * 1024 else f"{sz / (1024*1024):.2f} MB"

            self._send_json({
                "model": MODEL_NAME,
                "db_path": DB_PATH,
                "db_size": db_size,
                "counts": {
                    "facts": cnt_facts,
                    "episodes": cnt_episodes,
                    "learnings": cnt_learnings,
                    "links": cnt_links,
                    "queue_pending": q_stats.get("count", 0)
                },
                "queue": {
                    "pending_count": q_stats.get("count", 0),
                    "newest_age_seconds": q_stats.get("newest_age_seconds", 0),
                    "oldest_age_seconds": q_stats.get("oldest_age_seconds", 0),
                    "turns": q_turns
                },
                "facts": facts,
                "episodes": episodes,
                "learnings": learnings,
                "links": links,
                "snapshots": snapshots,
                "audit": audit
            })
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    def _handle_search(self, q: str):
        if not q.strip():
            self._send_json({"facts": [], "episodes": [], "learnings": [], "tokens": []})
            return

        try:
            with db_session() as conn:
                cursor = conn.cursor()
                vocab = get_all_vocabulary(cursor)
                words = extract_multilingual_tokens(q, vocab)
                if not words:
                    self._send_json({"facts": [], "episodes": [], "learnings": [], "tokens": []})
                    return

                fts_terms = [f'"{w}"*' if len(w) >= 4 else f'"{w}"' for w in words]
                fts_query = " OR ".join(fts_terms)

                # Query Facts
                cursor.execute("""
                    SELECT m.id, m.category, m.fact 
                    FROM memories m
                    JOIN memories_fts f ON m.id = f.id
                    WHERE memories_fts MATCH ?
                    ORDER BY f.rank LIMIT 10;
                """, (fts_query,))
                facts = [{"id": r[0], "category": r[1], "fact": r[2]} for r in cursor.fetchall()]

                # Query Episodes
                cursor.execute("""
                    SELECT e.id, e.topic, e.title, e.period, e.status, e.narrative 
                    FROM episodes e
                    JOIN episodes_fts f ON e.id = f.id
                    WHERE episodes_fts MATCH ?
                    ORDER BY f.rank LIMIT 10;
                """, (fts_query,))
                episodes = [{"id": r[0], "topic": r[1], "title": r[2], "period": r[3], "status": r[4], "narrative": r[5]} for r in cursor.fetchall()]

                # Query Learnings
                cursor.execute("""
                    SELECT l.id, l.category, l.insight 
                    FROM learnings l
                    JOIN learnings_fts f ON l.id = f.id
                    WHERE learnings_fts MATCH ?
                    ORDER BY f.rank LIMIT 10;
                """, (fts_query,))
                learnings = [{"id": r[0], "category": r[1], "insight": r[2]} for r in cursor.fetchall()]

                self._send_json({
                    "tokens": words,
                    "facts": facts,
                    "episodes": episodes,
                    "learnings": learnings
                })
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)


def run_dashboard(host: str = DASHBOARD_HOST, port: int = DASHBOARD_PORT):
    """Start the multi-threaded HTTP server."""
    server = ThreadingHTTPServer((host, port), MemoryDashboardHandler)
    print(f"🚀 AGY Memory Debug Dashboard running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped.")
        server.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AGY Memory Real-Time Debug Dashboard")
    parser.add_argument("--port", type=int, default=DASHBOARD_PORT, help=f"Server port (default: {DASHBOARD_PORT})")
    parser.add_argument("--host", default=DASHBOARD_HOST, help=f"Server host (default: {DASHBOARD_HOST})")
    args = parser.parse_args()
    run_dashboard(host=args.host, port=args.port)
