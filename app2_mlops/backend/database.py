"""
database.py — SQLite database layer for the Self-Healing MLOps Engine.

- All 13 tables per Part 5 schema (execution-verified SQL).
- PRAGMA journal_mode = WAL so dashboard reads aren't blocked by writes.
- Shared asyncio.Lock for write serialization (Fix #12 concurrency note).
- Seeds 18 Q-table rows and the 'external' sentinel in model_registry.
"""

import sqlite3
import asyncio
import json
from datetime import datetime
from pathlib import Path

import sys, os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.config import DB_PATH, DATA_DIR


# ──────────────────────────────────────────────
# Shared write lock — ALL DB writes go through this
# ──────────────────────────────────────────────
db_write_lock = asyncio.Lock()


def get_connection() -> sqlite3.Connection:
    """Return a new SQLite connection with WAL mode and foreign keys enabled."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


# ──────────────────────────────────────────────
# Schema creation (all 13 tables, Part 5)
# ──────────────────────────────────────────────
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS model_registry (
    version TEXT PRIMARY KEY,
    accuracy REAL,
    f1_score REAL,
    adversarial_accuracy REAL,
    training_samples INTEGER,
    retrain_strategy TEXT,
    retrain_reason TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    parent_version TEXT,
    FOREIGN KEY (parent_version) REFERENCES model_registry(version)
);

CREATE TABLE IF NOT EXISTS prediction_log (
    request_id TEXT PRIMARY KEY,
    received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    features TEXT NOT NULL,
    prediction INTEGER,
    model_version TEXT,
    true_label INTEGER,
    label_received_at TIMESTAMP,
    source TEXT NOT NULL DEFAULT 'predict',
    FOREIGN KEY (model_version) REFERENCES model_registry(version)
);

CREATE TABLE IF NOT EXISTS drift_event (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    drift_score REAL,
    drift_level TEXT,
    drifted_features TEXT,
    p_values TEXT,
    model_version TEXT,
    FOREIGN KEY (model_version) REFERENCES model_registry(version)
);

CREATE TABLE IF NOT EXISTS adversarial_result (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    clean_accuracy REAL,
    adversarial_accuracy REAL,
    epsilon REAL,
    status TEXT,
    drift_event_id INTEGER,
    FOREIGN KEY (drift_event_id) REFERENCES drift_event(event_id)
);

CREATE TABLE IF NOT EXISTS rl_decision (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    decided_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    drift_level INTEGER,
    adversarial_status INTEGER,
    time_since_retrain INTEGER,
    action INTEGER,
    action_name TEXT,
    reward REAL,
    epsilon_at_decision REAL,
    drift_event_id INTEGER,
    FOREIGN KEY (drift_event_id) REFERENCES drift_event(event_id)
);

CREATE TABLE IF NOT EXISTS retrain_event (
    retrain_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    strategy TEXT,
    training_samples INTEGER,
    new_samples INTEGER,
    new_model_version TEXT,
    old_model_version TEXT,
    decision_id INTEGER,
    FOREIGN KEY (decision_id) REFERENCES rl_decision(decision_id)
);

CREATE TABLE IF NOT EXISTS validation_result (
    validation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    gate1_pass BOOLEAN, gate1_pvalue REAL,
    gate2_pass BOOLEAN, gate2_new_f1 REAL, gate2_old_f1 REAL,
    gate3_pass BOOLEAN, gate3_new_adv REAL, gate3_old_adv REAL,
    gate4_pass BOOLEAN, gate4_new_f1 REAL, gate4_old_f1 REAL,
    all_passed BOOLEAN,
    retrain_id INTEGER,
    FOREIGN KEY (retrain_id) REFERENCES retrain_event(retrain_id)
);

CREATE TABLE IF NOT EXISTS human_action (
    action_id INTEGER PRIMARY KEY AUTOINCREMENT,
    acted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    action TEXT,
    reason TEXT,
    is_auto_deploy BOOLEAN DEFAULT 0,
    confidence_visits INTEGER,
    validation_id INTEGER,
    model_version TEXT,
    FOREIGN KEY (validation_id) REFERENCES validation_result(validation_id)
);

CREATE TABLE IF NOT EXISTS event_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    event_type TEXT NOT NULL,
    message TEXT NOT NULL,
    model_version TEXT
);

CREATE TABLE IF NOT EXISTS qtable_state (
    state_id INTEGER PRIMARY KEY AUTOINCREMENT,
    drift_level INTEGER NOT NULL,
    adversarial_status INTEGER NOT NULL,
    time_since_retrain INTEGER NOT NULL,
    q_action_0 REAL DEFAULT 0.0,
    q_action_1 REAL DEFAULT 0.0,
    q_action_2 REAL DEFAULT 0.0,
    q_action_3 REAL DEFAULT 0.0,
    q_action_4 REAL DEFAULT 0.0,
    q_action_5 REAL DEFAULT 0.0,
    synthetic_visits INTEGER DEFAULT 0,
    real_visits INTEGER DEFAULT 0,
    last_updated TIMESTAMP,
    UNIQUE(drift_level, adversarial_status, time_since_retrain)
);

CREATE TABLE IF NOT EXISTS accuracy_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    accuracy REAL,
    model_version TEXT,
    timestep INTEGER
);

CREATE TABLE IF NOT EXISTS drift_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    drift_score REAL,
    timestep INTEGER
);

CREATE TABLE IF NOT EXISTS real_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    state_drift_level INTEGER,
    state_adversarial INTEGER,
    state_time_since_retrain INTEGER,
    action INTEGER,
    reward REAL,
    next_state_drift_level INTEGER,
    next_state_adversarial INTEGER,
    next_state_time_since_retrain INTEGER,
    used_in_replay BOOLEAN DEFAULT 0
);
"""


def _seed_qtable(conn: sqlite3.Connection):
    """Seed all 18 Q-table states so /api/rl-state works before pretraining."""
    for drift_level in range(3):        # 0=low, 1=medium, 2=high
        for adv_status in range(2):     # 0=safe, 1=vulnerable
            for time_bin in range(3):   # 0=recent, 1=moderate, 2=stale
                conn.execute(
                    """INSERT OR IGNORE INTO qtable_state
                       (drift_level, adversarial_status, time_since_retrain,
                        q_action_0, q_action_1, q_action_2,
                        q_action_3, q_action_4, q_action_5,
                        synthetic_visits, real_visits, last_updated)
                       VALUES (?, ?, ?, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, ?)""",
                    (drift_level, adv_status, time_bin, datetime.utcnow().isoformat()),
                )


def _seed_model_registry(conn: sqlite3.Connection):
    """Seed the 'external' sentinel row for /api/observe (Fix #18)."""
    conn.execute(
        """INSERT OR IGNORE INTO model_registry (version, status, retrain_reason)
           VALUES ('external', 'external',
                   'Sentinel row for /api/observe-sourced predictions (Fix #18)')"""
    )


def init_db():
    """Create all tables, seed required rows. Safe to call multiple times."""
    conn = get_connection()
    try:
        conn.executescript(_SCHEMA_SQL)
        _seed_qtable(conn)
        _seed_model_registry(conn)
        conn.commit()
    finally:
        conn.close()


# ──────────────────────────────────────────────
# Helper: log an event to event_log
# ──────────────────────────────────────────────
async def log_event(event_type: str, message: str, model_version: str = None):
    """Thread-safe event logging behind the shared write lock."""
    async with db_write_lock:
        conn = get_connection()
        try:
            conn.execute(
                "INSERT INTO event_log (event_type, message, model_version) VALUES (?, ?, ?)",
                (event_type, message, model_version),
            )
            conn.commit()
        finally:
            conn.close()


# ──────────────────────────────────────────────
# Verification entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    conn = get_connection()

    # Verify tables
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()]
    print(f"✓ Created {len(tables)} tables: {tables}")

    # Verify Q-table seed
    q_count = conn.execute("SELECT COUNT(*) FROM qtable_state").fetchone()[0]
    print(f"✓ Q-table seeded with {q_count} rows (expected 18)")

    # Verify external sentinel
    ext = conn.execute(
        "SELECT version, status FROM model_registry WHERE version='external'"
    ).fetchone()
    print(f"✓ External sentinel: version='{ext[0]}', status='{ext[1]}'")

    conn.close()
    print("\n✓ Database initialization complete:", DB_PATH)
