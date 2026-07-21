"""
SQLite storage for eval run history. This is what makes "diff against the
previous run" possible across separate invocations of the eval script --
without persisted history there's nothing to compare a new run to.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent.parent / "eval_history.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    model TEXT NOT NULL,
    total_cases INTEGER NOT NULL,
    passed INTEGER NOT NULL,
    pass_rate REAL NOT NULL,
    avg_summary_score REAL,
    avg_latency_ms REAL,
    total_tokens INTEGER,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS case_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(run_id),
    case_id TEXT NOT NULL,
    expected_category TEXT NOT NULL,
    actual_category TEXT,
    category_match INTEGER NOT NULL,
    expected_summary TEXT NOT NULL,
    actual_summary TEXT,
    summary_score INTEGER,
    summary_score_reasoning TEXT,
    latency_ms REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    error TEXT,
    expected_difficulty TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_case_results_run_id ON case_results(run_id);
"""


@contextmanager
def get_connection(db_path: Path = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


@dataclass
class RunRecord:
    prompt_version: str
    dataset_version: str
    model: str
    total_cases: int
    passed: int
    pass_rate: float
    avg_summary_score: float | None
    avg_latency_ms: float | None
    total_tokens: int | None
    status: str


@dataclass
class CaseResultRecord:
    case_id: str
    expected_category: str
    actual_category: str | None
    category_match: bool
    expected_summary: str
    actual_summary: str | None
    summary_score: int | None
    summary_score_reasoning: str | None
    latency_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    error: str | None
    expected_difficulty: str


def save_run(conn: sqlite3.Connection, run: RunRecord, case_results: list[CaseResultRecord]) -> int:
    from datetime import datetime, timezone

    cur = conn.execute(
        """INSERT INTO runs
           (created_at, prompt_version, dataset_version, model, total_cases,
            passed, pass_rate, avg_summary_score, avg_latency_ms, total_tokens, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            run.prompt_version,
            run.dataset_version,
            run.model,
            run.total_cases,
            run.passed,
            run.pass_rate,
            run.avg_summary_score,
            run.avg_latency_ms,
            run.total_tokens,
            run.status,
        ),
    )
    run_id = cur.lastrowid

    conn.executemany(
        """INSERT INTO case_results
           (run_id, case_id, expected_category, actual_category, category_match,
            expected_summary, actual_summary, summary_score, summary_score_reasoning,
            latency_ms, prompt_tokens, completion_tokens, error, expected_difficulty)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                run_id,
                cr.case_id,
                cr.expected_category,
                cr.actual_category,
                int(cr.category_match),
                cr.expected_summary,
                cr.actual_summary,
                cr.summary_score,
                cr.summary_score_reasoning,
                cr.latency_ms,
                cr.prompt_tokens,
                cr.completion_tokens,
                cr.error,
                cr.expected_difficulty,
            )
            for cr in case_results
        ],
    )
    return run_id


def get_latest_run_id(conn: sqlite3.Connection, exclude_run_id: int | None = None) -> int | None:
    if exclude_run_id is not None:
        row = conn.execute(
            "SELECT run_id FROM runs WHERE run_id != ? ORDER BY run_id DESC LIMIT 1",
            (exclude_run_id,),
        ).fetchone()
    else:
        row = conn.execute("SELECT run_id FROM runs ORDER BY run_id DESC LIMIT 1").fetchone()
    return row["run_id"] if row else None


def get_case_results(conn: sqlite3.Connection, run_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM case_results WHERE run_id = ? ORDER BY case_id", (run_id,)
    ).fetchall()


def get_run(conn: sqlite3.Connection, run_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()


def get_recent_runs(conn: sqlite3.Connection, limit: int = 10) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM runs ORDER BY run_id DESC LIMIT ?", (limit,)
    ).fetchall()
