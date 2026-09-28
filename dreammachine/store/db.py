"""SQLite results store: runs, per-response diagnoses, and JSON artifacts.

Plain stdlib sqlite3 with JSON columns. One file per project (default
runs/dreammachine.db) is easy to copy from the GPU laptop and to serve from the API.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    config TEXT NOT NULL,
    metrics TEXT,
    created_at REAL NOT NULL,
    finished_at REAL
);
CREATE TABLE IF NOT EXISTS responses (
    run_id TEXT NOT NULL REFERENCES runs(id),
    example_id TEXT NOT NULL,
    source TEXT NOT NULL,
    sample INTEGER NOT NULL,
    text TEXT NOT NULL,
    correct INTEGER NOT NULL,
    error_type TEXT NOT NULL,
    diagnosis TEXT NOT NULL,
    features TEXT NOT NULL,
    PRIMARY KEY (run_id, example_id, sample)
);
CREATE TABLE IF NOT EXISTS artifacts (
    run_id TEXT NOT NULL REFERENCES runs(id),
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (run_id, key)
);
CREATE INDEX IF NOT EXISTS idx_resp_run ON responses(run_id);
"""


class Store:
    def __init__(self, path: str | Path = "runs/dreammachine.db") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------ runs
    def create_run(self, name: str, kind: str, config: dict[str, Any]) -> str:
        run_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs (id, name, kind, status, config, created_at) VALUES (?,?,?,?,?,?)",
                (run_id, name, kind, "running", json.dumps(config), time.time()))
            self._conn.commit()
        return run_id

    def finish_run(self, run_id: str, status: str = "done", metrics: dict | None = None) -> None:
        with self._lock:
            self._conn.execute("UPDATE runs SET status=?, metrics=?, finished_at=? WHERE id=?",
                               (status, json.dumps(metrics or {}), time.time(), run_id))
            self._conn.commit()

    @staticmethod
    def _run_row(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["config"] = json.loads(d["config"])
        d["metrics"] = json.loads(d["metrics"]) if d["metrics"] else None
        return d

    def get_run(self, run_id: str) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return self._run_row(r) if r else None

    def list_runs(self, kind: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM runs", ()
        if kind:
            q, args = q + " WHERE kind=?", (kind,)
        with self._lock:
            rows = self._conn.execute(q + " ORDER BY created_at DESC", args).fetchall()
        return [self._run_row(r) for r in rows]

    # ------------------------------------------------------------- responses
    def add_responses(self, run_id: str, records: Iterable) -> None:
        rows = [(run_id, r.example_id, r.source, r.sample, r.text, int(r.correct), r.error_type,
                 json.dumps(r.diagnosis), json.dumps(r.features)) for r in records]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?,?,?,?)", rows)
            self._conn.commit()

    def get_responses(self, run_id: str, limit: int | None = None, offset: int = 0) -> list[dict]:
        q = "SELECT * FROM responses WHERE run_id=? ORDER BY example_id, sample"
        args: tuple = (run_id,)
        if limit is not None:
            q, args = q + " LIMIT ? OFFSET ?", (run_id, limit, offset)
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["correct"] = bool(d["correct"])
            d["diagnosis"] = json.loads(d["diagnosis"])
            d["features"] = json.loads(d["features"])
            out.append(d)
        return out

    # ------------------------------------------------------------- artifacts
    def put_artifact(self, run_id: str, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO artifacts VALUES (?,?,?)",
                               (run_id, key, json.dumps(value)))
            self._conn.commit()

    def get_artifact(self, run_id: str, key: str) -> Any:
        with self._lock:
            r = self._conn.execute("SELECT value FROM artifacts WHERE run_id=? AND key=?",
                                   (run_id, key)).fetchone()
        return json.loads(r["value"]) if r else None

    def list_artifacts(self, run_id: str) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT key FROM artifacts WHERE run_id=? ORDER BY key",
                                      (run_id,)).fetchall()
        return [r["key"] for r in rows]

    def close(self) -> None:
        self._conn.close()
