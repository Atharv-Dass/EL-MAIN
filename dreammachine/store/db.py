"""SQLite results store: runs, per-response diagnoses, JSON artifacts, and the job queue (PLAN.md §7.1).

Plain stdlib sqlite3 with JSON columns. One file per project (default runs/dreammachine.db) is easy to copy
from the GPU laptop and to serve from the API. The API, the worker and every step subprocess open the same
file, so connections use WAL mode and a busy timeout.

Migrations are additive only: new tables are created if missing, new columns on `runs` are added if missing
(checked with PRAGMA table_info). Nothing is ever dropped. Times are epoch floats.
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
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,                 -- 'pipeline' | 'benchmark_run'
    name TEXT NOT NULL,
    mode TEXT NOT NULL,                 -- 'explore' | 'paper'
    status TEXT NOT NULL,               -- queued | running | done | failed | cancelled
    request TEXT NOT NULL,              -- JSON: exactly what was sent
    resolved TEXT,                      -- JSON
    baseline_hash TEXT,
    diagnosis_hash TEXT,
    config_hash TEXT,
    output_dir TEXT,
    reuse_from TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    warnings TEXT NOT NULL DEFAULT '[]', -- JSON list
    error TEXT,                         -- JSON {code, message, details, step_key}
    provenance TEXT,                    -- JSON
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at);
CREATE TABLE IF NOT EXISTS job_steps (
    job_id TEXT NOT NULL REFERENCES jobs(id),
    idx INTEGER NOT NULL,
    key TEXT NOT NULL,
    stage TEXT NOT NULL,
    params TEXT NOT NULL DEFAULT '{}',  -- JSON
    status TEXT NOT NULL,               -- pending | running | done | failed | skipped | cancelled
    progress_done INTEGER NOT NULL DEFAULT 0,
    progress_total INTEGER NOT NULL DEFAULT 0,
    progress_unit TEXT NOT NULL DEFAULT 'items',
    run_ids TEXT NOT NULL DEFAULT '[]', -- JSON list
    metrics TEXT,                       -- JSON
    started_at REAL,
    finished_at REAL,
    error TEXT,                         -- JSON
    PRIMARY KEY (job_id, idx)
);
CREATE TABLE IF NOT EXISTS worker_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    pid INTEGER,
    heartbeat_at REAL,
    current_job_id TEXT,
    current_step_idx INTEGER
);
"""

# Columns added to `runs` after the first release (PLAN.md §7.1): name -> SQL type.
_RUN_COLUMNS = {"job_id": "TEXT", "mode": "TEXT"}

JOB_STATUSES = ("queued", "running", "done", "failed", "cancelled")
STEP_STATUSES = ("pending", "running", "done", "failed", "skipped", "cancelled")

_JOB_JSON = ("request", "resolved", "warnings", "error", "provenance")
_STEP_JSON = ("params", "run_ids", "metrics", "error")
_JOB_FIELDS = ("kind", "name", "mode", "status", "request", "resolved", "baseline_hash", "diagnosis_hash",
               "config_hash", "output_dir", "reuse_from", "cancel_requested", "warnings", "error", "provenance",
               "started_at", "finished_at")
_STEP_FIELDS = ("key", "stage", "params", "status", "progress_done", "progress_total", "progress_unit", "run_ids",
                "metrics", "started_at", "finished_at", "error")


def _dumps(v: Any) -> str | None:
    return None if v is None else json.dumps(v)


def _loads(v: str | None) -> Any:
    return None if v is None else json.loads(v)


class Store:
    def __init__(self, path: str | Path = "runs/dreammachine.db") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path) if str(path) == ":memory:" else str(Path(path).resolve())
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            # WAL lets readers (API) and one writer (worker or a step subprocess) share the file;
            # busy_timeout makes a writer wait instead of failing with "database is locked".
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(_SCHEMA)
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        have = {r["name"] for r in self._conn.execute("PRAGMA table_info(runs)")}
        for col, typ in _RUN_COLUMNS.items():
            if col not in have:
                self._conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {typ}")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_job ON runs(job_id)")

    def journal_mode(self) -> str:
        with self._lock:
            return self._conn.execute("PRAGMA journal_mode").fetchone()[0]

    # ------------------------------------------------------------------ runs
    def create_run(self, name: str, kind: str, config: dict[str, Any], job_id: str | None = None,
                   mode: str | None = None) -> str:
        run_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs (id, name, kind, status, config, created_at, job_id, mode) VALUES (?,?,?,?,?,?,?,?)",
                (run_id, name, kind, "running", json.dumps(config), time.time(), job_id, mode))
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

    @staticmethod
    def _run_filter(kind: str | None, job_id: str | None) -> tuple[str, tuple]:
        where, args = [], []
        if kind:
            where.append("kind=?"); args.append(kind)
        if job_id:
            where.append("job_id=?"); args.append(job_id)
        return (" WHERE " + " AND ".join(where) if where else ""), tuple(args)

    def list_runs(self, kind: str | None = None, job_id: str | None = None, limit: int | None = None,
                  offset: int = 0) -> list[dict]:
        where, args = self._run_filter(kind, job_id)
        q = "SELECT * FROM runs" + where + " ORDER BY created_at DESC"
        if limit is not None:
            q, args = q + " LIMIT ? OFFSET ?", args + (limit, offset)
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        return [self._run_row(r) for r in rows]

    def count_runs(self, kind: str | None = None, job_id: str | None = None) -> int:
        where, args = self._run_filter(kind, job_id)
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM runs" + where, args).fetchone()[0]

    # ------------------------------------------------------------- responses
    def add_responses(self, run_id: str, records: Iterable) -> None:
        rows = [(run_id, r.example_id, r.source, r.sample, r.text, int(r.correct), r.error_type,
                 json.dumps(r.diagnosis), json.dumps(r.features)) for r in records]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?,?,?,?)", rows)
            self._conn.commit()

    @staticmethod
    def _resp_filter(run_id: str, correct: bool | None, error_type: str | None,
                     source: str | None) -> tuple[str, tuple]:
        where, args = ["run_id=?"], [run_id]
        if correct is not None:
            where.append("correct=?"); args.append(int(correct))
        if error_type:
            where.append("error_type=?"); args.append(error_type)
        if source:
            where.append("source=?"); args.append(source)
        return " WHERE " + " AND ".join(where), tuple(args)

    def get_responses(self, run_id: str, limit: int | None = None, offset: int = 0, correct: bool | None = None,
                      error_type: str | None = None, source: str | None = None) -> list[dict]:
        where, args = self._resp_filter(run_id, correct, error_type, source)
        q = "SELECT * FROM responses" + where + " ORDER BY example_id, sample"
        if limit is not None:
            q, args = q + " LIMIT ? OFFSET ?", args + (limit, offset)
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

    def count_responses(self, run_id: str, correct: bool | None = None, error_type: str | None = None,
                        source: str | None = None) -> int:
        """How many responses match the same filters as get_responses (for pagination totals)."""
        where, args = self._resp_filter(run_id, correct, error_type, source)
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM responses" + where, args).fetchone()[0]

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

    # ------------------------------------------------------------------ jobs
    @staticmethod
    def _job_row(r: sqlite3.Row) -> dict:
        d = dict(r)
        for k in _JOB_JSON:
            d[k] = _loads(d[k])
        d["cancel_requested"] = bool(d["cancel_requested"])
        return d

    def create_job(self, kind: str, name: str, mode: str, request: dict, job_id: str | None = None,
                   **fields: Any) -> str:
        job_id = job_id or uuid.uuid4().hex[:12]
        bad = set(fields) - set(_JOB_FIELDS)
        if bad:
            raise KeyError(f"unknown job fields {sorted(bad)}")
        row = {"kind": kind, "name": name, "mode": mode, "status": "queued", "request": request,
               "warnings": [], **fields}
        cols = ["id", *row, "created_at"]
        vals = [job_id, *(_dumps(v) if k in _JOB_JSON else v for k, v in row.items()), time.time()]
        with self._lock:
            self._conn.execute(f"INSERT INTO jobs ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", vals)
            self._conn.commit()
        return job_id

    def get_job(self, job_id: str) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._job_row(r) if r else None

    @staticmethod
    def _job_filter(kind: str | None, mode: str | None, status: str | None) -> tuple[str, tuple]:
        where, args = [], []
        for col, v in (("kind", kind), ("mode", mode), ("status", status)):
            if v:
                where.append(f"{col}=?"); args.append(v)
        return (" WHERE " + " AND ".join(where) if where else ""), tuple(args)

    def list_jobs(self, kind: str | None = None, mode: str | None = None, status: str | None = None,
                  limit: int | None = None, offset: int = 0, oldest_first: bool = False) -> list[dict]:
        where, args = self._job_filter(kind, mode, status)
        q = "SELECT * FROM jobs" + where + f" ORDER BY created_at {'ASC' if oldest_first else 'DESC'}"
        if limit is not None:
            q, args = q + " LIMIT ? OFFSET ?", args + (limit, offset)
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        return [self._job_row(r) for r in rows]

    def count_jobs(self, kind: str | None = None, mode: str | None = None, status: str | None = None) -> int:
        where, args = self._job_filter(kind, mode, status)
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM jobs" + where, args).fetchone()[0]

    def update_job(self, job_id: str, **fields: Any) -> None:
        bad = set(fields) - set(_JOB_FIELDS)
        if bad:
            raise KeyError(f"unknown job fields {sorted(bad)}")
        if "status" in fields and fields["status"] not in JOB_STATUSES:
            raise ValueError(f"bad job status {fields['status']!r}")
        if not fields:
            return
        sets = ", ".join(f"{k}=?" for k in fields)
        vals = [_dumps(v) if k in _JOB_JSON else (int(v) if k == "cancel_requested" else v)
                for k, v in fields.items()]
        with self._lock:
            self._conn.execute(f"UPDATE jobs SET {sets} WHERE id=?", (*vals, job_id))
            self._conn.commit()

    def add_job_warning(self, job_id: str, warning: str) -> None:
        job = self.get_job(job_id)
        if job is not None and warning not in job["warnings"]:
            self.update_job(job_id, warnings=job["warnings"] + [warning])

    def delete_job(self, job_id: str) -> int:
        """Remove the job, its steps, and the runs/responses/artifacts linked to it. Returns runs deleted.
        Files on disk and the rules on *which* jobs may be deleted (PLAN.md §7.8) belong to the caller."""
        with self._lock:
            run_ids = [r[0] for r in self._conn.execute("SELECT id FROM runs WHERE job_id=?", (job_id,))]
            for rid in run_ids:
                self._conn.execute("DELETE FROM responses WHERE run_id=?", (rid,))
                self._conn.execute("DELETE FROM artifacts WHERE run_id=?", (rid,))
            self._conn.execute("DELETE FROM runs WHERE job_id=?", (job_id,))
            self._conn.execute("DELETE FROM job_steps WHERE job_id=?", (job_id,))
            self._conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            self._conn.commit()
        return len(run_ids)

    # ------------------------------------------------------------- job steps
    @staticmethod
    def _step_row(r: sqlite3.Row) -> dict:
        d = dict(r)
        for k in _STEP_JSON:
            d[k] = _loads(d[k])
        return d

    def set_steps(self, job_id: str, steps: list[dict]) -> None:
        """Replace the job's step list. Each step: {key, stage, params?, progress_unit?}; status starts 'pending'."""
        with self._lock:
            self._conn.execute("DELETE FROM job_steps WHERE job_id=?", (job_id,))
            self._conn.executemany(
                "INSERT INTO job_steps (job_id, idx, key, stage, params, status, progress_unit) "
                "VALUES (?,?,?,?,?,?,?)",
                [(job_id, i, s["key"], s["stage"], json.dumps(s.get("params", {})), s.get("status", "pending"),
                  s.get("progress_unit", "items")) for i, s in enumerate(steps)])
            self._conn.commit()

    def get_steps(self, job_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM job_steps WHERE job_id=? ORDER BY idx", (job_id,)).fetchall()
        return [self._step_row(r) for r in rows]

    def get_step(self, job_id: str, idx: int) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM job_steps WHERE job_id=? AND idx=?", (job_id, idx)).fetchone()
        return self._step_row(r) if r else None

    def update_step(self, job_id: str, idx: int, **fields: Any) -> None:
        bad = set(fields) - set(_STEP_FIELDS)
        if bad:
            raise KeyError(f"unknown step fields {sorted(bad)}")
        if "status" in fields and fields["status"] not in STEP_STATUSES:
            raise ValueError(f"bad step status {fields['status']!r}")
        if not fields:
            return
        sets = ", ".join(f"{k}=?" for k in fields)
        vals = [_dumps(v) if k in _STEP_JSON else v for k, v in fields.items()]
        with self._lock:
            self._conn.execute(f"UPDATE job_steps SET {sets} WHERE job_id=? AND idx=?", (*vals, job_id, idx))
            self._conn.commit()

    # ---------------------------------------------------------- worker state
    def set_worker_state(self, pid: int | None, current_job_id: str | None = None,
                         current_step_idx: int | None = None, heartbeat_at: float | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO worker_state (id, pid, heartbeat_at, current_job_id, current_step_idx) "
                "VALUES (1,?,?,?,?) ON CONFLICT(id) DO UPDATE SET pid=excluded.pid, "
                "heartbeat_at=excluded.heartbeat_at, current_job_id=excluded.current_job_id, "
                "current_step_idx=excluded.current_step_idx",
                (pid, time.time() if heartbeat_at is None else heartbeat_at, current_job_id, current_step_idx))
            self._conn.commit()

    def get_worker_state(self) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT pid, heartbeat_at, current_job_id, current_step_idx "
                                   "FROM worker_state WHERE id=1").fetchone()
        return dict(r) if r else None

    def close(self) -> None:
        self._conn.close()
