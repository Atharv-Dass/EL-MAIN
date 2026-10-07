"""Store changes for the job system (PLAN.md §7.1, milestone B3)."""

import json
import sqlite3
import threading

import pytest

from dreammachine.experiments.evaluate import ResponseRecord
from dreammachine.store import Store

# The schema as it was before B3, to build an "old" database file.
OLD_SCHEMA = """
CREATE TABLE runs (id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL,
                   config TEXT NOT NULL, metrics TEXT, created_at REAL NOT NULL, finished_at REAL);
CREATE TABLE responses (run_id TEXT NOT NULL REFERENCES runs(id), example_id TEXT NOT NULL, source TEXT NOT NULL,
                        sample INTEGER NOT NULL, text TEXT NOT NULL, correct INTEGER NOT NULL, error_type TEXT NOT NULL,
                        diagnosis TEXT NOT NULL, features TEXT NOT NULL, PRIMARY KEY (run_id, example_id, sample));
CREATE TABLE artifacts (run_id TEXT NOT NULL REFERENCES runs(id), key TEXT NOT NULL, value TEXT NOT NULL,
                        PRIMARY KEY (run_id, key));
CREATE INDEX idx_resp_run ON responses(run_id);
"""


def _rec(i, correct, err="PLAN_ERROR", source="gsm8k"):
    return ResponseRecord(f"e{i:02d}", source, 0, f"answer {i}", correct, "CORRECT" if correct else err, {"i": i}, {})


def test_new_tables_wal_and_busy_timeout(tmp_path):
    s = Store(tmp_path / "dm.db")
    tables = {r[0] for r in s._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"runs", "responses", "artifacts", "jobs", "job_steps", "worker_state"} <= tables
    assert s.journal_mode() == "wal"
    assert s._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    cols = {r["name"] for r in s._conn.execute("PRAGMA table_info(runs)")}
    assert {"job_id", "mode"} <= cols


def test_migration_keeps_old_rows(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript(OLD_SCHEMA)
    con.execute("INSERT INTO runs VALUES ('r1','main:eval:base','eval','done','{\"arm\": \"base\"}','{\"gsm8k\": 0.4}',1.0,2.0)")
    con.execute("INSERT INTO runs VALUES ('r2','main:diagnose','diagnose','done','{}',NULL,3.0,4.0)")
    con.executemany("INSERT INTO responses VALUES (?,?,?,?,?,?,?,?,?)",
                    [("r1", f"e{i}", "gsm8k", 0, "t", i % 2, "CORRECT" if i % 2 else "PLAN_ERROR", "{}", "{}")
                     for i in range(5)])
    con.execute("INSERT INTO artifacts VALUES ('r1','lltm','{\"eta\": [1.0]}')")
    con.commit(); con.close()

    s = Store(path)
    assert {r["name"] for r in s._conn.execute("PRAGMA table_info(runs)")} >= {"job_id", "mode"}
    r1 = s.get_run("r1")
    assert r1["config"] == {"arm": "base"} and r1["metrics"] == {"gsm8k": 0.4} and r1["job_id"] is None
    assert s.count_runs() == 2 and s.count_responses("r1") == 5 and s.get_artifact("r1", "lltm") == {"eta": [1.0]}
    s.create_run("new", "eval", {}, job_id="j1", mode="explore")   # old file now accepts the new columns
    s.close()
    s2 = Store(path)                                                # migrating again changes nothing
    assert s2.count_runs() == 3 and s2.count_responses("r1") == 5 and s2.journal_mode() == "wal"


def test_response_filters_and_counts():
    s = Store(":memory:")
    run = s.create_run("x", "eval", {})
    s.add_responses(run, [_rec(i, i % 3 == 0, "ARITHMETIC_SLIP" if i % 2 else "PLAN_ERROR",
                               "gsm8k" if i < 6 else "probes_heldout_families") for i in range(10)])
    assert s.count_responses(run) == 10
    assert s.count_responses(run, correct=True) == 4                # 0, 3, 6, 9
    assert s.count_responses(run, correct=False, error_type="ARITHMETIC_SLIP") == 3   # 1, 5, 7 (3 and 9 are correct)
    assert s.count_responses(run, source="probes_heldout_families", correct=False) == 2   # 7, 8
    page = s.get_responses(run, limit=2, offset=1, correct=False)
    assert [r["example_id"] for r in page] == ["e02", "e04"] and page[0]["correct"] is False
    assert len(s.get_responses(run, correct=False)) == s.count_responses(run, correct=False) == 6


def test_list_runs_by_job_and_pages():
    s = Store(":memory:")
    a = [s.create_run(f"a{i}", "eval", {}, job_id="job-a", mode="explore") for i in range(3)]
    s.create_run("b", "train", {}, job_id="job-b", mode="paper")
    s.create_run("loose", "eval", {})
    assert {r["id"] for r in s.list_runs(job_id="job-a")} == set(a)
    assert s.count_runs(job_id="job-a") == 3 and s.count_runs(kind="eval") == 4
    assert s.count_runs(kind="eval", job_id="job-b") == 0
    assert s.list_runs(job_id="job-b")[0]["mode"] == "paper"
    assert len(s.list_runs(limit=2, offset=0)) == 2 and len(s.list_runs(limit=2, offset=4)) == 1


def test_jobs_crud():
    s = Store(":memory:")
    j1 = s.create_job("pipeline", "try1", "explore", {"mode": "explore", "model": "Qwen/Qwen3-0.6B"},
                      output_dir="runs/pipelines/x", baseline_hash="bh")
    j2 = s.create_job("benchmark_run", "check", "explore", {"model": "m"})
    j3 = s.create_job("pipeline", "paper", "paper", {"mode": "paper"})
    job = s.get_job(j1)
    assert job["status"] == "queued" and job["request"]["model"] == "Qwen/Qwen3-0.6B" and job["warnings"] == []
    assert job["cancel_requested"] is False and job["baseline_hash"] == "bh" and len(j1) == 12
    assert [j["id"] for j in s.list_jobs(status="queued", oldest_first=True)] == [j1, j2, j3]
    assert s.count_jobs(kind="pipeline") == 2 and s.count_jobs(mode="paper") == 1
    s.update_job(j1, status="running", started_at=10.0, resolved={"ratios": [3]}, cancel_requested=True)
    s.update_job(j1, error={"code": "out_of_memory", "message": "m", "details": None, "step_key": "train:x"})
    job = s.get_job(j1)
    assert job["status"] == "running" and job["resolved"] == {"ratios": [3]} and job["cancel_requested"] is True
    assert job["error"]["code"] == "out_of_memory"
    s.add_job_warning(j1, "worker_not_running"); s.add_job_warning(j1, "worker_not_running")
    assert s.get_job(j1)["warnings"] == ["worker_not_running"]
    with pytest.raises(ValueError):
        s.update_job(j1, status="finished")
    with pytest.raises(KeyError):
        s.update_job(j1, colour="red")
    assert s.get_job("nope") is None


def test_steps():
    s = Store(":memory:")
    j = s.create_job("pipeline", "p", "explore", {})
    s.set_steps(j, [{"key": "preflight", "stage": "preflight"},
                    {"key": "train:targeted_r3_s0", "stage": "train", "params": {"arm": "targeted", "ratio": 3},
                     "progress_unit": "steps"}])
    steps = s.get_steps(j)
    assert [(x["idx"], x["key"], x["status"]) for x in steps] == [(0, "preflight", "pending"),
                                                                 (1, "train:targeted_r3_s0", "pending")]
    assert steps[1]["params"] == {"arm": "targeted", "ratio": 3} and steps[1]["run_ids"] == []
    s.update_step(j, 1, status="running", progress_done=120, progress_total=400, run_ids=["r1"],
                  metrics={"load_s": 14.2, "run_s": None, "gpu": None})
    st = s.get_step(j, 1)
    assert (st["progress_done"], st["progress_total"], st["progress_unit"]) == (120, 400, "steps")
    assert st["run_ids"] == ["r1"] and st["metrics"]["load_s"] == 14.2
    with pytest.raises(ValueError):
        s.update_step(j, 1, status="paused")


def test_delete_job_removes_only_its_rows():
    s = Store(":memory:")
    j, other = s.create_job("pipeline", "p", "explore", {}), s.create_job("pipeline", "q", "explore", {})
    s.set_steps(j, [{"key": "baseline", "stage": "baseline"}])
    mine, keep = s.create_run("mine", "eval", {}, job_id=j), s.create_run("keep", "eval", {}, job_id=other)
    for r in (mine, keep):
        s.add_responses(r, [_rec(0, True)]); s.put_artifact(r, "lltm", {})
    assert s.delete_job(j) == 1
    assert s.get_job(j) is None and s.get_steps(j) == [] and s.get_run(mine) is None
    assert s.count_responses(mine) == 0 and s.get_artifact(mine, "lltm") is None
    assert s.get_run(keep) and s.count_responses(keep) == 1 and s.get_job(other)


def test_worker_state_single_row():
    s = Store(":memory:")
    assert s.get_worker_state() is None
    s.set_worker_state(1234, "job1", 3, heartbeat_at=100.0)
    s.set_worker_state(1234, None, None, heartbeat_at=105.0)
    assert s.get_worker_state() == {"pid": 1234, "heartbeat_at": 105.0, "current_job_id": None,
                                    "current_step_idx": None}
    assert s._conn.execute("SELECT COUNT(*) FROM worker_state").fetchone()[0] == 1


def test_two_connections_share_one_file(tmp_path):
    """API and worker (and step subprocesses) open the same file: writes are visible, concurrent writes wait."""
    path = tmp_path / "shared.db"
    api, worker = Store(path), Store(path)
    j = api.create_job("pipeline", "p", "explore", {})
    worker.update_job(j, status="running")
    assert api.get_job(j)["status"] == "running"
    errors = []

    def write(store, n):
        try:
            for i in range(n):
                store.create_run(f"r{i}", "eval", {"i": i})
        except sqlite3.OperationalError as e:
            errors.append(e)

    ts = [threading.Thread(target=write, args=(st, 50)) for st in (api, worker)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors and api.count_runs() == 100
    assert json.loads(json.dumps(api.get_job(j)))["status"] == "running"
