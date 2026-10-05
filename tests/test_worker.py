"""Worker, cancel, resume, recovery and delete (PLAN.md milestone B5). Steps run as real subprocesses
(tests/fake_executor.py) so process handling is tested the way the worker really works, without a model."""

import sys
import threading
import time
from pathlib import Path

import pytest

from dreammachine.jobs import JobError
from dreammachine.jobs.control import cancel_job, delete_job, resume_job
from dreammachine.jobs.worker import Worker, WorkerBusy, gpu_summary
from dreammachine.store import Store

FAKE = str(Path(__file__).with_name("fake_executor.py"))


@pytest.fixture
def env(tmp_path):
    db = tmp_path / "dm.db"
    store = Store(db)

    def make_job(name="ok", n_steps=3, mode="explore", kind="pipeline"):
        jid = store.create_job(kind, name, mode, {"mode": mode})
        out = tmp_path / "pipelines" / jid
        out.mkdir(parents=True)
        store.update_job(jid, output_dir=str(out))
        store.set_steps(jid, [{"key": f"s{i}", "stage": "preflight"} for i in range(n_steps)])
        return jid

    def worker(**kw):
        kw.setdefault("sample_gpu", None)
        return Worker(store, str(db), step_command=lambda j, i, d: [sys.executable, FAKE, j, str(i), d],
                      poll_s=0.05, heartbeat_s=0.2, log=lambda s: None, **kw)

    def calls():
        p = tmp_path / "calls.txt"
        return p.read_text(encoding="utf-8").split() if p.exists() else []

    return store, make_job, worker, calls, tmp_path


def _statuses(store, jid):
    return [s["status"] for s in store.get_steps(jid)]


def _set_cancel_when_running(store, jid, idx, timeout=60):
    def go():
        t = time.time()
        while time.time() - t < timeout:
            if store.get_step(jid, idx)["status"] == "running":
                cancel_job(store, jid)
                return
            time.sleep(0.05)
    th = threading.Thread(target=go, daemon=True)
    th.start()
    return th


def test_queued_job_runs_to_done_with_logs(env):
    store, make_job, worker, calls, _ = env
    jid = make_job()
    w = worker()
    assert w.run_once() == "done"
    job = store.get_job(jid)
    assert job["status"] == "done" and job["started_at"] and job["finished_at"] and job["error"] is None
    assert _statuses(store, jid) == ["done", "done", "done"]
    log = (Path(job["output_dir"]) / "logs" / "job.log").read_text(encoding="utf-8")
    assert "fake step 0 s0" in log and "worker: job" in log and "finished: done" in log
    assert w.run_once() is None                          # queue empty
    st = store.get_worker_state()
    assert st["pid"] == w.pid and st["current_job_id"] is None and time.time() - st["heartbeat_at"] < 5


def test_jobs_run_oldest_first(env):
    store, make_job, worker, calls, _ = env
    a, b = make_job(n_steps=1), make_job(n_steps=1)
    w = worker()
    w.run_once(); w.run_once()
    assert [c for c in calls()[::2]] == [a, b]


def test_failure_fails_the_job_and_resume_continues_after_it(env):
    store, make_job, worker, calls, _ = env
    jid = make_job("fail:1")
    assert worker().run_once() == "failed"
    assert _statuses(store, jid) == ["done", "failed", "pending"]
    err = store.get_job(jid)["error"]
    assert err["code"] == "out_of_memory" and err["step_key"] == "s1"
    with pytest.raises(JobError) as e:
        resume_job(store, make_job(n_steps=1))          # a queued job cannot be resumed
    assert e.value.code == "conflict"
    store.update_job(jid, name="ok")                     # the next attempt succeeds
    resume_job(store, jid)
    assert store.get_job(jid)["status"] == "queued" and _statuses(store, jid) == ["done", "pending", "pending"]
    assert worker().run_once() == "done"
    ran = [int(c) for j, c in zip(calls()[::2], calls()[1::2]) if j == jid]
    assert ran == [0, 1, 1, 2]                           # step 0 was not run again


def test_crashed_step_without_result_is_failed(env):
    store, make_job, worker, _, _ = env
    jid = make_job("crash:0", n_steps=2)
    assert worker().run_once() == "failed"
    s = store.get_step(jid, 0)
    assert s["status"] == "failed" and s["error"]["code"] == "internal_error" and "exit" in s["error"]["message"]


def test_cooperative_cancel(env):
    store, make_job, worker, _, _ = env
    jid = make_job("coop:1")
    th = _set_cancel_when_running(store, jid, 1)
    assert worker(cancel_grace_s=60).run_once() == "cancelled"
    th.join(1)
    assert _statuses(store, jid) == ["done", "cancelled", "pending"]
    assert store.get_step(jid, 1)["error"]["message"] == "cancelled at a safe point"   # stopped by itself
    assert store.get_job(jid)["error"]["code"] == "cancelled"


def test_step_ignoring_cancel_is_killed_after_grace(env):
    store, make_job, worker, _, _ = env
    jid = make_job("ignore:0", n_steps=2)
    th = _set_cancel_when_running(store, jid, 0)
    t0 = time.time()
    assert worker(cancel_grace_s=1.0).run_once() == "cancelled"
    th.join(1)
    assert 1.0 <= time.time() - t0 < 30
    s = store.get_step(jid, 0)
    assert s["status"] == "cancelled" and "by force" in s["error"]["message"]
    assert _statuses(store, jid) == ["cancelled", "pending"]
    log = (Path(store.get_job(jid)["output_dir"]) / "logs" / "job.log").read_text(encoding="utf-8")
    assert "killing it" in log


def test_cancel_queued_and_finished_jobs(env):
    store, make_job, worker, calls, _ = env
    jid = make_job()
    assert cancel_job(store, jid)["status"] == "cancelled"
    assert worker().run_once() is None and calls() == []          # never started
    with pytest.raises(JobError) as e:
        cancel_job(store, jid)
    assert e.value.code == "conflict"
    with pytest.raises(JobError) as e:
        cancel_job(store, "nope")
    assert e.value.code == "not_found"


def test_recovery_marks_stale_running_steps_interrupted(env):
    store, make_job, worker, _, _ = env
    jid = make_job()
    store.update_job(jid, status="running", started_at=time.time())
    store.update_step(jid, 0, status="done")
    store.update_step(jid, 1, status="running", started_at=time.time())
    assert worker().recover() == 1
    s = store.get_step(jid, 1)
    assert s["status"] == "failed" and s["error"]["code"] == "interrupted"
    job = store.get_job(jid)
    assert job["status"] == "failed" and job["error"]["code"] == "interrupted" and job["error"]["step_key"] == "s1"
    resume_job(store, jid)
    assert worker().run_once() == "done" and _statuses(store, jid) == ["done", "done", "done"]


def test_second_worker_is_refused_while_the_first_is_alive(env):
    store, _, worker, _, _ = env
    first = worker()
    first.claim()
    second = worker()
    second.pid = first.pid + 1
    with pytest.raises(WorkerBusy):
        second.claim()
    store.set_worker_state(first.pid, None, None, heartbeat_at=time.time() - 60)   # first worker died long ago
    second.claim()
    assert store.get_worker_state()["pid"] == second.pid


def test_gpu_telemetry_is_logged_and_summarised(env):
    store, make_job, worker, _, _ = env
    jid = make_job("coop:0", n_steps=1)
    readings = iter([{"temp_c": 61.0, "sm_clock_mhz": 2100.0, "power_w": 40.0, "util_pct": 90.0,
                      "memory_used_mb": 5000.0},
                     {"temp_c": 70.0, "sm_clock_mhz": 1800.0, "power_w": 45.0, "util_pct": 95.0,
                      "memory_used_mb": 6200.0}])
    sampler = lambda: next(readings, {"temp_c": 65.0, "sm_clock_mhz": 1900.0, "power_w": 41.0, "util_pct": 80.0,
                                      "memory_used_mb": 5500.0})
    w = worker(sample_gpu=sampler, gpu_sample_s=0.05)
    def cancel_later():
        time.sleep(1.0)
        cancel_job(store, jid)
    threading.Thread(target=cancel_later, daemon=True).start()
    w.run_once()
    gpu = store.get_step(jid, 0)["metrics"]["gpu"]
    assert gpu["max_temp_c"] == 70.0 and gpu["min_sm_clock_mhz"] == 1800.0 and gpu["max_memory_used_mb"] == 6200.0
    rows = (Path(store.get_job(jid)["output_dir"]) / "logs" / "gpu.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("timestamp,step_key,temp_c") and len(rows) >= 3 and ",s0,61," in rows[1]


def test_no_nvidia_smi_means_no_gpu_summary(env):
    store, make_job, worker, _, _ = env
    jid = make_job(n_steps=1)
    worker(sample_gpu=lambda: None, gpu_sample_s=0.01).run_once()
    assert store.get_step(jid, 0)["metrics"]["gpu"] is None
    assert gpu_summary([]) is None


def test_delete_rules(env):
    store, make_job, worker, _, tmp = env
    paper = make_job(mode="paper", n_steps=1)
    running = make_job(n_steps=1)
    store.update_job(running, status="running")
    for jid, code in ((paper, "paper_protected"), (running, "conflict"), ("nope", "not_found")):
        with pytest.raises(JobError) as e:
            delete_job(store, jid)
        assert e.value.code == code
    done = make_job(n_steps=1)
    while store.get_job(done)["status"] != "done":     # the queue runs oldest first (the paper job before it)
        worker().run_once()
    out = Path(store.get_job(done)["output_dir"])
    (out / "data.bin").write_bytes(b"x" * 1000)
    run = store.create_run("r", "eval", {}, job_id=done)
    res = delete_job(store, done)
    assert res["deleted"] is True and res["freed_bytes"] >= 1000
    assert not out.exists() and store.get_job(done) is None and store.get_run(run) is None
    assert store.get_job(paper) is not None
