"""A stand-in for `python -m dreammachine.jobs.execute`, used by tests/test_worker.py.

    python fake_executor.py <job_id> <step_idx> <db>

The job's name picks the behaviour of one step: "<mode>:<idx>" (other steps succeed quickly).
    fail      record a failed step, exit 1
    crash     exit 1 without recording anything (the step stays `running`)
    coop      wait for cancel_requested, then record `cancelled`, exit 3 (cooperative cancel)
    ignore    ignore cancel_requested and keep running (must be hard-killed)
Every call is appended to calls.txt next to the database, so tests can see which steps ran.
"""

import sys
import time
from pathlib import Path

from dreammachine.store import Store

job_id, idx, db = sys.argv[1], int(sys.argv[2]), sys.argv[3]
store = Store(db)
job, step = store.get_job(job_id), store.get_step(job_id, idx)
with (Path(db).parent / "calls.txt").open("a", encoding="utf-8") as f:
    f.write(f"{job_id} {idx}\n")
mode, _, at = (job["name"] or "").partition(":")
store.update_step(job_id, idx, status="running", started_at=time.time())
print(f"fake step {idx} {step['key']} (mode {mode or 'ok'})", flush=True)

if at.isdigit() and int(at) == idx:
    if mode == "fail":
        store.update_step(job_id, idx, status="failed", finished_at=time.time(),
                          error={"code": "out_of_memory", "message": "fake OOM", "details": None})
        sys.exit(1)
    if mode == "crash":
        sys.exit(1)
    if mode == "coop":
        while not store.get_job(job_id)["cancel_requested"]:
            time.sleep(0.1)
        store.update_step(job_id, idx, status="cancelled", finished_at=time.time(),
                          error={"code": "cancelled", "message": "cancelled at a safe point", "details": None})
        sys.exit(3)
    if mode == "ignore":
        while True:
            time.sleep(0.2)

store.update_step(job_id, idx, status="done", finished_at=time.time(),
                  metrics={"load_s": 0.0, "run_s": 0.01, "gpu": None})
sys.exit(0)
