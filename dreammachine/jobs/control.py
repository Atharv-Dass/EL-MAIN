"""Cancel, resume and delete a job (PLAN.md §7.3, §7.8; contract §6.2). The API (B7) and the CLI call these;
they only change the database (and, for delete, the job's own folder). Errors carry contract codes:
not_found (404), conflict / paper_protected (409)."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from ..store.db import Store
from .errors import JobError

ACTIVE = ("queued", "running")


def _get(store: Store, job_id: str, kind: str | None = None) -> dict:
    job = store.get_job(job_id)
    if job is None or (kind is not None and job["kind"] != kind):
        raise JobError("not_found", f"no {kind or 'job'} {job_id!r}")
    return job


def cancel_job(store: Store, job_id: str, kind: str | None = None) -> dict:
    """queued -> cancelled at once; running -> cancel_requested (the step stops at its next safe point and the
    worker hard-kills it only after the grace period); finished -> conflict."""
    job = _get(store, job_id, kind)
    if job["status"] == "queued":
        store.update_job(job_id, status="cancelled", cancel_requested=True, finished_at=time.time(),
                         error={"code": "cancelled", "message": "cancelled before it started", "details": None,
                                "step_key": None})
    elif job["status"] == "running":
        store.update_job(job_id, cancel_requested=True)
    else:
        raise JobError("conflict", f"job {job_id} is already {job['status']}")
    return store.get_job(job_id)


def resume_job(store: Store, job_id: str, kind: str | None = None) -> dict:
    """failed / cancelled -> queued. Done and skipped steps are kept; failed, cancelled and pending steps start
    again from the beginning of the step (training continues from its last complete checkpoint)."""
    job = _get(store, job_id, kind)
    if job["status"] not in ("failed", "cancelled"):
        raise JobError("conflict", f"only failed or cancelled jobs can be resumed; job {job_id} is {job['status']}")
    for s in store.get_steps(job_id):
        if s["status"] in ("failed", "cancelled", "running", "pending"):
            store.update_step(job_id, s["idx"], status="pending", error=None, started_at=None, finished_at=None,
                              progress_done=0)
    store.update_job(job_id, status="queued", cancel_requested=False, error=None, finished_at=None)
    return store.get_job(job_id)


def folder_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def delete_job(store: Store, job_id: str, kind: str | None = None) -> dict:
    """Remove a finished explore pipeline or benchmark run: its rows (job, steps, linked runs/responses/
    artifacts) and its output folder. Paper jobs -> paper_protected; queued/running -> conflict."""
    job = _get(store, job_id, kind)
    if job["mode"] == "paper":
        raise JobError("paper_protected", "paper jobs cannot be deleted through the API; delete them by hand "
                       "if you really mean it")
    if job["status"] in ACTIVE:
        raise JobError("conflict", f"job {job_id} is {job['status']}; cancel it first")
    out = Path(job["output_dir"]) if job["output_dir"] else None
    freed = 0
    # Only ever remove the job's own folder: its last path part must be the job id.
    if out is not None and out.name == job_id and out.exists():
        freed = folder_bytes(out)
        shutil.rmtree(out)
    store.delete_job(job_id)
    return {"deleted": True, "freed_bytes": freed}
