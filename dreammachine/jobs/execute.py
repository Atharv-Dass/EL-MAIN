"""Run exactly one step of a job (PLAN.md §7.4), in this process.

    python -m dreammachine.jobs.execute --job <id> --step <idx> [--db runs/dreammachine.db]

Exit code: 0 done (or already done/skipped), 3 cancelled, 1 failed. The worker (B5) runs every step in its own
subprocess this way, so GPU memory is freed after each step and a crash fails one step, not the worker.
`run_job` runs a whole job in the foreground (development; `python -m dreammachine.jobs run`).
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from functools import partial
from typing import Callable

from ..errors import Cancelled
from ..experiments import pipeline as P
from ..store.db import Store
from .errors import error_from_exception
from .stages import GPU_STAGES, STAGE_FUNCS, StepContext

EXIT = {"done": 0, "skipped": 0, "cancelled": 3, "failed": 1}


def _default_factory(cfg: P.Config) -> P.RunnerFactory:
    return partial(P.hf_runner_factory, load_in_4bit=bool(cfg.eval.get("load_in_4bit", False)))


def _cap_gpu_memory(cfg: P.Config, log: Callable[[str], None]) -> None:
    """Make an oversized batch fail with OOM instead of Windows spilling VRAM into system RAM (RUNNING.md §8)."""
    frac = cfg.gpu.get("memory_fraction", 0.92)
    if not frac:
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(float(frac))
            log(f"GPU memory capped at {float(frac):.0%} of dedicated VRAM")
    except Exception as e:  # noqa: BLE001 - a missing cap must not stop the step
        log(f"could not cap GPU memory ({type(e).__name__})")


def execute_step(store: Store, job_id: str, idx: int, runner_factory: P.RunnerFactory | None = None,
                 log: Callable[[str], None] = print) -> str:
    """Run step `idx` of job `job_id`. Returns the step's final status. Never raises for a step failure."""
    job, step = store.get_job(job_id), store.get_step(job_id, idx)
    if job is None or step is None:
        raise KeyError(f"no step {idx} in job {job_id}")
    if step["status"] in ("done", "skipped"):
        return step["status"]
    cfg = P.Config(**job["resolved"]["config"])
    ctx = StepContext(store, job, step, cfg, runner_factory or _default_factory(cfg), log=log)
    t0 = time.time()
    store.update_step(job_id, idx, status="running", started_at=t0, finished_at=None, error=None,
                      progress_done=0, progress_total=0)
    if job["status"] == "queued":
        store.update_job(job_id, status="running", started_at=job["started_at"] or t0)
    log(f"[{time.strftime('%H:%M:%S')}] step {idx} {step['key']}: start")
    if step["stage"] in GPU_STAGES:
        _cap_gpu_memory(cfg, log)
    status, error = "done", None
    try:
        STAGE_FUNCS[step["stage"]](ctx)
    except Cancelled as e:
        status, error = "cancelled", error_from_exception(e, step["key"])
    except Exception as e:  # noqa: BLE001 - every failure becomes a structured step error
        status, error = "failed", error_from_exception(e, step["key"])
        log(f"[{time.strftime('%H:%M:%S')}] step {idx} {step['key']} FAILED: {error['code']}")
        log(traceback.format_exc())
    total = time.time() - t0
    if ctx.metrics.get("run_s") is None:
        ctx.metrics["run_s"] = round(max(total - (ctx.metrics.get("load_s") or 0.0), 0.0), 3)
    store.update_step(job_id, idx, status=status, finished_at=time.time(), error=error, metrics=ctx.metrics,
                      run_ids=ctx.run_ids)
    log(f"[{time.strftime('%H:%M:%S')}] step {idx} {step['key']}: {status} ({total:.1f}s)")
    return status


def finish_job(store: Store, job_id: str) -> str:
    """Set the job status from its steps: first failed/cancelled step wins, else done when all are finished."""
    steps = store.get_steps(job_id)
    for s in steps:
        if s["status"] in ("failed", "cancelled"):
            err = {**(s["error"] or {"code": s["status"], "message": s["status"], "details": None}),
                   "step_key": s["key"]}
            store.update_job(job_id, status=s["status"], finished_at=time.time(), error=err)
            return s["status"]
    if all(s["status"] in ("done", "skipped") for s in steps):
        store.update_job(job_id, status="done", finished_at=time.time(), error=None)
        return "done"
    return store.get_job(job_id)["status"]


def run_job(store: Store, job_id: str, runner_factory: P.RunnerFactory | None = None,
            log: Callable[[str], None] = print) -> str:
    """Run every pending step in order, in this process (no worker). Stops at the first failed or cancelled step."""
    for step in store.get_steps(job_id):
        if step["status"] in ("done", "skipped"):
            continue
        job = store.get_job(job_id)
        if job["cancel_requested"]:
            store.update_step(job_id, step["idx"], status="cancelled", finished_at=time.time(),
                              error={"code": "cancelled", "message": "cancelled before the step started",
                                     "details": None})
            break
        if execute_step(store, job_id, step["idx"], runner_factory, log) in ("failed", "cancelled"):
            break
    return finish_job(store, job_id)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m dreammachine.jobs.execute", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--job", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--db", default="runs/dreammachine.db")
    args = p.parse_args(argv)
    status = execute_step(Store(args.db), args.job, args.step, log=lambda s: print(s, flush=True))
    return EXIT.get(status, 1)


if __name__ == "__main__":
    sys.exit(main())
