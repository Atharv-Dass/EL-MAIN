"""The job worker (PLAN.md §7.3, §7.7). One worker, one job at a time (one GPU).

    python -m dreammachine.jobs.worker [--db runs/dreammachine.db] [--cancel-grace-s 120]

Loop: take the oldest queued job, run each pending step in its own subprocess
(`python -m dreammachine.jobs.execute --job <id> --step <idx>`), output appended to <output_dir>/logs/job.log.
A subprocess per step frees all GPU memory after the step, and a crash fails one step, not the worker.

Cancel is cooperative first: the step sees `cancel_requested` at a safe point and exits as `cancelled`.
Only if it is still alive `cancel_grace_s` later does the worker call Popen.kill(). On Windows
Popen.terminate() is the same instant TerminateProcess, so there is no "terminate, then wait"; no POSIX signals.

Heartbeat every ~5 s in worker_state; the API reports the worker as down after 30 s without one.
On start, steps left `running` by a dead worker become `failed` (code `interrupted`) so the user can resume.
While a step runs, nvidia-smi is sampled every 30 s into logs/gpu.csv (skipped if nvidia-smi is missing).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from ..store.db import Store
from .execute import finish_job

HEARTBEAT_STALE_S = 30
GPU_QUERY = ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"]
GPU_FIELDS = ("temp_c", "sm_clock_mhz", "power_w", "util_pct", "memory_used_mb")

StepCommand = Callable[[str, int, str], list[str]]
GpuSampler = Callable[[], "dict | None"]


def default_step_command(job_id: str, idx: int, db: str) -> list[str]:
    return [sys.executable, "-m", "dreammachine.jobs.execute", "--job", job_id, "--step", str(idx), "--db", db]


def sample_nvidia_smi() -> dict | None:
    """One GPU reading, or None if nvidia-smi is missing or fails."""
    try:
        out = subprocess.run(GPU_QUERY, capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    first = out.strip().splitlines()[0] if out.strip() else ""
    vals = [v.strip() for v in first.split(",")]
    if len(vals) != len(GPU_FIELDS):
        return None
    reading = {}
    for k, v in zip(GPU_FIELDS, vals):
        try:
            reading[k] = float(v)
        except ValueError:
            reading[k] = None
    return reading


def gpu_summary(readings: list[dict]) -> dict | None:
    """Per-step summary for job_steps.metrics.gpu (PLAN.md §7.7)."""
    def pick(key, fn):
        vals = [r[key] for r in readings if r.get(key) is not None]
        return fn(vals) if vals else None
    if not readings:
        return None
    return {"max_temp_c": pick("temp_c", max), "min_sm_clock_mhz": pick("sm_clock_mhz", min),
            "max_memory_used_mb": pick("memory_used_mb", max)}


class WorkerBusy(RuntimeError):
    """Another worker has a fresh heartbeat."""


class Worker:
    def __init__(self, store: Store, db_path: str | None = None, step_command: StepCommand = default_step_command,
                 cancel_grace_s: float = 120.0, poll_s: float = 1.0, heartbeat_s: float = 5.0,
                 gpu_sample_s: float = 30.0, sample_gpu: GpuSampler | None = sample_nvidia_smi,
                 log: Callable[[str], None] = print) -> None:
        self.store, self.db = store, db_path or store.path
        self.step_command, self.cancel_grace_s = step_command, cancel_grace_s
        self.poll_s, self.heartbeat_s, self.gpu_sample_s = poll_s, heartbeat_s, gpu_sample_s
        self.sample_gpu, self.log = sample_gpu, log
        self.pid = os.getpid()
        self._last_beat = 0.0
        self._current: tuple[str | None, int | None] = (None, None)
        self.child: subprocess.Popen | None = None

    # ---------------------------------------------------------- lifecycle
    def heartbeat(self, force: bool = False) -> None:
        now = time.time()
        if force or now - self._last_beat >= self.heartbeat_s:
            self.store.set_worker_state(self.pid, *self._current, heartbeat_at=now)
            self._last_beat = now

    def claim(self) -> None:
        """Refuse to start if another worker is alive (fresh heartbeat from another process)."""
        st = self.store.get_worker_state()
        if st and st["pid"] != self.pid and st["heartbeat_at"] and time.time() - st["heartbeat_at"] < HEARTBEAT_STALE_S:
            raise WorkerBusy(f"another worker (pid {st['pid']}) is running; stop it first")
        self.heartbeat(force=True)

    def recover(self) -> int:
        """Steps left `running` by a worker that died: mark them failed (`interrupted`) and fail their job."""
        n = 0
        for job in self.store.list_jobs(status="running"):
            for s in self.store.get_steps(job["id"]):
                if s["status"] == "running":
                    self.store.update_step(job["id"], s["idx"], status="failed", finished_at=time.time(),
                                           error={"code": "interrupted", "message": "the worker stopped while this "
                                                  "step was running; resume the job to continue", "details": None})
                    n += 1
            finish_job(self.store, job["id"])
            if self.store.get_job(job["id"])["status"] == "running":   # no step was running: just fail the job
                self.store.update_job(job["id"], status="failed", finished_at=time.time(),
                                      error={"code": "interrupted", "message": "the worker stopped",
                                             "details": None, "step_key": None})
        return n

    def next_job(self) -> dict | None:
        jobs = self.store.list_jobs(status="queued", limit=1, oldest_first=True)
        return jobs[0] if jobs else None

    # ---------------------------------------------------------------- jobs
    def _log_file(self, job: dict) -> Path:
        p = Path(job["output_dir"]) / "logs" / "job.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _note(self, job: dict, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] worker: {msg}"
        self.log(line)
        with self._log_file(job).open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def run_job(self, job: dict) -> str:
        jid = job["id"]
        self.store.update_job(jid, status="running", started_at=job["started_at"] or time.time())
        self._note(job, f"job {jid} ({job['kind']}, {job['mode']}) started")
        for step in self.store.get_steps(jid):
            if step["status"] in ("done", "skipped"):
                continue
            if self.store.get_job(jid)["cancel_requested"]:
                self.store.update_step(jid, step["idx"], status="cancelled", finished_at=time.time(),
                                       error={"code": "cancelled", "message": "cancelled before the step started",
                                              "details": None})
                break
            status = self.run_step(job, step)
            if status in ("failed", "cancelled"):
                break
        status = finish_job(self.store, jid)
        self._note(job, f"job {jid} finished: {status}")
        self._current = (None, None)
        self.heartbeat(force=True)
        return status

    def run_step(self, job: dict, step: dict) -> str:
        jid, idx, key = job["id"], step["idx"], step["key"]
        self._current = (jid, idx)
        self.heartbeat(force=True)
        self._note(job, f"step {idx} {key}: launching subprocess")
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0   # Ctrl+C reaches only the worker
        log_path = self._log_file(job)
        gpu_csv = log_path.parent / "gpu.csv"
        readings: list[dict] = []
        cancel_seen_at: float | None = None
        killed = False
        with log_path.open("ab") as logf:
            self.child = subprocess.Popen(self.step_command(jid, idx, self.db), stdout=logf, stderr=subprocess.STDOUT,
                                          env=env, creationflags=flags)
            next_sample = time.time()
            try:
                while self.child.poll() is None:
                    self.heartbeat()
                    now = time.time()
                    if self.sample_gpu is not None and now >= next_sample:
                        next_sample = now + self.gpu_sample_s
                        r = self.sample_gpu()
                        if r is None and not readings:
                            self.sample_gpu = None        # nvidia-smi missing: stop trying for this worker
                        elif r is not None:
                            readings.append(r)
                            self._write_gpu_row(gpu_csv, key, r)
                    if cancel_seen_at is None and self.store.get_job(jid)["cancel_requested"]:
                        cancel_seen_at = now
                        self._note(job, f"step {idx} {key}: cancel requested; waiting up to "
                                        f"{self.cancel_grace_s:g}s for it to stop by itself")
                    if cancel_seen_at is not None and now - cancel_seen_at > self.cancel_grace_s:
                        self._note(job, f"step {idx} {key}: still running after the grace period; killing it")
                        self.child.kill()
                        killed = True
                        self.child.wait(timeout=30)
                        break
                    time.sleep(self.poll_s)
            except KeyboardInterrupt:
                self._note(job, f"step {idx} {key}: worker stopped (Ctrl+C); killing the step")
                self.child.kill()
                self.child.wait(timeout=30)
                self._mark(jid, idx, "failed", "interrupted", "the worker was stopped while this step was running; "
                           "resume the job to continue")
                finish_job(self.store, jid)
                raise
        code = self.child.returncode
        self.child = None
        step_now = self.store.get_step(jid, idx)
        status = step_now["status"]
        if killed:
            status = "cancelled"
            self._mark(jid, idx, status, "cancelled", f"stopped by force: the step did not stop within "
                       f"{self.cancel_grace_s:g}s of the cancel request")
        elif status == "running":   # the subprocess died without recording a result
            status = "failed"
            self._mark(jid, idx, status, "internal_error", f"the step process exited with code {code} without "
                       "recording a result; see the job log")
        metrics = {**(step_now.get("metrics") or {}), "gpu": gpu_summary(readings)}
        self.store.update_step(jid, idx, metrics=metrics)
        self._note(job, f"step {idx} {key}: {status} (exit code {code})")
        return status

    def _mark(self, jid: str, idx: int, status: str, code: str, message: str) -> None:
        self.store.update_step(jid, idx, status=status, finished_at=time.time(),
                               error={"code": code, "message": message, "details": None})

    @staticmethod
    def _write_gpu_row(path: Path, key: str, r: dict) -> None:
        new = not path.exists()
        with path.open("a", encoding="utf-8") as f:
            if new:
                f.write("timestamp,step_key," + ",".join(GPU_FIELDS) + "\n")
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')},{key}," +
                    ",".join("" if r.get(k) is None else f"{r[k]:g}" for k in GPU_FIELDS) + "\n")

    # ---------------------------------------------------------------- loop
    def run_once(self) -> str | None:
        """Run the oldest queued job, if any. Returns its final status, or None if the queue was empty."""
        self.heartbeat()
        job = self.next_job()
        return self.run_job(job) if job else None

    def serve_forever(self, idle_s: float = 2.0, should_stop: Callable[[], bool] | None = None,
                      until_empty: bool = False) -> None:
        self.claim()
        n = self.recover()
        self.log(f"worker pid {self.pid} ready (db {self.db}); {n} interrupted step(s) marked failed")
        try:
            while not (should_stop and should_stop()):
                if self.run_once() is None:
                    if until_empty:
                        break
                    time.sleep(idle_s)
        finally:
            self._current = (None, None)
            self.store.set_worker_state(self.pid, None, None, heartbeat_at=0.0)   # reads as "down" at once


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m dreammachine.jobs.worker", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="runs/dreammachine.db")
    p.add_argument("--cancel-grace-s", type=float, default=120.0)
    p.add_argument("--until-empty", action="store_true", help="exit when the queue is empty (scripts, checks)")
    args = p.parse_args(argv)
    worker = Worker(Store(args.db), cancel_grace_s=args.cancel_grace_s, log=lambda s: print(s, flush=True))
    try:
        worker.serve_forever(until_empty=args.until_empty)
    except KeyboardInterrupt:
        print("worker stopped", flush=True)
    except WorkerBusy as e:
        print(str(e), flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
