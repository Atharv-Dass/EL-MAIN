"""DreamMachine API v1: exactly docs/API_CONTRACT.md (the frontend relies on nothing else).

    python -m dreammachine.serve                                   # API on http://127.0.0.1:8000 + the worker
    uvicorn dreammachine.api.app:app --host 127.0.0.1 --port 8000  # API only

All routes live under /api/v1. Every error uses the envelope {"error": {code, message, details}}. Times are ISO 8601
UTC strings ending in "Z"; missing values are null, never NaN. The API binds to 127.0.0.1 only and has no
authentication. It never imports torch (GPU facts come from nvidia-smi), so it holds no GPU memory.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
from collections import Counter
import json
import logging
import os
import platform
import re
import shutil
import time
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from fastapi import APIRouter, Body, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import __version__
from ..diagnosis.align import ReferenceTrace
from ..diagnosis.lltm import fit_lltm
from ..diagnosis.taxonomy import classify
from ..experiments.pipeline import ARMS, Config
from ..generator.families import FAMILIES
from ..generator.features import FEATURES
from ..generator.sampler import GenerationError, GenSpec, generate_many
from ..jobs import control
from ..jobs.errors import JobError
from ..jobs.requests import (
    ALLOWED_RATIOS, ALLOWED_SEEDS, DEFAULT_EXPLORE_BENCHMARKS, MIN_BENCHMARK_LIMIT, create_job_from_request,
    load_presets,
)
from ..jobs.results import RunData, build_results, compare_runs, error_shares, paper_eligibility
from ..jobs.steps import STAGE_LABELS, STAGES, arm_keys, label
from ..provenance import package_versions
from ..store.db import Store
from . import schemas as S

API_VERSION = "0.2.0"          # must equal "Contract version" in docs/API_CONTRACT.md (tests/test_api_contract.py)
PREFIX = "/api/v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORS = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000"]
HTTP_STATUS = {"not_found": 404, "conflict": 409, "not_ready": 409, "paper_protected": 409, "validation_error": 422,
               "paper_mode_locked": 422, "reuse_mismatch": 422, "invalid_feature": 422, "unknown_model": 422,
               "unknown_benchmark": 422, "internal_error": 500}
WORKER_STALE_S = 30
LOG_CHUNK = 64 * 1024
ARM_KEY = re.compile(r"^(real_only|untargeted|matched_control|targeted)_r\d+(\.\d+)?_s\d+$")
log = logging.getLogger("dreammachine.api")

FEATURE_LABELS = {
    "steps": ("Number of steps", "How many arithmetic operations the solution needs."),
    "log10_max": ("Size of numbers", "How large the biggest number in the calculation is (log10)."),
    "n_mul": ("Multiplications", "How many multiplications the solution needs."),
    "n_div": ("Divisions", "How many divisions the solution needs."),
    "n_carry": ("Carries and borrows", "How many column carries (addition) and borrows (subtraction) the problem "
                "needs."),
    "n_distractors": ("Distractors", "How many sentences contain numbers the solution does not need."),
    "has_merge": ("Two chains merged", "Whether the solution combines two separate chains of calculation."),
}
ERROR_LABELS = {
    "CORRECT": ("Correct", "The final answer matches the gold answer."),
    "FORMAT_ERROR": ("Format error", "No final answer could be read, or the right value was not given as the answer."),
    "ARITHMETIC_SLIP": ("Arithmetic slip", "A written equation is false."),
    "DISTRACTOR_USE": ("Used a distractor", "The work uses a number the problem did not need."),
    "PLAN_ERROR": ("Plan error", "The arithmetic is consistent, but the wrong calculation was set up."),
    "UNVERIFIABLE": ("Unverifiable", "A wrong answer with no equations to check."),
}
ARM_LABELS = {
    "base": ("Untrained model", "The model before any fine-tuning (results only)."),
    "real_only": ("Real only", "Real GSM8K training problems only, no synthetic data."),
    "untargeted": ("Untargeted", "Synthetic problems drawn at random, not aimed at any weakness."),
    "matched_control": ("Matched control", "Equally hard problems, hard for a different reason."),
    "targeted": ("Targeted", "Synthetic problems aimed at the model's biggest weakness."),
}


# ------------------------------------------------------------------ helpers
def iso(t: float | None) -> str | None:
    if not t:
        return None
    return dt.datetime.fromtimestamp(t, tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _nan_to_none(x: Any) -> Any:
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return None
    if isinstance(x, list):
        return [_nan_to_none(v) for v in x]
    if isinstance(x, dict):
        return {k: _nan_to_none(v) for k, v in x.items()}
    return x


def _read_json(path: Path) -> Any:
    return _nan_to_none(json.loads(path.read_text(encoding="utf-8")))


_ROW = re.compile(r"^\|\s*(C\w+)\s*\|(.+)\|\s*$")


def parse_components(path: Path) -> list[dict[str, str]]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line)
        if not m:
            continue
        cells = [c.strip() for c in m.group(2).split("|")]
        cells += [""] * (5 - len(cells))
        out.append({"id": m.group(1), "component": cells[0], "path": cells[1].strip("`"),
                    "status": cells[2], "verify": cells[3].strip("`"), "notes": cells[4]})
    return out


def read_log_chunk(path: Path, offset: int, max_bytes: int = LOG_CHUNK) -> dict:
    """Up to max_bytes from byte `offset`, cut back to the last complete UTF-8 character, so a character is never
    split between two chunks; next_offset is the byte right after the returned text (contract §3)."""
    if not path.exists():
        return {"text": "", "next_offset": offset, "eof": True}
    size = path.stat().st_size
    offset = min(max(offset, 0), size)
    with path.open("rb") as fh:
        fh.seek(offset)
        data = fh.read(max_bytes)
    for cut in range(min(4, len(data) + 1)):      # a UTF-8 character is at most 4 bytes
        try:
            text, used = data[:len(data) - cut].decode("utf-8"), len(data) - cut
            break
        except UnicodeDecodeError:
            continue
    else:                                            # not UTF-8 at all: never stall the reader
        text, used = data.decode("utf-8", errors="replace"), len(data)
    nxt = offset + used
    return {"text": text, "next_offset": nxt, "eof": nxt >= size}


def _folder_bytes(path: Path) -> int:
    return control.folder_bytes(path) if path and path.exists() else 0


def _nvidia_smi_gpu() -> dict | None:
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10,
                             check=True).stdout.strip().splitlines()
        name, total, free = [x.strip() for x in out[0].split(",")]
        return {"name": name, "vram_total_mb": float(total), "vram_free_mb": float(free)}
    except Exception:  # noqa: BLE001 - no NVIDIA GPU or driver
        return None


# ------------------------------------------------------------------ app
def create_app(store: Store | None = None, components_path: Path | None = None,
               presets: dict[str, Config] | None = None, known_models: list[str] | None = None,
               runs_root: str | Path = "runs/pipelines", allow_local_models: bool = False) -> FastAPI:
    app = FastAPI(title="DreamMachine API", version=API_VERSION,
                  description="Backend API for DreamMachine. Contract: docs/API_CONTRACT.md.")
    app.state.api_version = API_VERSION
    db = store or Store(os.environ.get("DREAMMACHINE_DB", "runs/dreammachine.db"))
    comp_path = components_path or REPO_ROOT / "docs" / "COMPONENTS.md"
    _presets = presets
    runs_root = Path(runs_root)
    origins = [o.strip() for o in os.environ.get("DREAMMACHINE_CORS_ORIGINS", "").split(",") if o.strip()] or DEFAULT_CORS
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])
    disk_cache: dict[str, Any] = {}

    def get_presets() -> dict[str, Config]:
        nonlocal _presets
        if _presets is None:
            _presets = load_presets()
        return _presets

    def models() -> list[str]:
        if known_models is not None:
            return known_models
        from ..models.catalog import model_ids
        return model_ids()

    # ------------------------------------------------------------ errors
    def envelope(status: int, code: str, message: str, details: dict | None = None) -> JSONResponse:
        return JSONResponse(status_code=status, content={"error": {"code": code, "message": message,
                                                                   "details": details}})

    @app.exception_handler(JobError)
    async def _job_error(request: Request, e: JobError):
        return envelope(HTTP_STATUS.get(e.code, 422), e.code, e.message, e.details)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, e: RequestValidationError):
        fields = [{"loc": list(err.get("loc", [])), "msg": err.get("msg", "")} for err in e.errors()]
        return envelope(422, "validation_error", fields[0]["msg"] if fields else "invalid request", {"fields": fields})

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, e: StarletteHTTPException):
        code = {404: "not_found", 405: "validation_error", 409: "conflict"}.get(e.status_code, "validation_error")
        return envelope(e.status_code, code, str(e.detail))

    @app.exception_handler(Exception)
    async def _crash(request: Request, e: Exception):
        log.exception("unhandled error on %s", request.url.path)
        return envelope(500, "internal_error", f"Unexpected server error ({type(e).__name__}); see the server log.")

    # -------------------------------------------------------- job views
    def worker_alive() -> tuple[bool, dict | None]:
        st = db.get_worker_state()
        return bool(st and st["heartbeat_at"] and time.time() - st["heartbeat_at"] < WORKER_STALE_S), st

    def get_job(job_id: str, kind: str) -> dict:
        job = db.get_job(job_id)
        if job is None or job["kind"] != kind:
            raise JobError("not_found", f"no {kind.replace('_', ' ')} {job_id!r}")
        return job

    def step_view(s: dict) -> dict:
        done, total = s["progress_done"] or 0, s["progress_total"] or 0
        dur = (s["finished_at"] - s["started_at"]) if s["finished_at"] and s["started_at"] else None
        m = s["metrics"]
        return {"index": s["idx"], "key": s["key"], "stage": s["stage"], "label": label(s["stage"], s["params"]),
                "status": s["status"],
                "progress": {"done": done, "total": total, "unit": s["progress_unit"],
                             "fraction": (done / total) if total else (1.0 if s["status"] in ("done", "skipped") else 0.0)},
                "started_at": iso(s["started_at"]), "finished_at": iso(s["finished_at"]),
                "duration_s": round(dur, 3) if dur is not None else None, "run_ids": s["run_ids"] or [],
                "metrics": None if m is None else {"load_s": m.get("load_s"), "run_s": m.get("run_s"),
                                                   "gpu": m.get("gpu")},
                "error": s["error"]}

    def stage_durations(steps: list[dict]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for s in steps:
            if s["status"] == "done" and s["started_at"] and s["finished_at"]:
                out.setdefault(s["stage"], []).append(s["finished_at"] - s["started_at"])
        return out

    def eta(job: dict, steps: list[dict]) -> float | None:
        """Rough remaining seconds (PLAN.md §7.4): per unfinished step, the mean duration of finished steps of the
        same stage in this job, else in the latest finished job with the same model. None without data."""
        if job["status"] not in ("queued", "running"):
            return None
        here = stage_durations(steps)
        fallback: dict[str, list[float]] = {}
        for other in db.list_jobs(status="done", limit=20):
            if other["id"] != job["id"] and (other["resolved"] or {}).get("model") == job["resolved"]["model"]:
                fallback = stage_durations(db.get_steps(other["id"]))
                break
        total, known, now = 0.0, False, time.time()
        for s in steps:
            if s["status"] not in ("pending", "running"):
                continue
            d = here.get(s["stage"]) or fallback.get(s["stage"])
            if not d:
                continue
            avg = float(np.mean(d))
            total += max(avg - (now - s["started_at"]), 0.0) if s["status"] == "running" and s["started_at"] else avg
            known = True
        return round(total, 1) if known else None

    def target_of(job: dict) -> dict | None:
        p = Path(job["output_dir"] or ".") / "target.json"
        if not p.exists():
            return None
        t = _read_json(p)
        return {"strategy": t["strategy"], "feature": t.get("feature"), "threshold": t.get("threshold")}

    def job_view(job: dict) -> dict:
        steps = db.get_steps(job["id"])
        n_done = sum(s["status"] in ("done", "skipped") for s in steps)
        queued = [j["id"] for j in db.list_jobs(status="queued", oldest_first=True)]
        r = job["resolved"] or {}
        view = {
            "id": job["id"], "kind": job["kind"], "name": job["name"], "mode": job["mode"], "status": job["status"],
            "created_at": iso(job["created_at"]), "started_at": iso(job["started_at"]),
            "finished_at": iso(job["finished_at"]),
            "queue_position": queued.index(job["id"]) + 1 if job["id"] in queued else None,
            "cancel_requested": job["cancel_requested"], "request": job["request"] or {},
            "resolved": {"model": r.get("model"), "benchmarks": r.get("benchmarks", []),
                         "benchmark_limit": r.get("benchmark_limit"), "arms": r.get("arms", []),
                         "ratios": r.get("ratios", []), "seeds": r.get("seeds", []),
                         "prompt_version": r.get("prompt_version"), "config_hash": job["config_hash"],
                         "baseline_hash": job["baseline_hash"], "diagnosis_hash": job["diagnosis_hash"]},
            "progress": {"done_steps": n_done, "total_steps": len(steps),
                         "fraction": n_done / len(steps) if steps else 0.0},
            "eta_s": eta(job, steps), "output_dir": job["output_dir"] or "",
            "output_bytes": _folder_bytes(Path(job["output_dir"])) if job["output_dir"] else 0,
            "current_step": next((s["key"] for s in steps if s["status"] == "running"), None),
            "steps": [step_view(s) for s in steps], "reuse_from": job["reuse_from"],
            "warnings": job["warnings"] or [], "error": job["error"],
        }
        if job["kind"] == "pipeline":
            view["target"] = target_of(job)
            view["paper_eligible"] = paper_eligibility(db, job)[0]
        return view

    def summary(job: dict) -> dict:
        v = job_view(job)
        keys = ("id", "name", "mode", "status", "created_at", "finished_at", "progress")
        out = {k: v[k] for k in keys} | {"model": (job["resolved"] or {}).get("model")}
        if job["kind"] == "pipeline":
            out |= {"target": v["target"], "paper_eligible": v["paper_eligible"]}
        return out

    def page(items: list, total: int, limit: int, offset: int) -> dict:
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    def create(kind: str, body: dict) -> dict:
        job_id = create_job_from_request(db, kind, body, presets=get_presets(), known_models=models(),
                                         runs_root=runs_root, allow_local_models=allow_local_models)
        if not worker_alive()[0]:
            db.add_job_warning(job_id, "worker_not_running")
        return job_view(db.get_job(job_id))

    def step_of(job: dict, stage: str, statuses=("done", "skipped")) -> dict | None:
        return next((s for s in db.get_steps(job["id"]) if s["stage"] == stage and s["status"] in statuses), None)

    def baseline_of(job: dict) -> dict:
        from ..benchmarks import registry

        s = step_of(job, "baseline")
        if s is None or not s["run_ids"]:
            raise JobError("not_ready", "the baseline step has not finished yet")
        rd = RunData(db, s["run_ids"][0])
        benches = []
        for b in (job["resolved"] or {}).get("benchmarks", []):
            if rd.accuracy(b) is None:
                continue
            wrong = Counter(r["error_type"] for r in db.get_responses(s["run_ids"][0], source=b, correct=False))
            benches.append({"name": b, "kind": registry.get(b).kind, "n": len(rd.items[b]),
                            "accuracy": rd.accuracy(b), "error_distribution": error_shares(wrong)})
        return {"model": job["resolved"]["model"], "reused_from": job["reuse_from"] if s["status"] == "skipped" else None,
                "run_ids": s["run_ids"], "benchmarks": benches}

    def diagnosis_of(job: dict) -> dict:
        s = step_of(job, "diagnose")
        path = Path(job["output_dir"] or ".") / "diagnosis.json"
        if s is None or not path.exists():
            raise JobError("not_ready", "the diagnose step has not finished yet")
        d = _read_json(path)
        lltm, summ = d.get("lltm") or {}, d.get("summary") or {}
        weak = [{"rank": i + 1, "feature": w["feature"], "label": FEATURE_LABELS.get(w["feature"], (w["feature"],))[0],
                 "eta": w["eta"], "ci_low": w.get("ci_low"), "ci_high": w.get("ci_high"), "range": w["range"],
                 "effect": w["effect"], "significant": bool(w["significant"])}
                for i, w in enumerate(d.get("weaknesses", []))]
        t = None
        tpath = Path(job["output_dir"]) / "target.json"
        if tpath.exists():
            tj = _read_json(tpath)
            t = {"strategy": tj["strategy"], "feature": tj.get("feature"), "threshold": tj.get("threshold"),
                 "significant": bool(tj.get("significant"))}
        return {"run_id": d["run_id"], "probe_accuracy": summ.get("accuracy", 0.0), "theta": lltm.get("theta"),
                "fit": {"converged": bool(lltm.get("converged")), "mcfadden_r2": lltm.get("mcfadden_r2"),
                        "n_items": int(lltm.get("n_items", 0)), "n_obs": int(lltm.get("n_obs", 0))},
                "weaknesses": weak, "target": t, "error_distribution": summ.get("error_distribution", {})}

    def log_of(job: dict, offset: int) -> dict:
        return read_log_chunk(Path(job["output_dir"] or ".") / "logs" / "job.log", offset)

    api = APIRouter(prefix=PREFIX)

    # ------------------------------------------------- system and catalog
    @api.get("/health", response_model=S.Health)
    def health():
        return {"status": "ok", "version": __version__, "api_version": API_VERSION}

    @api.get("/system", response_model=S.SystemInfo)
    def system():
        torch_ok = importlib.util.find_spec("torch") is not None
        g = _nvidia_smi_gpu() if torch_ok else None
        alive, st = worker_alive()
        cur_step = None
        if st and st["current_job_id"] is not None and st["current_step_idx"] is not None:
            s = db.get_step(st["current_job_id"], st["current_step_idx"])
            cur_step = s["key"] if s else None
        root = runs_root.resolve()
        probe = root
        while not probe.exists():
            probe = probe.parent
        now = time.time()
        if now - disk_cache.get("t", 0) > 60:
            disk_cache.update(t=now, runs_bytes=_folder_bytes(root.parent))
        ver = package_versions()
        return {"gpu": {"available": g is not None, "name": g and g["name"], "vram_total_mb": g and g["vram_total_mb"],
                        "vram_free_mb": g and g["vram_free_mb"]},
                "worker": {"alive": alive, "pid": st and st["pid"], "heartbeat_at": iso(st and st["heartbeat_at"]),
                           "current_job_id": (st["current_job_id"] if st else None) if alive else None,
                           "current_step": cur_step if alive else None},
                "versions": {"python": platform.python_version(), **{k: ver[k] for k in
                                                                     ("torch", "transformers", "peft", "bitsandbytes")}},
                "platform": platform.platform(),
                "disk": {"free_gb": round(shutil.disk_usage(probe).free / 1e9, 1), "runs_bytes": disk_cache["runs_bytes"],
                         "min_free_gb": float(get_presets()["explore"].preflight.get("min_free_gb", 10))}}

    @api.get("/meta", response_model=S.Meta)
    def meta():
        return {"features": [{"name": f, "label": FEATURE_LABELS[f][0], "description": FEATURE_LABELS[f][1]}
                             for f in FEATURES],
                "error_types": [{"name": k, "label": v[0], "description": v[1]} for k, v in ERROR_LABELS.items()],
                "arms": [{"name": k, "label": v[0], "description": v[1]} for k, v in ARM_LABELS.items()],
                "stages": [{"name": s, "label": STAGE_LABELS[s]} for s in STAGES],
                "families": {n: {"split": f.split, "unit": f.unit} for n, f in FAMILIES.items()}}

    @api.get("/models", response_model=list[S.ModelInfo])
    def list_models():
        from ..models.catalog import load_models
        return load_models()

    @api.get("/benchmarks", response_model=list[S.BenchmarkInfo])
    def list_benchmarks():
        from ..benchmarks import registry
        return [b.info() for b in registry.available()]

    @api.get("/presets", response_model=S.Presets)
    def presets_():
        ex, pp = get_presets()["explore"], get_presets()["paper"]
        ratios = [float(r) for r in pp.data.get("ratios", [3])]
        main_ratio = float(pp.data.get("main_ratio", ratios[0]))
        n_keys = len(arm_keys(list(ARMS), ratios, main_ratio, list(pp.seeds), paper=True))
        return {"explore": {"defaults": {"benchmarks": DEFAULT_EXPLORE_BENCHMARKS,
                                         "benchmark_limit": ex.eval.get("benchmark_limit"),
                                         "target": {"strategy": "auto"}, "arms": ["targeted", "matched_control"],
                                         "ratio": int(ex.data.get("main_ratio", 3)), "seeds": list(ex.seeds)},
                            "allowed": {"ratio": list(ALLOWED_RATIOS), "seeds": list(ALLOWED_SEEDS),
                                        "max_seeds": len(ALLOWED_SEEDS), "min_benchmark_limit": MIN_BENCHMARK_LIMIT}},
                "paper": {"description": "Fixed protocol: 4 arms incl. matched control, 3 seeds, ratios "
                                         f"{'/'.join(f'{r:g}' for r in ratios)} for targeted, equal token budget.",
                          "fixed": {"arms": list(ARMS), "seeds": list(pp.seeds), "main_ratio": main_ratio,
                                    "targeted_ratios": ratios},
                          "settable": ["name", "model"], "n_steps": 5 + 2 * n_keys + 1}}

    @api.get("/components", response_model=list[S.Component])
    def components():
        return parse_components(comp_path)

    @api.get("/queue", response_model=S.Queue)
    def queue():
        alive, st = worker_alive()
        current = None
        running = db.list_jobs(status="running", limit=1)
        if running:
            j = running[0]
            step = next((s["key"] for s in db.get_steps(j["id"]) if s["status"] == "running"), None)
            current = {"job_id": j["id"], "kind": j["kind"], "name": j["name"], "current_step": step}
        queued = db.list_jobs(status="queued", oldest_first=True)
        return {"current": current, "queued": [{"job_id": j["id"], "kind": j["kind"], "name": j["name"],
                                                "position": i + 1} for i, j in enumerate(queued)]}

    # ----------------------------------------------------------- pipelines
    @api.post("/pipelines", response_model=S.Pipeline, status_code=202)
    def create_pipeline(body: dict[str, Any] = Body(...)):
        return create("pipeline", body)

    @api.get("/pipelines", response_model=S.Page[S.PipelineSummary])
    def list_pipelines(mode: S.Mode | None = None, status: S.JobStatus | None = None,
                       limit: Annotated[int, Query(ge=1, le=500)] = 50, offset: Annotated[int, Query(ge=0)] = 0):
        jobs = db.list_jobs(kind="pipeline", mode=mode, status=status, limit=limit, offset=offset)
        return page([summary(j) for j in jobs], db.count_jobs(kind="pipeline", mode=mode, status=status), limit, offset)

    @api.get("/pipelines/{pipeline_id}", response_model=S.Pipeline)
    def get_pipeline(pipeline_id: str):
        return job_view(get_job(pipeline_id, "pipeline"))

    @api.post("/pipelines/{pipeline_id}/cancel", response_model=S.Pipeline)
    def cancel_pipeline(pipeline_id: str):
        get_job(pipeline_id, "pipeline")
        return job_view(control.cancel_job(db, pipeline_id))

    @api.post("/pipelines/{pipeline_id}/resume", response_model=S.Pipeline)
    def resume_pipeline(pipeline_id: str):
        get_job(pipeline_id, "pipeline")
        return job_view(control.resume_job(db, pipeline_id))

    @api.delete("/pipelines/{pipeline_id}", response_model=S.DeleteResult)
    def delete_pipeline(pipeline_id: str):
        get_job(pipeline_id, "pipeline")
        return control.delete_job(db, pipeline_id)

    @api.get("/pipelines/{pipeline_id}/baseline", response_model=S.BaselineResult)
    def pipeline_baseline(pipeline_id: str):
        return baseline_of(get_job(pipeline_id, "pipeline"))

    @api.get("/pipelines/{pipeline_id}/diagnosis", response_model=S.DiagnosisResult)
    def pipeline_diagnosis(pipeline_id: str):
        return diagnosis_of(get_job(pipeline_id, "pipeline"))

    def manifest_of(job: dict) -> dict:
        path = Path(job["output_dir"] or ".") / "data" / "manifest.json"
        if step_of(job, "build_data", ("done",)) is None or not path.exists():
            raise JobError("not_ready", "the build_data step has not finished yet")
        return _read_json(path)

    def arm_file(job: dict, arm_key: str) -> Path:
        m = manifest_of(job)
        if not ARM_KEY.match(arm_key) or arm_key not in m["arms"]:
            raise JobError("not_found", f"no training set {arm_key!r} in this pipeline")
        return Path(job["output_dir"]) / "data" / f"{arm_key}.jsonl"

    @api.get("/pipelines/{pipeline_id}/data", response_model=S.DataResult)
    def pipeline_data(pipeline_id: str):
        job = get_job(pipeline_id, "pipeline")
        m = manifest_of(job)
        arms, matching = [], []
        for k, v in m["arms"].items():
            if k.startswith("match_s"):
                matching.append({"seed": int(k[len("match_s"):]), **v})
            elif ARM_KEY.match(k):
                arm, r, sd = k.rsplit("_", 2)
                arms.append({"key": k, "arm": arm, "ratio": float(r[1:]), "seed": int(sd[1:]), **v})
        return {"target": m.get("target"), "threshold": m.get("threshold"), "budget_tokens": m["budget_tokens"],
                "arms": arms, "matching": matching, "warnings": m.get("warnings", [])}

    @api.get("/pipelines/{pipeline_id}/data/{arm_key}/samples", response_model=S.Page[S.TrainingSample])
    def pipeline_samples(pipeline_id: str, arm_key: str, limit: Annotated[int, Query(ge=1, le=50)] = 5,
                         offset: Annotated[int, Query(ge=0)] = 0, source: Literal["synthetic", "real"] | None = None):
        job = get_job(pipeline_id, "pipeline")
        path = arm_file(job, arm_key)
        seed = int(arm_key.rsplit("_s", 1)[1])
        fpath = path.parent / f"features_s{seed}.json"
        feats = json.loads(fpath.read_text(encoding="utf-8")) if fpath.exists() else {}
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if source is not None:
            rows = [r for r in rows if r["source"].startswith("synthetic") == (source == "synthetic")]
        items = [{"prompt": r["prompt"], "completion": r["completion"], "source": r["source"],
                  "features": feats.get(r["id"]) if r["source"].startswith("synthetic") else None}
                 for r in rows[offset:offset + limit]]
        return page(items, len(rows), limit, offset)

    @api.get("/pipelines/{pipeline_id}/data/{arm_key}/download", response_class=FileResponse,
             responses={200: {"content": {"application/x-ndjson": {}}}})
    def pipeline_download(pipeline_id: str, arm_key: str):
        path = arm_file(get_job(pipeline_id, "pipeline"), arm_key)
        return FileResponse(path, media_type="application/x-ndjson", filename=f"{arm_key}.jsonl")

    @api.get("/pipelines/{pipeline_id}/training", response_model=list[S.TrainingResult])
    def pipeline_training(pipeline_id: str):
        job = get_job(pipeline_id, "pipeline")
        out = []
        for s in db.get_steps(job["id"]):
            if s["stage"] != "train":
                continue
            curve = (s["metrics"] or {}).get("loss_curve") or []
            dur = (s["finished_at"] - s["started_at"]) if s["finished_at"] and s["started_at"] else None
            p = s["params"]
            out.append({"key": s["key"].split(":", 1)[1], "arm": p["arm"], "ratio": p["ratio"], "seed": p["seed"],
                        "status": s["status"], "step": s["progress_done"] or 0, "total_steps": s["progress_total"] or 0,
                        "final_loss": curve[-1]["loss"] if curve and s["status"] == "done" else None,
                        "duration_s": round(dur, 3) if dur is not None else None, "loss_curve": curve})
        return out

    @api.get("/pipelines/{pipeline_id}/results", response_model=S.PipelineResults)
    def pipeline_results(pipeline_id: str):
        job = get_job(pipeline_id, "pipeline")
        path = Path(job["output_dir"] or ".") / "results.json"
        if step_of(job, "compare", ("done",)) is not None and path.exists():
            return _read_json(path)
        return _nan_to_none(build_results(db, pipeline_id))

    @api.get("/pipelines/{pipeline_id}/report", response_class=PlainTextResponse,
             responses={200: {"content": {"text/markdown": {}}}})
    def pipeline_report(pipeline_id: str):
        job = get_job(pipeline_id, "pipeline")
        path = Path(job["output_dir"] or ".") / "report.md"
        if step_of(job, "compare", ("done",)) is None or not path.exists():
            raise JobError("not_ready", "the compare step has not finished yet")
        return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="text/markdown; charset=utf-8")

    @api.get("/pipelines/{pipeline_id}/log", response_model=S.LogChunk)
    def pipeline_log(pipeline_id: str, offset: Annotated[int, Query(ge=0)] = 0):
        return log_of(get_job(pipeline_id, "pipeline"), offset)

    # ------------------------------------------------------ benchmark runs
    @api.post("/benchmark-runs", response_model=S.BenchmarkRun, status_code=202)
    def create_benchmark_run(body: dict[str, Any] = Body(...)):
        return create("benchmark_run", body)

    @api.get("/benchmark-runs", response_model=S.Page[S.BenchmarkRunSummary])
    def list_benchmark_runs(status: S.JobStatus | None = None, limit: Annotated[int, Query(ge=1, le=500)] = 50,
                            offset: Annotated[int, Query(ge=0)] = 0):
        jobs = db.list_jobs(kind="benchmark_run", status=status, limit=limit, offset=offset)
        return page([summary(j) for j in jobs], db.count_jobs(kind="benchmark_run", status=status), limit, offset)

    @api.get("/benchmark-runs/{job_id}", response_model=S.BenchmarkRun)
    def get_benchmark_run(job_id: str):
        return job_view(get_job(job_id, "benchmark_run"))

    @api.post("/benchmark-runs/{job_id}/cancel", response_model=S.BenchmarkRun)
    def cancel_benchmark_run(job_id: str):
        get_job(job_id, "benchmark_run")
        return job_view(control.cancel_job(db, job_id))

    @api.delete("/benchmark-runs/{job_id}", response_model=S.DeleteResult)
    def delete_benchmark_run(job_id: str):
        get_job(job_id, "benchmark_run")
        return control.delete_job(db, job_id)

    @api.get("/benchmark-runs/{job_id}/baseline", response_model=S.BaselineResult)
    def benchmark_run_baseline(job_id: str):
        return baseline_of(get_job(job_id, "benchmark_run"))

    @api.get("/benchmark-runs/{job_id}/diagnosis", response_model=S.DiagnosisResult)
    def benchmark_run_diagnosis(job_id: str):
        return diagnosis_of(get_job(job_id, "benchmark_run"))

    @api.get("/benchmark-runs/{job_id}/log", response_model=S.LogChunk)
    def benchmark_run_log(job_id: str, offset: Annotated[int, Query(ge=0)] = 0):
        return log_of(get_job(job_id, "benchmark_run"), offset)

    # ------------------------------------------------------- runs, compare
    def run_view(r: dict) -> dict:
        return {"id": r["id"], "name": r["name"], "kind": r["kind"], "status": r["status"], "mode": r.get("mode"),
                "job_id": r.get("job_id"), "config": _nan_to_none(r["config"]),
                "metrics": _nan_to_none(r["metrics"]), "created_at": iso(r["created_at"]),
                "finished_at": iso(r["finished_at"])}

    def get_run(run_id: str) -> dict:
        r = db.get_run(run_id)
        if r is None:
            raise JobError("not_found", f"no run {run_id!r}")
        return r

    @api.get("/runs", response_model=S.Page[S.Run])
    def list_runs(kind: S.RunKind | None = None, job_id: str | None = None,
                  limit: Annotated[int, Query(ge=1, le=500)] = 50, offset: Annotated[int, Query(ge=0)] = 0):
        rows = db.list_runs(kind, job_id=job_id, limit=limit, offset=offset)
        return page([run_view(r) for r in rows], db.count_runs(kind, job_id=job_id), limit, offset)

    @api.get("/runs/{run_id}", response_model=S.RunWithArtifacts)
    def run_detail(run_id: str):
        return run_view(get_run(run_id)) | {"artifacts": db.list_artifacts(run_id)}

    @api.get("/runs/{run_id}/responses", response_model=S.Page[S.ResponseRecord])
    def run_responses(run_id: str, correct: bool | None = None, error_type: S.ErrorTypeName | None = None,
                      source: str | None = None, limit: Annotated[int, Query(ge=1, le=500)] = 50,
                      offset: Annotated[int, Query(ge=0)] = 0):
        get_run(run_id)
        rows = db.get_responses(run_id, limit=limit, offset=offset, correct=correct, error_type=error_type,
                                source=source)
        return page([_nan_to_none(r) for r in rows],
                    db.count_responses(run_id, correct=correct, error_type=error_type, source=source), limit, offset)

    @api.get("/runs/{run_id}/artifacts/{key}")
    def run_artifact(run_id: str, key: str) -> Any:
        get_run(run_id)
        if key not in db.list_artifacts(run_id):
            raise JobError("not_found", f"run {run_id} has no artifact {key!r}")
        return _nan_to_none(db.get_artifact(run_id, key))

    @api.get("/compare", response_model=S.CompareResult)
    def compare(before: str, after: str):
        return _nan_to_none(compare_runs(db, before, after))

    # --------------------------------------------------------------- tools
    @api.post("/tools/generate")
    def tools_generate(req: S.GenerateRequest) -> list[dict]:
        try:
            spec = GenSpec(steps=req.steps, digits=req.digits, op_weights=req.op_weights,
                           n_distractors=req.n_distractors, merge=req.merge, split=req.split, family=req.family)
            return [it.to_dict() for it in generate_many(spec, req.n, seed=req.seed)]
        except (ValueError, KeyError, GenerationError) as e:
            raise JobError("validation_error", str(e))

    @api.post("/tools/classify", response_model=S.ResponseDiagnosis)
    def tools_classify(req: S.ClassifyRequest):
        if req.item is not None:
            try:
                from ..generator.sampler import Item
                trace = ReferenceTrace.from_item(Item.from_dict(req.item))
            except (TypeError, KeyError) as e:
                raise JobError("validation_error", f"invalid item: {e}")
        elif req.question and req.reference_solution:
            try:
                trace = ReferenceTrace.from_gsm8k(req.question, req.reference_solution)
            except (ValueError, ZeroDivisionError) as e:
                raise JobError("validation_error", f"invalid reference solution: {e}")
        else:
            raise JobError("validation_error", "give either `item` or `question` + `reference_solution`")
        return classify(req.response, trace).to_dict()

    @api.post("/tools/lltm-fit")
    def tools_lltm_fit(req: S.LLTMRequest) -> dict:
        try:
            r = fit_lltm(np.array(req.Q), np.array(req.successes),
                         None if req.trials is None else np.array(req.trials), req.feature_names, l2=req.l2)
        except ValueError as e:
            raise JobError("validation_error", str(e))
        return _nan_to_none(r.to_dict())

    app.include_router(api)

    def custom_openapi() -> dict:
        """Document the create bodies with the contract's request types (they are validated by the job system,
        which returns the contract's error codes such as paper_mode_locked)."""
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)
        comps = schema.setdefault("components", {}).setdefault("schemas", {})
        for m in (S.CreateExplorePipeline, S.CreatePaperPipeline, S.CreateBenchmarkRun):
            js = m.model_json_schema(ref_template="#/components/schemas/{model}")
            comps.update(js.pop("$defs", {}))
            comps[m.__name__] = js
        bodies = {f"{PREFIX}/pipelines": {"oneOf": [{"$ref": "#/components/schemas/CreateExplorePipeline"},
                                                    {"$ref": "#/components/schemas/CreatePaperPipeline"}]},
                  f"{PREFIX}/benchmark-runs": {"$ref": "#/components/schemas/CreateBenchmarkRun"}}
        for path, body in bodies.items():
            schema["paths"][path]["post"]["requestBody"] = {"required": True,
                                                            "content": {"application/json": {"schema": body}}}
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi
    return app


def __getattr__(name: str):  # lazily build `app` so importing this module has no side effects
    if name == "app":
        return create_app()
    raise AttributeError(name)
