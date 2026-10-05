"""From a request (API body or CLI flags) to a queued job (PLAN.md §5-6, docs/API_CONTRACT.md §6.2-6.3).

    validate_*()   check the request; raise JobError with a contract error code
    resolve()      the full experiment Config for the job, plus the summary the API shows (`resolved`)
    *_hash()       reuse hashes (PLAN.md §6.3)
    create_job_from_request()   validate + resolve + reuse + store the job and its steps (status queued)
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..experiments.pipeline import ARMS, Config, benchmark_options, default_benchmarks
from ..generator.features import FEATURES
from ..provenance import REPO_ROOT, hf_revision
from ..store.db import Store
from .errors import JobError
from .steps import build_steps

PRESET_PATHS = {"explore": REPO_ROOT / "configs" / "explore.yaml", "paper": REPO_ROOT / "configs" / "main.yaml"}
DEFAULT_EXPLORE_BENCHMARKS = ["gsm8k", "gsm_symbolic:main", "probes_train_families", "probes_heldout_families"]
ALLOWED_RATIOS = (1, 3, 9)
ALLOWED_SEEDS = (0, 1, 2)
MIN_BENCHMARK_LIMIT = 10
EXPLORE_KEYS = {"mode", "name", "model", "benchmarks", "benchmark_limit", "target", "arms", "ratio", "seeds",
                "reuse_from"}
PAPER_KEYS = {"mode", "name", "model"}
BENCHMARK_RUN_KEYS = {"name", "model", "benchmarks", "benchmark_limit", "include_diagnosis"}


def load_presets() -> dict[str, Config]:
    return {mode: Config.load(path) for mode, path in PRESET_PATHS.items()}


def known_model_ids() -> list[str]:
    from ..models.catalog import model_ids

    return model_ids()


# ------------------------------------------------------------------ validation
def _bad(field: str, msg: str, code: str = "validation_error") -> JobError:
    return JobError(code, msg, {"fields": [{"loc": ["body", field], "msg": msg}]})


def _check_model(req: dict, known_models: list[str], allow_local: bool) -> None:
    model = req.get("model")
    if not isinstance(model, str) or not model:
        raise _bad("model", "model is required")
    if model not in known_models and not (allow_local and Path(model).is_dir()):
        raise _bad("model", f"unknown model {model!r}; pick one from GET /models", "unknown_model")


def _check_name(req: dict) -> None:
    name = req.get("name")
    if name is not None and (not isinstance(name, str) or not 1 <= len(name) <= 80):
        raise _bad("name", "name must be 1-80 characters")


def _check_benchmarks(req: dict, allow_jsonl: bool) -> None:
    from ..benchmarks import registry

    b = req.get("benchmarks")
    if b is None:
        return
    if not isinstance(b, list) or not b or len(set(b)) != len(b):
        raise _bad("benchmarks", "benchmarks must be a non-empty list without duplicates")
    names = {x.name for x in registry.available()}
    for n in b:
        if not (n in names or (allow_jsonl and isinstance(n, str) and n.startswith("jsonl:") and len(n) > 6)):
            raise _bad("benchmarks", f"unknown benchmark {n!r}; pick from GET /benchmarks", "unknown_benchmark")


def _check_limit(req: dict) -> None:
    lim = req.get("benchmark_limit")
    if lim is not None and (not isinstance(lim, int) or isinstance(lim, bool) or lim < MIN_BENCHMARK_LIMIT):
        raise _bad("benchmark_limit", f"benchmark_limit must be an integer >= {MIN_BENCHMARK_LIMIT}, or null")


def validate_pipeline_request(req: dict, known_models: list[str], allow_local_models: bool = False,
                              allow_jsonl: bool = False) -> dict:
    """Check a CreatePipeline body (contract §6.2). Returns a normalised copy with defaults filled in."""
    if not isinstance(req, dict):
        raise JobError("validation_error", "the request body must be a JSON object")
    mode = req.get("mode")
    if mode not in ("explore", "paper"):
        raise _bad("mode", "mode must be 'explore' or 'paper'")
    _check_model(req, known_models, allow_local_models)
    _check_name(req)
    if mode == "paper":
        extra = sorted(set(req) - PAPER_KEYS)
        if extra:
            raise JobError("paper_mode_locked", f"paper mode only accepts mode, name and model; got {extra}",
                           {"fields": [{"loc": ["body", k], "msg": "not allowed in paper mode"} for k in extra]})
        return {"mode": "paper", "model": req["model"], "name": req.get("name")}
    extra = sorted(set(req) - EXPLORE_KEYS)
    if extra:
        raise _bad(extra[0], f"unknown field(s) {extra}")
    _check_benchmarks(req, allow_jsonl)
    _check_limit(req)
    target = req.get("target") or {"strategy": "auto"}
    if not isinstance(target, dict) or target.get("strategy") not in ("auto", "feature", "untargeted"):
        raise _bad("target", "target.strategy must be 'auto', 'feature' or 'untargeted'")
    if target["strategy"] == "feature":
        if target.get("feature") not in FEATURES:
            raise _bad("target", f"target.feature must be one of {list(FEATURES)}", "invalid_feature")
    elif set(target) - {"strategy"}:
        raise _bad("target", f"target.strategy {target['strategy']!r} takes no other fields")
    arms = req.get("arms", ["targeted", "matched_control"])
    if (not isinstance(arms, list) or not 1 <= len(arms) <= 4 or len(set(arms)) != len(arms)
            or any(a not in ARMS for a in arms)):
        raise _bad("arms", f"arms must be 1-4 unique values from {list(ARMS)}")
    if target["strategy"] == "untargeted" and set(arms) - {"untargeted", "real_only"}:
        raise _bad("arms", "target strategy 'untargeted' allows only the arms 'untargeted' and 'real_only' "
                           "(the matched control needs a target)")
    ratio = req.get("ratio", 3)
    if ratio not in ALLOWED_RATIOS or isinstance(ratio, bool):
        raise _bad("ratio", f"ratio must be one of {list(ALLOWED_RATIOS)}")
    seeds = req.get("seeds", [0])
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 3 or len(set(seeds)) != len(seeds)
            or any(s not in ALLOWED_SEEDS or isinstance(s, bool) for s in seeds)):
        raise _bad("seeds", "seeds must be 1-3 unique values from {0, 1, 2}")
    reuse = req.get("reuse_from")
    if reuse is not None and not isinstance(reuse, str):
        raise _bad("reuse_from", "reuse_from must be a job id or null")
    return {"mode": "explore", "model": req["model"], "name": req.get("name"),
            "benchmarks": req.get("benchmarks"), "benchmark_limit": req.get("benchmark_limit", "default"),
            "target": dict(target), "arms": list(arms), "ratio": int(ratio), "seeds": sorted(seeds),
            "reuse_from": reuse}


def validate_benchmark_run_request(req: dict, known_models: list[str], allow_local_models: bool = False,
                                   allow_jsonl: bool = False) -> dict:
    """Check a CreateBenchmarkRun body (contract §6.3)."""
    if not isinstance(req, dict):
        raise JobError("validation_error", "the request body must be a JSON object")
    extra = sorted(set(req) - BENCHMARK_RUN_KEYS)
    if extra:
        raise _bad(extra[0], f"unknown field(s) {extra}")
    _check_model(req, known_models, allow_local_models)
    _check_name(req)
    _check_benchmarks(req, allow_jsonl)
    _check_limit(req)
    inc = req.get("include_diagnosis", True)
    if not isinstance(inc, bool):
        raise _bad("include_diagnosis", "include_diagnosis must be true or false")
    return {"mode": "explore", "model": req["model"], "name": req.get("name"), "benchmarks": req.get("benchmarks"),
            "benchmark_limit": req.get("benchmark_limit", "default"), "include_diagnosis": inc}


# ------------------------------------------------------------------ resolution
def _default_name(model: str) -> str:
    return f"{Path(model).name or model}-{time.strftime('%Y%m%d-%H%M')}"


def resolve(kind: str, norm: dict, preset: Config, job_id: str, runs_root: Path, db: str) -> tuple[Config, dict]:
    """The job's full experiment Config and the `resolved` summary (contract ResolvedConfig + internals)."""
    cfg = copy.deepcopy(preset)
    mode = norm["mode"]
    cfg.name, cfg.output_dir, cfg.db = f"pl_{job_id}", str((runs_root / job_id).resolve()), db
    cfg.model, cfg.job_id, cfg.mode = norm["model"], job_id, mode
    if mode == "paper":
        arms = list(ARMS)
        ratios = [float(r) for r in cfg.data.get("ratios", [3])]
        main_ratio = float(cfg.data.get("main_ratio", ratios[0]))
        benchmarks = cfg.eval.get("benchmarks") or default_benchmarks(cfg)
        limit = None
        target = {"strategy": "auto"}
    else:
        benchmarks = norm.get("benchmarks") or list(DEFAULT_EXPLORE_BENCHMARKS)
        limit = preset.eval.get("benchmark_limit") if norm.get("benchmark_limit") == "default" else norm.get("benchmark_limit")
        cfg.eval["benchmarks"] = list(benchmarks)
        cfg.eval["benchmark_limit"] = limit
        if kind == "pipeline":
            arms, ratios, main_ratio = norm["arms"], [float(norm["ratio"])], float(norm["ratio"])
            cfg.data["ratios"], cfg.data["main_ratio"] = [norm["ratio"]], norm["ratio"]
            cfg.seeds = list(norm["seeds"])
            target = norm["target"]
        else:
            arms, ratios, main_ratio, target = [], [], None, None
            cfg.seeds = [0]
    revision = hf_revision(cfg.model)
    resolved = {
        "kind": kind, "mode": mode, "model": cfg.model, "model_revision": revision,
        "benchmarks": list(benchmarks), "benchmark_limit": limit, "arms": list(arms), "ratios": ratios,
        "main_ratio": main_ratio, "seeds": list(cfg.seeds), "prompt_version": cfg.prompt_version,
        "target_request": target, "include_diagnosis": norm.get("include_diagnosis", True),
        "config_hash": config_hash(cfg), "baseline_hash": baseline_hash(cfg, list(benchmarks), limit, revision),
        "diagnosis_hash": (diagnosis_hash(cfg, revision)
                           if kind == "pipeline" or norm.get("include_diagnosis", True) else None),
        "reuse": None, "config": asdict(cfg),
    }
    return cfg, resolved


# ---------------------------------------------------------------------- hashes
def _h(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _gen_settings(cfg: Config) -> dict:
    return {k: v for k, v in asdict(cfg.gen()).items() if k != "prompt_version"}


def baseline_hash(cfg: Config, benchmarks: list[str], limit: int | None, revision: str | None) -> str:
    """Same hash = the base model's benchmark results can be reused (PLAN.md §6.3)."""
    opts = {b: {k: (asdict(v) if hasattr(v, "__dataclass_fields__") else v)
                for k, v in benchmark_options(cfg, b).items()} for b in benchmarks}
    return _h({"model": cfg.model, "revision": revision, "benchmarks": benchmarks, "benchmark_limit": limit,
               "benchmark_options": opts, "prompt_version": cfg.prompt_version, "generation": _gen_settings(cfg),
               "eval_load_in_4bit": bool(cfg.eval.get("load_in_4bit", False)),
               "eval_data": cfg.eval_data})


def diagnosis_hash(cfg: Config, revision: str | None) -> str:
    """Same hash = the diagnosis (probe grid, samples, LLTM, bootstrap) can be reused (PLAN.md §6.3)."""
    return _h({"model": cfg.model, "revision": revision, "prompt_version": cfg.prompt_version,
               "diagnose": cfg.diagnose, "generation": _gen_settings(cfg),
               "eval_load_in_4bit": bool(cfg.eval.get("load_in_4bit", False))})


def config_hash(cfg: Config) -> str:
    d = asdict(cfg)
    for k in ("name", "output_dir", "db", "job_id"):
        d.pop(k, None)
    return _h(d)


# ------------------------------------------------------------------- reuse
def _finished_step(store: Store, job_id: str, stage: str) -> dict | None:
    for s in store.get_steps(job_id):
        if s["stage"] == stage and s["status"] in ("done", "skipped"):
            return s
    return None


def plan_reuse(store: Store, source_id: str, resolved: dict) -> dict:
    """Which steps a new explore pipeline can skip by reusing `source_id` (PLAN.md §6.3)."""
    src = store.get_job(source_id)
    if src is None:
        raise JobError("reuse_mismatch", f"reuse_from: no job {source_id!r}")
    if not (src["kind"] == "benchmark_run" or (src["kind"] == "pipeline" and src["mode"] == "explore")):
        raise JobError("reuse_mismatch", "reuse_from must be an explore pipeline or a benchmark run")
    base = _finished_step(store, source_id, "baseline")
    if base is None:
        raise JobError("reuse_mismatch", "the source job has no finished baseline to reuse")
    if src["baseline_hash"] != resolved["baseline_hash"]:
        raise JobError("reuse_mismatch", "the source job used a different model, benchmarks, benchmark_limit, "
                       "prompt version, generation settings or precision; nothing can be reused",
                       {"source_baseline_hash": src["baseline_hash"], "baseline_hash": resolved["baseline_hash"]})
    diag = _finished_step(store, source_id, "diagnose")
    diag_ok = diag is not None and src["diagnosis_hash"] == resolved["diagnosis_hash"]
    return {"source": source_id, "source_output_dir": src["output_dir"],
            "baseline_run_ids": base["run_ids"], "diagnosis_reused": diag_ok,
            "diagnosis_run_ids": diag["run_ids"] if diag_ok else []}


# ------------------------------------------------------------------ create
def create_job_from_request(store: Store, kind: str, request: dict, presets: dict[str, Config] | None = None,
                            known_models: list[str] | None = None, runs_root: str | Path = "runs/pipelines",
                            allow_local_models: bool = False, allow_jsonl: bool = False) -> str:
    """Validate, resolve and store a job with its steps (status queued). Raises JobError on a bad request."""
    presets = presets or load_presets()
    known = known_models if known_models is not None else known_model_ids()
    if kind == "pipeline":
        norm = validate_pipeline_request(request, known, allow_local_models, allow_jsonl)
    elif kind == "benchmark_run":
        norm = validate_benchmark_run_request(request, known, allow_local_models, allow_jsonl)
    else:
        raise JobError("validation_error", f"unknown job kind {kind!r}")
    job_id = uuid.uuid4().hex[:12]
    db = str(getattr(store, "path", "")) or presets[norm["mode"]].db
    cfg, resolved = resolve(kind, norm, presets[norm["mode"]], job_id, Path(runs_root), db)
    steps = build_steps(kind, resolved)
    if kind == "pipeline" and norm.get("reuse_from"):
        reuse = plan_reuse(store, norm["reuse_from"], resolved)
        resolved["reuse"] = reuse
        for s in steps:
            if s["stage"] == "baseline":
                s.update(status="skipped", run_ids=reuse["baseline_run_ids"])
            elif s["stage"] == "diagnose" and reuse["diagnosis_reused"]:
                s.update(status="skipped", run_ids=reuse["diagnosis_run_ids"])
    store.create_job(kind, norm.get("name") or _default_name(cfg.model), norm["mode"], request, job_id=job_id,
                     resolved=resolved, baseline_hash=resolved["baseline_hash"],
                     diagnosis_hash=resolved["diagnosis_hash"], config_hash=resolved["config_hash"],
                     output_dir=cfg.output_dir, reuse_from=norm.get("reuse_from"))
    store.set_steps(job_id, steps)
    for i, s in enumerate(steps):
        if s.get("status") == "skipped":
            store.update_step(job_id, i, run_ids=s["run_ids"], finished_at=time.time())
    return job_id
