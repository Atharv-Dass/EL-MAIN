"""One function per pipeline stage (PLAN.md §5, S0-S7). Each reuses the existing stage function in
experiments/pipeline.py; this module only adds job plumbing: progress, cancel checks, run ids, warnings, timing.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from ..errors import Cancelled
from ..experiments import pipeline as P
from ..provenance import hf_revision, provenance
from ..store.db import Store
from .errors import JobError


@dataclass
class StepContext:
    store: Store
    job: dict
    step: dict
    cfg: P.Config
    runner_factory: P.RunnerFactory
    log: Callable[[str], None] = print
    metrics: dict = field(default_factory=lambda: {"load_s": 0.0, "run_s": None, "gpu": None})
    run_ids: list = field(default_factory=list)
    _last_cancel_check: float = 0.0

    @property
    def job_id(self) -> str:
        return self.job["id"]

    def cancel_requested(self) -> bool:
        job = self.store.get_job(self.job_id)
        return bool(job and job["cancel_requested"])

    def progress(self, done: int, total: int) -> None:
        """Progress callback for evaluation/diagnosis chunks: also the cancel safe point (PLAN.md §7.3)."""
        self.store.update_step(self.job_id, self.step["idx"], progress_done=int(done), progress_total=int(total))
        if self.cancel_requested():
            raise Cancelled(f"cancelled during {self.step['key']} after {done}/{total}")

    def warn(self, message: str) -> None:
        self.store.add_job_warning(self.job_id, message)
        self.log(f"WARNING: {message}")

    def timed_factory(self) -> P.RunnerFactory:
        """Wrap the runner factory so model (+ adapter) loading time is recorded as load_s."""
        def factory(model, adapter, gen):
            t = time.time()
            runner = self.runner_factory(model, adapter, gen)
            self.metrics["load_s"] = round(self.metrics["load_s"] + time.time() - t, 3)
            return runner
        return factory


# ------------------------------------------------------------------ S0
def _free_gb(path: Path) -> float:
    p = path
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def _check_model_access(ctx: StepContext) -> None:
    model = ctx.cfg.model
    if Path(model).is_dir() or os.environ.get("HF_HUB_OFFLINE"):
        return
    try:
        from huggingface_hub import model_info

        model_info(model)
    except Exception as e:  # noqa: BLE001 - classified below
        name = type(e).__name__
        status = getattr(getattr(e, "response", None), "status_code", None)
        if name in ("GatedRepoError", "RepositoryNotFoundError") or status in (401, 403):
            raise JobError("model_access_denied",
                           f"No access to {model}. Accept its licence on Hugging Face and run "
                           "`huggingface-cli login` on this laptop.") from e
        if hf_revision(model) is None:
            ctx.warn(f"could not check access to {model} ({name}); it is not in the local cache either")


def _check_datasets(ctx: StepContext, benchmarks: list[str], needs_real: bool) -> None:
    from ..benchmarks import registry

    for name in benchmarks:
        b = registry.get(name)
        if b.kind != "external":
            continue
        try:
            b.load(**{**P.benchmark_options(ctx.cfg, name), "limit": 1})
        except Exception as e:  # noqa: BLE001
            raise JobError("dataset_unavailable", f"benchmark {name} could not be loaded ({type(e).__name__}). "
                           "Check the network, or set local JSONL files under eval_data.") from e
    if needs_real:
        try:
            P._real_pool(ctx.cfg)[:1]
        except Exception as e:  # noqa: BLE001
            raise JobError("dataset_unavailable", f"real training data ({ctx.cfg.real_data}) could not be loaded "
                           f"({type(e).__name__}).") from e


def _bnb_cuda() -> bool | None:
    try:
        from ..models.runner import check_4bit_support

        check_4bit_support()
        return True
    except Exception:  # noqa: BLE001
        return False


def _copy_reused(ctx: StepContext, reuse: dict) -> None:
    src, dst = Path(reuse["source_output_dir"]), ctx.cfg.out
    files = ["evals/base.json"] + (["diagnosis.json", "annotate_errors.csv"] if reuse["diagnosis_reused"] else [])
    for rel in files:
        if (src / rel).exists():
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / rel, dst / rel)


def preflight(ctx: StepContext) -> None:
    cfg, resolved = ctx.cfg, ctx.job["resolved"]
    from ..models.catalog import model_ids

    if not cfg.model:
        raise JobError("validation_error", "the config has no model")
    cfg.gen()   # validates generation settings + prompt version
    if cfg.model not in model_ids() and not Path(cfg.model).is_dir():
        raise JobError("validation_error", f"model {cfg.model!r} is not in configs/models.yaml")
    train_4bit = ctx.job["kind"] == "pipeline" and bool(cfg.train.get("load_in_4bit", True))
    need_gpu = cfg.gpu.get("required", True) or train_4bit or bool(cfg.eval.get("load_in_4bit"))
    cuda = None
    if need_gpu:
        import torch

        cuda = torch.cuda.is_available()
        if not cuda:
            raise JobError("gpu_unavailable", "This job needs a CUDA GPU, but torch.cuda.is_available() is False.")
    min_gb = float(cfg.preflight.get("min_free_gb", 10))
    free = _free_gb(cfg.out)
    if free < min_gb:
        raise JobError("disk_low", f"only {free:.1f} GB free on the drive holding {cfg.out}; "
                       f"preflight.min_free_gb is {min_gb:g}", {"free_gb": round(free, 1), "min_free_gb": min_gb})
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    if _free_gb(hf_home) < min_gb:
        ctx.warn(f"less than {min_gb:g} GB free on the drive holding the Hugging Face cache ({hf_home})")
    _check_model_access(ctx)
    _check_datasets(ctx, resolved["benchmarks"], needs_real=ctx.job["kind"] == "pipeline")
    if resolved.get("reuse"):
        _copy_reused(ctx, resolved["reuse"])
    import yaml

    (cfg.out / "config.resolved.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8")
    prov = {**provenance(), "mode": ctx.job["mode"], "model": cfg.model, "model_revision": hf_revision(cfg.model),
            "prompt_version": cfg.prompt_version, "seeds": cfg.seeds,
            "precision": {"train_load_in_4bit": train_4bit, "eval_load_in_4bit": bool(cfg.eval.get("load_in_4bit")),
                          "bitsandbytes_cuda": _bnb_cuda() if train_4bit and cuda else None}}
    (cfg.out / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    ctx.store.update_job(ctx.job_id, provenance=prov)
    ctx.progress(1, 1)


# --------------------------------------------------------------- S1, S2
def baseline(ctx: StepContext) -> None:
    out = P.evaluate_model(ctx.cfg, ctx.store, ctx.timed_factory(), "base", progress=ctx.progress)
    ctx.run_ids.append(out["run_id"])


def diagnose(ctx: StepContext) -> None:
    out = P.diagnose(ctx.cfg, ctx.store, ctx.timed_factory(), progress=ctx.progress)
    ctx.run_ids.append(out["run_id"])


# --------------------------------------------------------------- S3, S4
def choose_target(ctx: StepContext) -> None:
    req = ctx.job["resolved"].get("target_request") or {"strategy": "auto"}
    result = P.resolve_target(ctx.cfg, req["strategy"], req.get("feature"))
    for w in result["warnings"]:
        ctx.warn(w)
    ctx.progress(1, 1)


def build_data(ctx: StepContext) -> None:
    r = ctx.job["resolved"]
    if ctx.job["mode"] == "paper":
        manifest = P.build_data(ctx.cfg, ctx.store)
    else:
        keys = sorted({(arm, ratio) for arm, ratio in
                       ((s["params"]["arm"], s["params"]["ratio"]) for s in ctx.store.get_steps(ctx.job_id)
                        if s["stage"] == "train")})
        manifest = P.build_data(ctx.cfg, ctx.store, arms=keys)
    for w in manifest.get("warnings", []):
        ctx.warn(w)
    for k, m in manifest["arms"].items():
        if k.startswith("match_") and m["n_matched"] < m["n_requested"]:
            ctx.warn(f"{k}: matched only {m['n_matched']}/{m['n_requested']} targeted items")
    runs = ctx.store.list_runs("build-data", job_id=ctx.job_id, limit=1)
    ctx.run_ids += [x["id"] for x in runs]
    ctx.progress(1, 1)


# --------------------------------------------------------------- S5, S6
def train(ctx: StepContext) -> None:
    p = ctx.step["params"]
    curve: list[dict] = []

    def on_progress(step: int, total: int, loss: float | None) -> None:
        if loss is not None:
            curve.append({"step": step, "loss": round(loss, 5)})
        ctx.store.update_step(ctx.job_id, ctx.step["idx"], progress_done=step, progress_total=total,
                              metrics={**ctx.metrics, "loss_curve": curve})

    P.train_arm(ctx.cfg, ctx.store, p["arm"], p["ratio"], p["seed"], on_progress=on_progress,
                should_cancel=ctx.cancel_requested)
    key = P.adapter_dir(ctx.cfg, p["arm"], p["ratio"], p["seed"]).name
    manifest = json.loads((P.adapter_dir(ctx.cfg, p["arm"], p["ratio"], p["seed"]) / "train_manifest.json")
                          .read_text(encoding="utf-8"))
    ctx.metrics["load_s"] = round(manifest.get("load_seconds", 0.0), 3)
    ctx.metrics["run_s"] = round(manifest.get("wall_seconds", 0.0), 3)
    ctx.metrics["loss_curve"] = manifest.get("loss_curve", curve)
    ctx.run_ids += [r["id"] for r in ctx.store.list_runs("train", job_id=ctx.job_id)
                    if r["name"] == f"{ctx.cfg.name}:train:{key}"][:1]


def evaluate(ctx: StepContext) -> None:
    p = ctx.step["params"]
    out = P.evaluate_model(ctx.cfg, ctx.store, ctx.timed_factory(), p["arm"], p["ratio"], p["seed"],
                           progress=ctx.progress)
    ctx.run_ids.append(out["run_id"])


# ------------------------------------------------------------------- S7
def compare(ctx: StepContext) -> None:
    """Stage compare (PLAN.md S7, §9): results.json (contract PipelineResults) + report.md;
    paper mode also keeps the original report.json."""
    from .results import write_results

    if ctx.job["mode"] == "paper":
        reuse = ctx.job["resolved"].get("reuse") or {}
        P.report(ctx.cfg, ctx.store, include_run_ids=reuse.get("baseline_run_ids", []), write_md=False)
    res = write_results(ctx.store, ctx.job_id)
    for r in res["regressions"]:
        ctx.warn(f"regression: {r['arm']} ratio {r['ratio']} on {r['benchmark']} is worse than the untrained model "
                 f"({r['mean_diff']:+.3f}, CI [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}])")
    ctx.progress(1, 1)


STAGE_FUNCS: dict[str, Callable[[StepContext], None]] = {
    "preflight": preflight, "baseline": baseline, "diagnose": diagnose, "choose_target": choose_target,
    "build_data": build_data, "train": train, "evaluate": evaluate, "compare": compare,
}
GPU_STAGES = ("baseline", "diagnose", "train", "evaluate")
