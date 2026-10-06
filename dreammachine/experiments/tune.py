"""Tune training settings on the dev set (PLAN.md §14 question 7). EXPLORE ONLY: never paper evidence.

    python -m dreammachine.experiments.tune --config configs/tune.yaml

The dev set is the last `dev_holdout` GSM8K *train* problems; they are removed from the real training data
(`data.dev_holdout`), so the evaluation never sees a trained-on problem, and GSM8K test is never touched.
Each variant = one arm trained with its own settings, then evaluated on the dev set and a held-out probe set,
paired against the untrained model on the same problems. An existing diagnosis (target + LLTM) is reused.
Resumable: finished variants (results/<name>.json) are skipped; training resumes from its last checkpoint.

Variant keys: name, arm, ratio, train: {TrainConfig overrides}, data: {data-section overrides, e.g. token_budget},
distill: {DistillConfig fields} (PLAN.md D14: the base model rewrites the solution texts before training).
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..benchmarks import registry
from ..store.db import Store
from . import pipeline as P
from .evaluate import evaluate, slice_accuracy, summarize


def free_gpu() -> None:
    """Release GPU memory between variants. All variants run in one process (unlike pipeline steps, which each get
    their own subprocess), so without this the cached memory of earlier variants piles up (an OOM on the 5th
    variant overnight 2026-10-06)."""
    import gc
    import sys

    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _item_correct(recs) -> dict[str, float]:
    per: dict[str, list[float]] = {}
    for r in recs:
        per.setdefault(r.example_id, []).append(float(r.correct))
    return {k: float(np.mean(v)) for k, v in per.items()}


def _vs(a: dict[str, float], b: dict[str, float]) -> dict:
    ids = sorted(set(a) & set(b))
    return P.paired_bootstrap(np.array([a[i] for i in ids]), np.array([b[i] for i in ids]))


def eval_sets(cfg: P.Config, spec: dict) -> dict[str, list]:
    sets = {"dev": P.dev_set(cfg)}
    n_probe = spec.get("probes_heldout_families")
    if n_probe:
        sets["probes_heldout_families"] = registry.get("probes_heldout_families").load(
            limit=int(n_probe), **{k: v for k, v in P.benchmark_options(cfg, "probes_heldout_families").items()
                                   if k != "limit"})
    return sets


def run_eval(cfg: P.Config, factory: P.RunnerFactory, adapter: str | None, sets: dict, target: dict) -> dict:
    runner = factory(cfg.model, adapter, cfg.gen())
    out: dict[str, Any] = {}
    for name, exs in sets.items():
        recs = evaluate(runner, exs)
        s = summarize(recs)
        out[name] = {"accuracy": s["accuracy"], "items": _item_correct(recs),
                     "error_distribution": s["error_distribution"], "truncated_share": s.get("truncated_share"),
                     "mean_chars": float(np.mean([len(r.text) for r in recs]))}
        if name.startswith("probes") and target.get("feature"):
            out[name]["target_slice"] = slice_accuracy(recs, target["feature"], target["threshold"])
    del runner
    free_gpu()
    return out


def tune(config_path: str | Path, factory: P.RunnerFactory | None = None, log: Callable[[str], None] = print) -> dict:
    import yaml

    spec = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    base = P.Config.load(spec["base"])
    base.model, base.mode = spec["model"], "explore"
    base.seeds = [int(spec.get("seed", 0))]
    base.data = {**base.data, "dev_holdout": int(spec.get("dev_holdout", 200))}
    root = Path(spec["output_dir"])
    (root / "results").mkdir(parents=True, exist_ok=True)
    src = Path(spec["diagnosis_from"])
    store = Store(spec.get("db", str(root / "tune.db")))
    if factory is None:
        from functools import partial

        factory = partial(P.hf_runner_factory, load_in_4bit=bool(base.eval.get("load_in_4bit", False)))
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(float(base.gpu.get("memory_fraction", 0.92)))
        except ImportError:
            pass

    def variant_cfg(name: str, v: dict) -> P.Config:
        cfg = copy.deepcopy(base)
        cfg.name, cfg.output_dir = f"tune:{name}", str(root / name)
        cfg.train = {**cfg.train, **v.get("train", {})}
        cfg.data = {**cfg.data, **v.get("data", {})}
        ratio = float(v.get("ratio", 3))
        if v["arm"] != "real_only":
            cfg.data = {**cfg.data, "ratios": [ratio], "main_ratio": ratio}
        for f in ("diagnosis.json", "target.json"):
            if (src / f).exists() and not (cfg.out / f).exists():
                shutil.copy2(src / f, cfg.out / f)
        return cfg

    target = json.loads((src / "target.json").read_text(encoding="utf-8")) if (src / "target.json").exists() else {}
    base_cfg = variant_cfg("base", {"arm": "base"})
    sets = eval_sets(base_cfg, spec.get("eval", {}))
    log(f"dev set: {len(sets['dev'])} GSM8K-train problems (excluded from training); "
        f"probes: {len(sets.get('probes_heldout_families', []))}")
    base_file = root / "results" / "base.json"
    if not base_file.exists() and spec.get("base_results"):   # reuse an earlier untrained-model eval (same sets)
        shutil.copy2(spec["base_results"], base_file)
    if base_file.exists():
        base_res = json.loads(base_file.read_text(encoding="utf-8"))
    else:
        t = time.time()
        base_res = {"name": "base", "eval": run_eval(base_cfg, factory, None, sets, target), "eval_s": time.time() - t}
        base_file.write_text(json.dumps(base_res, indent=2), encoding="utf-8")
    log(f"base: dev {base_res['eval']['dev']['accuracy']:.3f}")

    results = {"base": base_res}
    for v in spec["variants"]:
        name = v["name"]
        out_file = root / "results" / f"{name}.json"
        if out_file.exists():
            results[name] = json.loads(out_file.read_text(encoding="utf-8"))
            log(f"{name}: already done")
            continue
        cfg = variant_cfg(name, v)
        arm, ratio = v["arm"], 0.0 if v["arm"] == "real_only" else float(v.get("ratio", 3))
        t0 = time.time()
        if "distill" in v:
            stats = distill_data(cfg, store, factory, arm, ratio, v["distill"], log)
        else:
            stats = P.build_data(cfg, store, arms=[(arm, ratio)])
        t1 = time.time()
        free_gpu()
        adapter = P.train_arm(cfg, store, arm, ratio, cfg.seeds[0])
        free_gpu()
        t2 = time.time()
        ev = run_eval(cfg, factory, str(adapter), sets, target)
        t3 = time.time()
        res = {"name": name, "arm": arm, "ratio": ratio, "train": cfg.train, "data": stats["arms"],
               "eval": ev, "build_s": t1 - t0, "train_s": t2 - t1, "eval_s": t3 - t2,
               "vs_base": {k: _vs(ev[k]["items"], base_res["eval"][k]["items"]) for k in ev}}
        dfile = P._arm_path(cfg, arm, ratio, cfg.seeds[0]).with_suffix(".distill.json")
        if dfile.exists():
            res["distill"] = json.loads(dfile.read_text(encoding="utf-8"))["distill"]
        out_file.write_text(json.dumps(res, indent=2), encoding="utf-8")
        results[name] = res
        d = res["vs_base"]["dev"]
        log(f"{name}: dev {ev['dev']['accuracy']:.3f} ({d['mean_diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]) "
            f"train {res['train_s'] / 60:.0f} min")
    summary = summarise(results)
    (root / "summary.md").write_text(summary, encoding="utf-8")
    log(summary)
    return results


def distill_data(cfg: P.Config, store: Store, factory: P.RunnerFactory, arm: str, ratio: float, spec: dict,
                 log: Callable[[str], None] = print) -> dict:
    """build_data for one arm, then self-distil its solution texts (PLAN.md D14) and trim it back to the budget.
    The stats file marks the step done, so a resumed run does not rebuild (and so undo) the distilled arm."""
    from ..data.distill import DistillConfig, distill_arm

    seed = cfg.seeds[0]
    path = P._arm_path(cfg, arm, ratio, seed)
    stats_file = path.with_name(path.stem + ".distill.json")
    if stats_file.exists() and path.exists():
        return json.loads(stats_file.read_text(encoding="utf-8"))["build"]
    built = P.build_data(cfg, store, arms=[(arm, ratio)])
    rows = P._read_train(path)
    dc = DistillConfig(**spec)
    runner = factory(cfg.model, None, cfg.gen(n_samples=dc.samples, temperature=dc.temperature, top_p=dc.top_p,
                                              max_new_tokens=dc.max_new_tokens, batch_size=dc.batch_size))
    retry = None
    if dc.retry_samples:   # second try for items without an accepted answer: other seed, more samples, same model
        g2 = cfg.gen(n_samples=dc.retry_samples, temperature=dc.temperature, top_p=dc.top_p,
                     max_new_tokens=dc.max_new_tokens, batch_size=dc.batch_size)
        g2.seed += 1
        if hasattr(runner, "model") and hasattr(runner, "tokenizer"):
            from ..models.runner import HFRunner

            retry = HFRunner.from_objects(runner.model, runner.tokenizer, g2)
        else:
            retry = factory(cfg.model, None, g2)
    t = time.time()
    out, st = distill_arm(rows, runner, dc, int(cfg.data.get("token_budget", 600_000)), ratio, seed,
                          progress=lambda i, n: log(f"  distill {i}/{n} rows, {time.time() - t:.0f} s"),
                          retry_runner=retry)
    del runner, retry
    free_gpu()
    P._write_train(rows, path.with_name(path.stem + ".original.jsonl"))
    P._write_train(out, path)
    built = {**built, "arms": {**built["arms"], path.stem: P._arm_stats(out)}}
    stats_file.write_text(json.dumps({"build": built, "distill": {**st.to_dict(), "config": spec,
                                                                    "seconds": time.time() - t}}, indent=2),
                          encoding="utf-8")
    log(f"  distilled {st.n_distilled}/{st.n_items} rows ({st.by_source}); arm now {P._arm_stats(out)}")
    return built


def summarise(results: dict) -> str:
    base = results["base"]["eval"]
    lines = ["# Dev-set tuning (EXPLORE — not paper evidence)", "",
             "| variant | arm | ratio | lr | epochs | dev acc | Δ dev vs base [95% CI] | probes acc | Δ probes | "
             "target slice above (before → after) | mean answer chars | train min |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    b = base["dev"]
    pb = base.get("probes_heldout_families", {})
    lines.append(f"| base | — | — | — | — | {b['accuracy']:.3f} | — | {pb.get('accuracy', float('nan')):.3f} | — | "
                 f"{(pb.get('target_slice') or {}).get('above', {}).get('accuracy')} | {b['mean_chars']:.0f} | — |")
    for name, r in results.items():
        if name == "base":
            continue
        e, vs = r["eval"], r["vs_base"]
        d, p = vs["dev"], vs.get("probes_heldout_families")
        ts = (e.get("probes_heldout_families", {}).get("target_slice") or {}).get("above", {}).get("accuracy")
        ts0 = (pb.get("target_slice") or {}).get("above", {}).get("accuracy")
        lines.append(
            f"| {name} | {r['arm']} | {r['ratio']:g} | {r['train'].get('learning_rate')} | {r['train'].get('num_epochs')} "
            f"| {e['dev']['accuracy']:.3f} | {d['mean_diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] "
            f"| {e.get('probes_heldout_families', {}).get('accuracy', float('nan')):.3f} "
            f"| {p['mean_diff']:+.3f} | {ts0} → {ts} | {e['dev']['mean_chars']:.0f} | {r['train_s'] / 60:.0f} |"
            if p else f"| {name} | {r['arm']} | {r['ratio']:g} | | | {e['dev']['accuracy']:.3f} | {d['mean_diff']:+.3f} | | | | | |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m dreammachine.experiments.tune", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    args = p.parse_args(argv)
    # Windows: a redirected stdout uses the ANSI code page (cp1252), which cannot print "Δ" (seen 2026-10-06).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    tune(args.config, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
