"""End-to-end experiment stages (docs/RESEARCH.md §4).

    screen      candidate models on GSM8K + a small probe set -> pick the base model
    diagnose    factorial probes -> LLTM -> ranked weaknesses -> target factor (+ 150 errors to annotate)
    build-data  pools -> targeted / matched-control / untargeted arms -> token-budget mixes (JSONL)
    train       QLoRA per (arm, ratio, seed)
    evaluate    GSM8K test, GSM-Symbolic, held-out-family probes, fresh probes (-> refitted LLTM)
    report      arm x metric table over seeds + paired bootstrap differences

Every stage records a run in the SQLite store. File outputs go under cfg.output_dir.
Model/runner construction is injected (`runner_factory`), so the whole pipeline
is testable on CPU with EchoRunner.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..data.loaders import Example, from_items, load_gsm8k, load_gsm_symbolic, load_jsonl, save_jsonl
from ..data.mixer import TrainExample, mix, to_train, whitespace_tokens
from ..diagnosis.agreement import stratified_sample
from ..diagnosis.lltm import LLTMResult, bootstrap_lltm, fit_lltm
from ..diagnosis.targeting import (
    baseline_threshold, choose_target, design_matrix, rank_weaknesses, select_matched_control,
    select_targeted, select_untargeted,
)
from ..generator.features import FEATURES
from ..generator.probes import ProbeGrid, build_pool, factorial_probe_set
from ..models.runner import GenConfig, Runner
from ..store.db import Store
from .evaluate import ResponseRecord, evaluate, item_counts, slice_accuracy, summarize

ARMS = ("real_only", "untargeted", "matched_control", "targeted")

RunnerFactory = Callable[[str, "str | None", GenConfig], Runner]


def hf_runner_factory(model: str, adapter: str | None, gen: GenConfig, load_in_4bit: bool = False) -> Runner:
    from ..models.runner import HFRunner

    return HFRunner(model, adapter=adapter, gen=gen, load_in_4bit=load_in_4bit)


# ------------------------------------------------------------------- config
@dataclass
class Config:
    name: str
    output_dir: str
    model: str = ""
    candidates: list[str] = field(default_factory=list)
    db: str = "runs/dreammachine.db"
    generation: dict = field(default_factory=dict)
    screen: dict = field(default_factory=dict)
    diagnose: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    train: dict = field(default_factory=dict)
    eval: dict = field(default_factory=dict)
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    real_data: str = "gsm8k"          # "gsm8k" or a path to a JSONL of Examples
    eval_data: dict = field(default_factory=dict)  # optional local JSONL overrides

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**raw)

    @property
    def out(self) -> Path:
        p = Path(self.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def gen(self, **overrides: Any) -> GenConfig:
        return GenConfig(**{**self.generation, **overrides})

    def grid(self, section: dict) -> ProbeGrid:
        g = section.get("grid", {})
        return ProbeGrid(**{k: tuple(v) for k, v in g.items()}) if g else ProbeGrid()


def _examples(cfg: Config, key: str, loader: Callable[[], list[Example]]) -> list[Example]:
    """Local JSONL override (cfg.eval_data[key]) or the Hugging Face loader."""
    path = cfg.eval_data.get(key)
    return load_jsonl(path) if path else loader()


def _real_pool(cfg: Config) -> list[Example]:
    if cfg.real_data == "gsm8k":
        return _examples(cfg, "gsm8k_train", lambda: load_gsm8k("train"))
    return load_jsonl(cfg.real_data)


def dev_set(cfg: Config) -> list[Example]:
    """The last `data.dev_holdout` real training problems: a dev set for tuning settings (never GSM8K test)."""
    n = int(cfg.data.get("dev_holdout", 0))
    return _real_pool(cfg)[-n:] if n > 0 else []


def _real_train(cfg: Config) -> list[Example]:
    """Real training problems, minus the dev hold-out (if set) so tuning never evaluates on trained items."""
    pool = _real_pool(cfg)
    n = int(cfg.data.get("dev_holdout", 0))
    return pool[:-n] if n > 0 else pool


# -------------------------------------------------------------------- screen
def screen(cfg: Config, store: Store, runner_factory: RunnerFactory) -> dict:
    s = cfg.screen
    gsm = _examples(cfg, "gsm8k_test", lambda: load_gsm8k("test", limit=s.get("gsm8k_limit", 500)))
    gsm = gsm[: s.get("gsm8k_limit", len(gsm))]
    probes = from_items(factorial_probe_set(cfg.grid(s), per_cell=s.get("probe_per_cell", 1), seed=11))
    rows = []
    for model in cfg.candidates:
        run = store.create_run(f"{cfg.name}:screen:{model}", "screen", {"model": model, **s})
        runner = runner_factory(model, None, cfg.gen())
        recs = evaluate(runner, gsm + probes)
        store.add_responses(run, recs)
        summ = summarize(recs)
        fmt = summ["error_distribution"].get("FORMAT_ERROR", 0.0)
        acc_gsm = summ["accuracy_by_source"].get("gsm8k", float("nan"))
        row = {"model": model, "run_id": run, "gsm8k_accuracy": acc_gsm,
               "probe_accuracy": summ["accuracy_by_source"].get("synthetic:train", float("nan")),
               "format_error_share": fmt}
        store.finish_run(run, metrics=row)
        rows.append(row)
    # "Usable headroom": enough competence to learn from (>= 25%), room to improve,
    # and output format that follows instructions (format errors < 20% of errors).
    usable = [r for r in rows if r["gsm8k_accuracy"] >= s.get("min_accuracy", 0.25)
              and r["format_error_share"] < s.get("max_format_share", 0.2)]
    pick = max(usable, key=lambda r: 1 - r["gsm8k_accuracy"]) if usable else None
    result = {"candidates": rows, "recommended": pick["model"] if pick else None}
    (cfg.out / "screen.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# ------------------------------------------------------------------ diagnose
def diagnose(cfg: Config, store: Store, runner_factory: RunnerFactory) -> dict:
    d = cfg.diagnose
    run = store.create_run(f"{cfg.name}:diagnose", "diagnose", {"model": cfg.model, **d})
    probe_items = factorial_probe_set(cfg.grid(d), per_cell=d.get("probe_per_cell", 4), seed=d.get("seed", 0))
    gen = cfg.gen(n_samples=d.get("n_samples", 3), temperature=d.get("temperature", 0.7))
    recs = evaluate(runner_factory(cfg.model, None, gen), from_items(probe_items))
    store.add_responses(run, recs)

    Q, s, n, _ = item_counts(recs)
    lltm = bootstrap_lltm(Q, s, n, list(FEATURES), n_boot=d.get("bootstrap", 500), seed=d.get("seed", 0))
    natural = build_pool(d.get("natural_pool", 5000), seed=d.get("seed", 0) + 1)
    nat_Q = design_matrix(natural, FEATURES)
    ranked = rank_weaknesses(lltm, nat_Q)
    target = choose_target(ranked)
    threshold = baseline_threshold(nat_Q, list(FEATURES), target.feature) if target else None

    result = {
        "run_id": run, "model": cfg.model, "summary": summarize(recs), "lltm": lltm.to_dict(),
        "weaknesses": [w.to_dict() for w in ranked],
        "target": target.to_dict() if target else None, "threshold": threshold,
        "probe_ids": sorted({it.id for it in probe_items}),
    }
    for k, v in result.items():
        if k != "run_id":
            store.put_artifact(run, k, v)
    (cfg.out / "diagnosis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _export_annotation_sheet(recs, from_items(probe_items), cfg.out / "annotate_errors.csv",
                             d.get("annotate", 150), d.get("seed", 0))
    store.finish_run(run, metrics={"accuracy": result["summary"]["accuracy"],
                                   "target": target.feature if target else None})
    return result


def _export_annotation_sheet(recs: list[ResponseRecord], examples: list[Example], path: Path,
                             n: int, seed: int) -> None:
    """CSV for two human annotators; the classifier's label is in a separate column to hide
    during annotation (blind labelling), then compare with diagnosis.agreement.cohen_kappa."""
    by_id = {e.id: e for e in examples}
    errors = [r for r in recs if not r.correct]
    sample = stratified_sample(errors, key=lambda r: r.error_type, n=n, seed=seed)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["example_id", "sample", "question", "gold", "response", "classifier_label",
                    "annotator_1", "annotator_2"])
        for r in sample:
            ex = by_id[r.example_id]
            w.writerow([r.example_id, r.sample, ex.question, ex.answer, r.text, r.error_type, "", ""])


# ---------------------------------------------------------------- build data
def _arm_path(cfg: Config, arm: str, ratio: float, seed: int) -> Path:
    return cfg.out / "data" / f"{arm}_r{ratio:g}_s{seed}.jsonl"


def build_data(cfg: Config, store: Store) -> dict:
    dz = cfg.data
    diag = json.loads((cfg.out / "diagnosis.json").read_text(encoding="utf-8"))
    if not diag["target"]:
        raise RuntimeError("diagnosis found no significant weakness; nothing to target")
    lltm = LLTMResult.from_dict(diag["lltm"])
    target, threshold = diag["target"]["feature"], diag["threshold"]
    budget = int(dz.get("token_budget", 600_000))
    ratios = [float(r) for r in dz.get("ratios", [3])]
    p_band = tuple(dz.get("p_band", [0.3, 0.7]))
    exclude = set(diag["probe_ids"])
    real = [to_train(e) for e in _real_train(cfg)]
    manifest: dict[str, Any] = {"target": target, "threshold": threshold, "budget_tokens": budget, "arms": {}}

    for seed in cfg.seeds:
        pool = build_pool(dz.get("pool_size", 40_000), seed=100 + seed, grid=cfg.grid(dz), exclude_ids=exclude)
        natural = build_pool(dz.get("natural_size", 20_000), seed=200 + seed, exclude_ids=exclude)
        max_syn_tokens = budget * max(ratios) / (1 + max(ratios))
        mean_len = np.mean([whitespace_tokens(it.question) + whitespace_tokens(it.solution) for it in pool[:500]])
        need = int(math.ceil(1.3 * max_syn_tokens / mean_len))
        targeted = select_targeted(pool, lltm, target, threshold, n=need, p_band=p_band, seed=seed)
        control, match = select_matched_control(pool, lltm, target, threshold, targeted, seed=seed)
        untargeted = select_untargeted(natural, need, seed=seed)
        if len(targeted) < need:
            manifest.setdefault("warnings", []).append(
                f"seed {seed}: only {len(targeted)}/{need} targeted items; increase data.pool_size")
        syn = {
            "targeted": [to_train(e) for e in from_items(targeted)],
            "matched_control": [to_train(e) for e in from_items(control)],
            "untargeted": [to_train(e) for e in from_items(untargeted)],
        }
        manifest["arms"][f"match_s{seed}"] = match.to_dict()
        path = _arm_path(cfg, "real_only", 0, seed)
        rows = mix(real, [], 0, budget, seed=seed)
        _write_train(rows, path)
        manifest["arms"][path.stem] = _arm_stats(rows)
        for arm, data in syn.items():
            for ratio in ratios:
                rows = mix(real, data, ratio, budget, seed=seed)
                path = _arm_path(cfg, arm, ratio, seed)
                _write_train(rows, path)
                manifest["arms"][path.stem] = _arm_stats(rows)
    (cfg.out / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    run = store.create_run(f"{cfg.name}:build-data", "build-data", dz)
    store.put_artifact(run, "manifest", manifest)
    store.finish_run(run)
    return manifest


def _write_train(rows: list[TrainExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def _read_train(path: Path) -> list[TrainExample]:
    with path.open(encoding="utf-8") as f:
        return [TrainExample(**json.loads(line)) for line in f if line.strip()]


def _arm_stats(rows: list[TrainExample]) -> dict:
    tokens = sum(whitespace_tokens(r.prompt) + whitespace_tokens(r.completion) for r in rows)
    syn = sum(r.source.startswith("synthetic") for r in rows)
    return {"n": len(rows), "tokens": tokens, "n_synthetic": syn, "n_real": len(rows) - syn}


# --------------------------------------------------------------------- train
def adapter_dir(cfg: Config, arm: str, ratio: float, seed: int) -> Path:
    return cfg.out / "adapters" / f"{arm}_r{ratio:g}_s{seed}"


def train_arm(cfg: Config, store: Store, arm: str, ratio: float, seed: int) -> Path:
    from ..train.qlora import TrainConfig, train

    rows = _read_train(_arm_path(cfg, arm, ratio, seed))
    out = adapter_dir(cfg, arm, ratio, seed)
    tcfg = TrainConfig(base_model=cfg.model, output_dir=str(out), seed=seed, **cfg.train)
    run = store.create_run(f"{cfg.name}:train:{out.name}", "train",
                           {"arm": arm, "ratio": ratio, "seed": seed, "n": len(rows)})
    try:
        path = train(tcfg, rows)
    except Exception as e:
        store.finish_run(run, status="failed", metrics={"error": repr(e)})
        raise
    store.finish_run(run, metrics=json.loads((out / "train_manifest.json").read_text(encoding="utf-8"))["train_metrics"])
    return path


# ------------------------------------------------------------------ evaluate
def eval_sets(cfg: Config) -> dict[str, list[Example]]:
    e = cfg.eval
    sets: dict[str, list[Example]] = {}
    lim = e.get("gsm8k_limit")
    sets["gsm8k"] = _examples(cfg, "gsm8k_test", lambda: load_gsm8k("test", limit=lim))[:lim]
    for v in e.get("gsm_symbolic", []):
        sets[f"gsm_symbolic:{v}"] = _examples(cfg, f"gsm_symbolic_{v}",
                                              lambda v=v: load_gsm_symbolic(v, limit=e.get("gsm_symbolic_limit")))
    per = e.get("probe_per_cell", 2)
    sets["probes_train_families"] = from_items(factorial_probe_set(cfg.grid(e), per_cell=per, seed=777))
    sets["probes_heldout_families"] = from_items(
        factorial_probe_set(cfg.grid(e), per_cell=per, seed=778, split="heldout"))
    return sets


def evaluate_model(cfg: Config, store: Store, runner_factory: RunnerFactory, arm: str,
                   ratio: float | None = None, seed: int | None = None) -> dict:
    """arm='base' evaluates the untrained model; otherwise the adapter for (arm, ratio, seed)."""
    adapter = None if arm == "base" else str(adapter_dir(cfg, arm, ratio, seed) / "adapter")
    tag = "base" if arm == "base" else f"{arm}_r{ratio:g}_s{seed}"
    run = store.create_run(f"{cfg.name}:eval:{tag}", "eval",
                           {"arm": arm, "ratio": ratio, "seed": seed, "adapter": adapter})
    runner = runner_factory(cfg.model, adapter, cfg.gen())
    diag = json.loads((cfg.out / "diagnosis.json").read_text(encoding="utf-8"))
    metrics: dict[str, Any] = {"arm": arm, "ratio": ratio, "seed": seed}
    all_recs: list[ResponseRecord] = []
    for name, exs in eval_sets(cfg).items():
        recs = evaluate(runner, exs)
        for r in recs:
            r.source = name
        all_recs += recs
        summ = summarize(recs)
        metrics[name] = summ["accuracy"]
        if "truncated_share" in summ:
            metrics[f"{name}:truncated_share"] = summ["truncated_share"]
        if name.startswith("probes") and diag.get("target"):
            metrics[f"{name}:target_slice"] = slice_accuracy(recs, diag["target"]["feature"], diag["threshold"])
    store.add_responses(run, all_recs)
    probe_recs = [r for r in all_recs if r.source == "probes_train_families"]
    Q, s, n, _ = item_counts(probe_recs)
    lltm = fit_lltm(Q, s, n, list(FEATURES))
    store.put_artifact(run, "lltm", lltm.to_dict())
    metrics["eta"] = dict(zip(FEATURES, [None if np.isnan(x) else float(x) for x in lltm.eta]))
    store.finish_run(run, metrics=metrics)
    (cfg.out / "evals").mkdir(exist_ok=True)
    (cfg.out / "evals" / f"{tag}.json").write_text(json.dumps({"run_id": run, **metrics}, indent=2), encoding="utf-8")
    return {"run_id": run, **metrics}


# -------------------------------------------------------------------- report
def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Mean difference a - b over paired items, with a 95% CI and a two-sided p-value."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (n_boot, len(diff)))
    boots = diff[idx].mean(axis=1)
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"mean_diff": float(diff.mean()), "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)), "p_value": float(min(1.0, p)), "n": int(len(diff))}


def report(cfg: Config, store: Store, ratio: float | None = None) -> dict:
    """Aggregate eval runs: per-arm means over seeds, and paired item-level comparisons,
    where each item's correctness is first averaged over seeds within an arm."""
    ratio = float(ratio if ratio is not None else cfg.data.get("main_ratio", 3))
    evals = [r for r in store.list_runs("eval") if r["name"].startswith(f"{cfg.name}:") and r["status"] == "done"]
    per_arm_item: dict[str, dict[str, dict[str, list[float]]]] = {}
    table: dict[str, dict[str, list[float]]] = {}
    for r in evals:
        c = r["config"]
        arm = c["arm"]
        if arm not in ("base", "real_only") and c["ratio"] != ratio:
            continue
        for k, v in r["metrics"].items():
            if isinstance(v, (int, float)) and k not in ("ratio", "seed"):
                table.setdefault(arm, {}).setdefault(k, []).append(float(v))
        for resp in store.get_responses(r["id"]):
            per_arm_item.setdefault(arm, {}).setdefault(resp["source"], {}).setdefault(
                resp["example_id"], []).append(float(resp["correct"]))
    summary = {arm: {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                         "n_seeds": len(v)} for k, v in m.items()} for arm, m in table.items()}
    comparisons: dict[str, dict] = {}
    pairs = [("targeted", "matched_control"), ("targeted", "untargeted"), ("matched_control", "untargeted"),
             ("targeted", "real_only")]
    for a, b in pairs:
        if a not in per_arm_item or b not in per_arm_item:
            continue
        for src in sorted(set(per_arm_item[a]) & set(per_arm_item[b])):
            ids = sorted(set(per_arm_item[a][src]) & set(per_arm_item[b][src]))
            if not ids:
                continue
            xa = np.array([np.mean(per_arm_item[a][src][i]) for i in ids])
            xb = np.array([np.mean(per_arm_item[b][src][i]) for i in ids])
            comparisons[f"{a} - {b} | {src}"] = paired_bootstrap(xa, xb)
    result = {"ratio": ratio, "arms": summary, "comparisons": comparisons}
    (cfg.out / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (cfg.out / "report.md").write_text(_report_md(result), encoding="utf-8")
    return result


def _report_md(r: dict) -> str:
    lines = [f"# Results (synthetic:real = {r['ratio']:g}:1)", "", "## Accuracy by arm (mean ± sd over seeds)", ""]
    metrics = sorted({k for m in r["arms"].values() for k in m})
    lines.append("| arm | " + " | ".join(metrics) + " |")
    lines.append("|---|" + "---|" * len(metrics))
    for arm in ("base", *ARMS):
        if arm in r["arms"]:
            cells = [f"{r['arms'][arm][m]['mean']:.3f} ± {r['arms'][arm][m]['sd']:.3f}" if m in r["arms"][arm] else ""
                     for m in metrics]
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired comparisons (item-level bootstrap)", "",
              "| comparison | Δ accuracy | 95% CI | p |", "|---|---|---|---|"]
    for k, c in r["comparisons"].items():
        lines.append(f"| {k} | {c['mean_diff']:+.3f} | [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] | {c['p_value']:.3f} |")
    return "\n".join(lines) + "\n"
