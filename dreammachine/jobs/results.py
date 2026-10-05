"""The final comparison (PLAN.md §9; contract `PipelineResults` and `CompareResult`).

Everything is computed from stored eval runs (responses + metrics), so results can be built at any time:
partially while a pipeline runs (`complete: false`) and finally by the compare step (results.json, report.md).

    9.1 before vs after   per arm (and ratio): each problem's correctness is averaged over seeds, then a paired
                          bootstrap against the base model on the shared problems; regression = ci_high < 0
    9.2 arm vs arm        the pairs of experiments.pipeline.report(), only at the same ratio (real_only: any ratio)
    9.3 eta change        eta_after - eta_before per feature (negative = the feature hurts less after training);
                          error-type shift (shares among wrong answers, after - before); target slice;
                          primary block (targeted - matched_control on GSM8K / GSM-Symbolic; status "proposed")
    compare_runs          any two eval runs (GET /compare)
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..diagnosis.taxonomy import ErrorType
from ..experiments.pipeline import ARMS, paired_bootstrap
from ..generator.features import FEATURES
from ..store.db import Store
from .errors import JobError

ERROR_TYPES = [e.value for e in ErrorType if e is not ErrorType.CORRECT]
PAIRS = [("targeted", "matched_control"), ("targeted", "untargeted"), ("matched_control", "untargeted"),
         ("targeted", "real_only")]
PRIMARY_BENCHMARKS = ["gsm8k", "gsm_symbolic:main", "gsm_symbolic:p1", "gsm_symbolic:p2"]
PRIMARY_QUESTION = "Does aiming at the weakness help beyond difficulty?"
PAPER_REQUEST_KEYS = {"mode", "name", "model"}


def _num(x: float | int | None) -> float | int | None:
    """3.0 -> 3 (ratios read like the contract); None stays None."""
    if x is None:
        return None
    return int(x) if float(x).is_integer() else float(x)


# ------------------------------------------------------------------ run data
class RunData:
    """One eval run's answers, grouped for the comparisons."""

    def __init__(self, store: Store, run_id: str) -> None:
        self.run = store.get_run(run_id)
        if self.run is None:
            raise JobError("not_found", f"no run {run_id!r}")
        self.id = run_id
        cfg, metrics = self.run["config"], self.run["metrics"] or {}
        self.arm, self.ratio, self.seed = cfg.get("arm"), cfg.get("ratio"), cfg.get("seed")
        self.eta = metrics.get("eta")
        self.items: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        self.features: dict[tuple[str, str], dict] = {}
        self.wrong: Counter = Counter()
        for r in store.get_responses(run_id):
            self.items[r["source"]][r["example_id"]].append(float(r["correct"]))
            self.features[(r["source"], r["example_id"])] = r["features"]
            if not r["correct"]:
                self.wrong[r["error_type"]] += 1

    def item_means(self, bench: str) -> dict[str, float]:
        return {ex: float(np.mean(v)) for ex, v in self.items.get(bench, {}).items()}

    def accuracy(self, bench: str) -> float | None:
        vals = [c for v in self.items.get(bench, {}).values() for c in v]
        return float(np.mean(vals)) if vals else None


def _mean_over_seeds(runs: list[RunData], bench: str) -> dict[str, float]:
    """Each problem's correctness, averaged first within a run, then over seeds (as report() does)."""
    per: dict[str, list[float]] = defaultdict(list)
    for r in runs:
        for ex, v in r.item_means(bench).items():
            per[ex].append(v)
    return {ex: float(np.mean(v)) for ex, v in per.items()}


def _paired(a: dict[str, float], b: dict[str, float]) -> dict | None:
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    return paired_bootstrap(np.array([a[i] for i in ids]), np.array([b[i] for i in ids]))


def _significant(bs: dict) -> bool:
    return bs["ci_low"] > 0 or bs["ci_high"] < 0


def error_shares(counter: Counter) -> dict[str, float]:
    n = sum(counter.values())
    return {k: counter[k] / n for k in ERROR_TYPES if counter.get(k)} if n else {}


def error_shift(after: Counter, before: Counter) -> dict[str, float]:
    a, b = error_shares(after), error_shares(before)
    return {k: round(a.get(k, 0.0) - b.get(k, 0.0), 6) for k in ERROR_TYPES if k in a or k in b}


def eta_change(after: list[dict | None], before: dict | None) -> dict[str, float | None]:
    """Mean eta over seeds after training, minus eta before; None where either is unknown."""
    out: dict[str, float | None] = {}
    for f in FEATURES:
        b = (before or {}).get(f)
        vals = [e.get(f) for e in after if e and e.get(f) is not None]
        out[f] = None if b is None or len(vals) != len(after) or not vals else round(float(np.mean(vals)) - b, 6)
    return out


def _slice(item_means: dict[str, float], feats: dict, bench: str, feature: str, threshold: float
           ) -> tuple[dict[str, list[float]], dict[str, int]]:
    groups: dict[str, list[float]] = {"above": [], "at_or_below": []}
    for ex, v in item_means.items():
        f = feats.get((bench, ex)) or {}
        if feature not in f:
            continue
        groups["above" if f[feature] > threshold else "at_or_below"].append(v)
    return groups, {k: len(v) for k, v in groups.items()}


# ------------------------------------------------------------- job results
def _job_runs(store: Store, job: dict) -> tuple[RunData | None, list[RunData], bool]:
    """(base run, evaluated arm runs, compare step done) from the job's steps."""
    base, arms, compare_done = None, [], False
    for s in store.get_steps(job["id"]):
        if s["stage"] == "baseline" and s["status"] in ("done", "skipped") and s["run_ids"]:
            base = RunData(store, s["run_ids"][0])
        elif s["stage"] == "evaluate" and s["status"] == "done" and s["run_ids"]:
            arms.append(RunData(store, s["run_ids"][0]))
        elif s["stage"] == "compare" and s["status"] == "done":
            compare_done = True
    return base, arms, compare_done


def paper_eligibility(store: Store, job: dict, treat_compare_done: bool = False) -> tuple[bool, list[str]]:
    """paper_eligible = paper mode, every step done, no overrides, clean git tree (PLAN.md §9.4)."""
    if job["mode"] != "paper":
        return False, []
    warnings = []
    steps = store.get_steps(job["id"])
    all_done = all(s["status"] == "done" or (treat_compare_done and s["stage"] == "compare") for s in steps)
    overrides = set(job["request"] or {}) - PAPER_REQUEST_KEYS
    dirty = ((job.get("provenance") or {}).get("git") or {}).get("dirty")
    if dirty is not False:
        warnings.append("the git working tree was dirty (or unknown) when this paper run started: not paper-eligible")
    if overrides:
        warnings.append(f"request overrides {sorted(overrides)}: not paper-eligible")
    return bool(all_done and not overrides and dirty is False), warnings


def build_results(store: Store, job_id: str, complete: bool | None = None) -> dict:
    """PipelineResults (contract §7). Raises JobError not_ready before any evaluate step is done."""
    from ..benchmarks import registry

    job = store.get_job(job_id)
    if job is None or job["kind"] != "pipeline":
        raise JobError("not_found", f"no pipeline {job_id!r}")
    base, arm_runs, compare_done = _job_runs(store, job)
    if base is None or not arm_runs:
        raise JobError("not_ready", "results need the baseline and at least one finished evaluate step")
    complete = compare_done if complete is None else complete
    resolved = job["resolved"]
    benchmarks = list(resolved["benchmarks"])
    tgt_path = Path(job["output_dir"]) / "target.json"
    tgt = json.loads(tgt_path.read_text(encoding="utf-8")) if tgt_path.exists() else None
    target = {"strategy": tgt["strategy"], "feature": tgt["feature"], "threshold": tgt["threshold"]} if tgt else None

    groups: dict[tuple[str, float], list[RunData]] = defaultdict(list)
    for r in arm_runs:
        groups[(r.arm, float(r.ratio))].append(r)
    order = {a: i for i, a in enumerate(ARMS)}
    keys = sorted(groups, key=lambda k: (order.get(k[0], 99), k[1]))
    base_items = {b: base.item_means(b) for b in benchmarks}

    arms_out, regressions = [], []
    for arm, ratio in keys:
        runs = sorted(groups[(arm, ratio)], key=lambda r: r.seed)
        bench_out = {}
        for b in benchmarks:
            accs = [a for a in (r.accuracy(b) for r in runs) if a is not None]
            bs = _paired(_mean_over_seeds(runs, b), base_items[b])
            if not accs or bs is None:
                continue
            reg = bs["ci_high"] < 0
            bench_out[b] = {"accuracy_mean": float(np.mean(accs)),
                            "accuracy_sd": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                            "n_seeds": len(accs), "vs_base": bs, "regression": reg}
            if reg:
                regressions.append({"arm": arm, "ratio": _num(ratio), "benchmark": b, "mean_diff": bs["mean_diff"],
                                    "ci_low": bs["ci_low"], "ci_high": bs["ci_high"]})
        slices = {}
        if target and target["feature"]:
            for b in [x for x in benchmarks if x.startswith("probes")]:
                after = _mean_over_seeds(runs, b)
                ga, n = _slice(after, {**base.features, **{k: v for r in runs for k, v in r.features.items()}},
                               b, target["feature"], target["threshold"])
                gb, _ = _slice(base_items[b], base.features, b, target["feature"], target["threshold"])
                slices[b] = {k: {"before": float(np.mean(gb[k])) if gb[k] else None,
                                 "after": float(np.mean(ga[k])) if ga[k] else None, "n": n[k]}
                             for k in ("above", "at_or_below")}
        wrong_after = sum((r.wrong for r in runs), Counter())
        arms_out.append({"arm": arm, "ratio": _num(ratio), "seeds": [r.seed for r in runs], "benchmarks": bench_out,
                         "target_slice": slices, "eta_change": eta_change([r.eta for r in runs], base.eta),
                         "error_type_shift": error_shift(wrong_after, base.wrong)})

    comparisons = []
    for a, b in PAIRS:
        for (arm_a, ra) in [k for k in keys if k[0] == a]:
            for (arm_b, rb) in [k for k in keys if k[0] == b]:
                if b != "real_only" and ra != rb:
                    continue
                for bench in benchmarks:
                    bs = _paired(_mean_over_seeds(groups[(arm_a, ra)], bench),
                                 _mean_over_seeds(groups[(arm_b, rb)], bench))
                    if bs is not None:
                        comparisons.append({"a": a, "b": b, "ratio": _num(ra), "benchmark": bench, **bs,
                                            "significant": _significant(bs)})

    main_ratio = resolved.get("main_ratio")
    prim_rows = [c for c in comparisons if c["a"] == "targeted" and c["b"] == "matched_control"
                 and c["ratio"] == _num(main_ratio) and c["benchmark"] in PRIMARY_BENCHMARKS]
    has_pair = any(k[0] == "targeted" for k in keys) and any(k[0] == "matched_control" for k in keys)
    primary = None
    if has_pair:
        primary = {"status": "proposed", "question": PRIMARY_QUESTION, "a": "targeted", "b": "matched_control",
                   "benchmarks": [b for b in PRIMARY_BENCHMARKS if b in benchmarks],
                   "results": [{k: c[k] for k in ("benchmark", "mean_diff", "ci_low", "ci_high", "p_value", "n",
                                                  "significant")} for c in prim_rows]}

    eligible, warn = paper_eligibility(store, job, treat_compare_done=complete)
    warnings = list(job["warnings"] or []) + warn
    requested = resolved.get("seeds") or []
    for arm, ratio in keys:
        got = {r.seed for r in groups[(arm, ratio)]}
        if complete and requested and got != set(requested):
            warnings.append(f"{arm} ratio {_num(ratio)}: results for seeds {sorted(got)} of {sorted(requested)}")

    return {
        "pipeline_id": job_id, "complete": bool(complete), "mode": job["mode"], "paper_eligible": eligible,
        "model": resolved["model"], "target": target,
        "benchmarks": [{"name": b, "kind": registry.get(b).kind} for b in benchmarks],
        "baseline": {b: {"accuracy": base.accuracy(b), "n": len(base_items[b])} for b in benchmarks
                     if base.accuracy(b) is not None},
        "arms": arms_out, "comparisons": comparisons, "primary": primary, "regressions": regressions,
        "warnings": warnings,
    }


# ---------------------------------------------------------------- compare
def compare_runs(store: Store, before_id: str, after_id: str) -> dict:
    """CompareResult (contract §6.4) for any two eval runs."""
    before, after = RunData(store, before_id), RunData(store, after_id)
    for r in (before, after):
        if r.run["kind"] != "eval":
            raise JobError("validation_error", f"run {r.id} is a {r.run['kind']} run; both runs must be eval runs")
    warnings = []
    for key, what in (("eval_load_in_4bit", "precision"), ("prompt_version", "prompt version"), ("model", "model")):
        va, vb = before.run["config"].get(key), after.run["config"].get(key)
        if va != vb:
            warnings.append(f"runs used different {what}: {va!r} vs {vb!r}")
    rows = []
    for b in [x for x in before.items if x in after.items]:
        bs = _paired(after.item_means(b), before.item_means(b))
        if bs is None:
            continue
        rows.append({"name": b, "before": before.accuracy(b), "after": after.accuracy(b), **bs,
                     "regression": bs["ci_high"] < 0})
        if bs["n"] < min(len(before.items[b]), len(after.items[b])):
            warnings.append(f"{b}: only {bs['n']} problems are in both runs")
    return {"before": before_id, "after": after_id, "benchmarks": rows,
            "eta_change": eta_change([after.eta], before.eta) if before.eta and after.eta else {},
            "error_type_shift": error_shift(after.wrong, before.wrong), "warnings": warnings}


# --------------------------------------------------------------- report.md
def _f(x: float | None, signed: bool = False) -> str:
    return "—" if x is None else (f"{x:+.3f}" if signed else f"{x:.3f}")


def results_markdown(res: dict, job: dict) -> str:
    mode = "PAPER" if res["mode"] == "paper" else "EXPLORE — not paper evidence"
    t = res["target"]
    lines = [f"# DreamMachine results — {job['name']}", "",
             f"**Mode:** {mode}  ·  **Paper-eligible:** {'yes' if res['paper_eligible'] else 'no'}  ·  "
             f"**Complete:** {'yes' if res['complete'] else 'no (partial)'}", "",
             f"**Model:** `{res['model']}`  ·  **Prompt:** `{job['resolved'].get('prompt_version')}`  ·  "
             f"**Target:** " + (f"`{t['feature']}` (threshold {t['threshold']}, {t['strategy']})" if t and t["feature"]
                                else (t["strategy"] if t else "—")), "",
             "## Baseline (untrained model)", "", "| benchmark | accuracy | n |", "|---|---|---|"]
    for b, v in res["baseline"].items():
        lines.append(f"| {b} | {_f(v['accuracy'])} | {v['n']} |")
    if res["primary"]:
        p = res["primary"]
        lines += ["", f"## Primary result (status: {p['status']})", "", f"_{p['question']}_ "
                  f"{p['a']} − {p['b']}", "", "| benchmark | Δ accuracy | 95% CI | p | n | significant |",
                  "|---|---|---|---|---|---|"]
        for r in p["results"]:
            lines.append(f"| {r['benchmark']} | {_f(r['mean_diff'], True)} | [{_f(r['ci_low'], True)}, "
                         f"{_f(r['ci_high'], True)}] | {r['p_value']:.3f} | {r['n']} | {'yes' if r['significant'] else 'no'} |")
    lines += ["", "## Before vs after (each arm vs the untrained model)", "",
              "| arm | ratio | benchmark | accuracy (mean ± sd) | seeds | Δ vs base | 95% CI | regression |",
              "|---|---|---|---|---|---|---|---|"]
    for a in res["arms"]:
        for b, v in a["benchmarks"].items():
            vb = v["vs_base"]
            lines.append(f"| {a['arm']} | {a['ratio']} | {b} | {v['accuracy_mean']:.3f} ± {v['accuracy_sd']:.3f} | "
                         f"{v['n_seeds']} | {_f(vb['mean_diff'], True)} | [{_f(vb['ci_low'], True)}, "
                         f"{_f(vb['ci_high'], True)}] | {'**yes**' if v['regression'] else 'no'} |")
    lines += ["", "## Arm vs arm (paired, item-level bootstrap)", "",
              "| comparison | ratio | benchmark | Δ accuracy | 95% CI | p | significant |", "|---|---|---|---|---|---|---|"]
    for c in res["comparisons"]:
        lines.append(f"| {c['a']} − {c['b']} | {c['ratio']} | {c['benchmark']} | {_f(c['mean_diff'], True)} | "
                     f"[{_f(c['ci_low'], True)}, {_f(c['ci_high'], True)}] | {c['p_value']:.3f} | "
                     f"{'yes' if c['significant'] else 'no'} |")
    lines += ["", "## Change in weaknesses (η after − before; negative = hurts less)", "",
              "| arm | ratio | " + " | ".join(FEATURES) + " |", "|---|---|" + "---|" * len(FEATURES)]
    for a in res["arms"]:
        lines.append(f"| {a['arm']} | {a['ratio']} | " + " | ".join(_f(a["eta_change"].get(f), True) for f in FEATURES)
                     + " |")
    lines += ["", "## Error-type shift among wrong answers (after − before)", "",
              "| arm | ratio | " + " | ".join(ERROR_TYPES) + " |", "|---|---|" + "---|" * len(ERROR_TYPES)]
    for a in res["arms"]:
        lines.append(f"| {a['arm']} | {a['ratio']} | " +
                     " | ".join(_f(a["error_type_shift"].get(e), True) for e in ERROR_TYPES) + " |")
    if res["regressions"]:
        lines += ["", "## Regressions (worse than the untrained model; whole 95% CI below 0)", ""]
        lines += [f"- {r['arm']} ratio {r['ratio']} on {r['benchmark']}: {_f(r['mean_diff'], True)} "
                  f"[{_f(r['ci_low'], True)}, {_f(r['ci_high'], True)}]" for r in res["regressions"]]
    if res["warnings"]:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in res["warnings"]]
    prov = job.get("provenance") or {}
    git = prov.get("git") or {}
    lines += ["", "## Provenance", "",
              f"git {git.get('commit', 'unknown')} (dirty: {git.get('dirty')}) · model revision "
              f"{prov.get('model_revision')} · precision {prov.get('precision')} · {prov.get('platform')} · "
              f"GPU {prov.get('gpu')}"]
    return "\n".join(lines) + "\n"


def write_results(store: Store, job_id: str) -> dict:
    """The compare step's output: results.json (complete) and report.md in the job folder."""
    res = build_results(store, job_id, complete=True)
    job = store.get_job(job_id)
    out = Path(job["output_dir"])
    (out / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out / "report.md").write_text(results_markdown(res, job), encoding="utf-8")
    return res
