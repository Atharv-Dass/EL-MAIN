"""The ordered step list of a job (PLAN.md §5).

pipeline:       preflight, baseline, diagnose, choose_target, build_data,
                then per seed and arm key: train:<key>, evaluate:<key> (interleaved, so results appear early),
                then compare
benchmark_run:  preflight, baseline, and diagnose if include_diagnosis
"""

from __future__ import annotations

STAGES = ("preflight", "baseline", "diagnose", "choose_target", "build_data", "train", "evaluate", "compare")
STAGE_LABELS = {"preflight": "Check setup", "baseline": "Benchmark the untrained model",
                "diagnose": "Find weaknesses", "choose_target": "Choose the target",
                "build_data": "Build training data", "train": "Train", "evaluate": "Evaluate",
                "compare": "Compare results"}
PROGRESS_UNIT = {"baseline": "examples", "diagnose": "examples", "train": "steps", "evaluate": "examples"}
ARM_ORDER = ("real_only", "untargeted", "matched_control", "targeted")   # as experiments/run.py::_jobs


def arm_key(arm: str, ratio: float, seed: int) -> str:
    return f"{arm}_r{float(ratio):g}_s{seed}"


def arm_keys(arms: list[str], ratios: list[float], main_ratio: float, seeds: list[int],
             paper: bool) -> list[tuple[str, float, int]]:
    """(arm, ratio, seed) in run order. Paper: real_only at 0, untargeted / matched_control at the main ratio,
    targeted at every ratio (18 keys for 3 seeds). Explore: each requested arm at its one ratio."""
    out = []
    for seed in seeds:
        for arm in ARM_ORDER:
            if arm not in arms:
                continue
            if arm == "real_only":
                out.append((arm, 0.0, seed))
            elif paper and arm == "targeted":
                out += [(arm, float(r), seed) for r in ratios]
            else:
                out.append((arm, float(main_ratio), seed))
    return out


def label(stage: str, params: dict | None = None) -> str:
    p = params or {}
    if stage in ("train", "evaluate"):
        return f"{STAGE_LABELS[stage]}: {p['arm'].replace('_', ' ')}, ratio {p['ratio']:g}, seed {p['seed']}"
    return STAGE_LABELS[stage]


def _step(stage: str, key: str | None = None, **params) -> dict:
    return {"key": key or stage, "stage": stage, "params": params,
            "progress_unit": PROGRESS_UNIT.get(stage, "items")}


def build_steps(kind: str, resolved: dict) -> list[dict]:
    if kind == "benchmark_run":
        steps = [_step("preflight"), _step("baseline")]
        if resolved.get("include_diagnosis", True):
            steps.append(_step("diagnose"))
        return steps
    steps = [_step(s) for s in ("preflight", "baseline", "diagnose", "choose_target", "build_data")]
    keys = arm_keys(resolved["arms"], resolved["ratios"], resolved["main_ratio"], resolved["seeds"],
                    paper=resolved["mode"] == "paper")
    for arm, ratio, seed in keys:
        k = arm_key(arm, ratio, seed)
        steps.append(_step("train", f"train:{k}", arm=arm, ratio=ratio, seed=seed))
        steps.append(_step("evaluate", f"evaluate:{k}", arm=arm, ratio=ratio, seed=seed))
    steps.append(_step("compare"))
    return steps
