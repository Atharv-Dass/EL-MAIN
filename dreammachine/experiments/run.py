"""Command-line entry point.

    python -m dreammachine.experiments.run screen     --config configs/main.yaml
    python -m dreammachine.experiments.run diagnose   --config configs/main.yaml
    python -m dreammachine.experiments.run build-data --config configs/main.yaml
    python -m dreammachine.experiments.run train      --config configs/main.yaml --arm targeted --ratio 3 --seed 0
    python -m dreammachine.experiments.run train      --config configs/main.yaml --all
    python -m dreammachine.experiments.run evaluate   --config configs/main.yaml --arm base
    python -m dreammachine.experiments.run evaluate   --config configs/main.yaml --all
    python -m dreammachine.experiments.run report     --config configs/main.yaml
"""

from __future__ import annotations

import argparse
import json
from functools import partial

from ..store.db import Store
from . import pipeline as P


def _jobs(cfg: P.Config, args) -> list[tuple[str, float, int]]:
    if not args.all:
        return [(args.arm, float(args.ratio), int(args.seed))]
    ratios = [float(r) for r in cfg.data.get("ratios", [3])]
    main = float(cfg.data.get("main_ratio", ratios[0]))
    jobs = []
    for seed in cfg.seeds:
        jobs.append(("real_only", 0.0, seed))
        for arm in ("untargeted", "matched_control", "targeted"):
            for ratio in (ratios if arm == "targeted" else [main]):
                jobs.append((arm, ratio, seed))
    return jobs


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="dreammachine", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["screen", "diagnose", "build-data", "train", "evaluate", "report"])
    p.add_argument("--config", required=True)
    p.add_argument("--arm", default="base", choices=["base", *P.ARMS])
    p.add_argument("--ratio", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--all", action="store_true", help="every (arm, ratio, seed) job in the config")
    args = p.parse_args(argv)

    cfg = P.Config.load(args.config)
    store = Store(cfg.db)
    factory = partial(P.hf_runner_factory, load_in_4bit=bool(cfg.eval.get("load_in_4bit", False)))

    if args.stage == "screen":
        out = P.screen(cfg, store, factory)
    elif args.stage == "diagnose":
        out = P.diagnose(cfg, store, factory)
        out = {k: out[k] for k in ("run_id", "target", "threshold", "weaknesses")}
    elif args.stage == "build-data":
        out = P.build_data(cfg, store)
    elif args.stage == "train":
        if args.arm == "base":
            p.error("train needs --arm (not base) or --all")
        out = {f"{a}_r{r:g}_s{s}": str(P.train_arm(cfg, store, a, r, s)) for a, r, s in _jobs(cfg, args)}
    elif args.stage == "evaluate":
        if args.all:
            out = [P.evaluate_model(cfg, store, factory, a, r, s) for a, r, s in _jobs(cfg, args)]
        else:
            out = P.evaluate_model(cfg, store, factory, args.arm,
                                   None if args.arm == "base" else args.ratio,
                                   None if args.arm == "base" else args.seed)
    else:
        out = P.report(cfg, store)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
