"""Benchmark command line (PLAN.md §8.4-8.5).

    python -m dreammachine.benchmarks list
    python -m dreammachine.benchmarks run --model Qwen/Qwen3-0.6B --benchmark gsm8k --limit 200
    python -m dreammachine.benchmarks run --model Qwen/Qwen3-0.6B --benchmark gsm8k --limit 200 --prompt-version v2_700
    python -m dreammachine.benchmarks import-jsonl results.jsonl --benchmark gsm8k --model Qwen/Qwen2.5-1.5B-Instruct

`run` writes runs/benchmarks/<model>__<benchmark>__<prompt>.jsonl (append-only, resumable) and stores one
eval run per benchmark in the SQLite store.
"""

from __future__ import annotations

import argparse
import json
from typing import Callable

from ..models.prompts import DEFAULT_PROMPT, PROMPTS
from ..models.runner import GenConfig, Runner
from . import registry
from .runner import run_benchmark

RunnerFactory = Callable[[str, GenConfig, bool], Runner]


def _hf_runner(model: str, gen: GenConfig, load_in_4bit: bool) -> Runner:
    from ..models.runner import HFRunner

    return HFRunner(model, gen=gen, load_in_4bit=load_in_4bit)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m dreammachine.benchmarks", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="registered benchmarks")

    r = sub.add_parser("run", help="benchmark one model")
    r.add_argument("--model", required=True)
    r.add_argument("--benchmark", action="append", help="repeatable; default gsm8k; jsonl:<path> allowed")
    r.add_argument("--limit", type=int, default=None, help="problems per benchmark (default: all)")
    r.add_argument("--seed", type=int, default=0, help="subset seed for probe sets")
    r.add_argument("--prompt-version", default=DEFAULT_PROMPT, choices=sorted(PROMPTS))
    r.add_argument("--max-new-tokens", type=int, default=512)
    r.add_argument("--batch-size", type=int, default=32, help="sequences per batch")
    r.add_argument("--n-samples", type=int, default=1)
    r.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    r.add_argument("--load-in-4bit", action="store_true")
    r.add_argument("--config", help="optional experiment config: local data overrides and probe grid")
    r.add_argument("--out", default="runs/benchmarks")
    r.add_argument("--db", default="runs/dreammachine.db")

    i = sub.add_parser("import-jsonl", help="import and re-score an existing results file")
    i.add_argument("file")
    i.add_argument("--benchmark", required=True, help="benchmark the file answers (for gold answers)")
    i.add_argument("--model", default=None, help="model id, if the rows do not carry one")
    i.add_argument("--prompt-version", default=None, help="prompt version, if the rows do not carry one")
    i.add_argument("--limit", type=int, default=None)
    i.add_argument("--id-field", default=None, help="row field with the example id or index (auto-detected)")
    i.add_argument("--text-field", default=None, help="row field with the model's answer (auto-detected)")
    i.add_argument("--db", default="runs/dreammachine.db")
    i.add_argument("--config", help="optional experiment config: local data overrides")
    return p


def _options(cfg, name: str) -> dict:
    if cfg is None:
        return {}
    from ..experiments.pipeline import benchmark_options

    return {k: v for k, v in benchmark_options(cfg, name).items() if k != "limit"}


def cmd_run(args, runner_factory: RunnerFactory) -> dict:
    from ..experiments.pipeline import Config
    from ..store.db import Store

    cfg = Config.load(args.config) if args.config else None
    gen = GenConfig(max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, n_samples=args.n_samples,
                    temperature=args.temperature, prompt_version=args.prompt_version)
    names = args.benchmark or ["gsm8k"]
    benches = [registry.get(n) for n in names]          # fail on a typo before loading the model
    store = Store(args.db)
    runner = runner_factory(args.model, gen, args.load_in_4bit)
    out = {}
    for b in benches:
        exs = b.load(limit=args.limit, seed=args.seed, **_options(cfg, b.name))
        res = run_benchmark(b, exs, runner, args.model, gen, args.out)
        run = store.create_run(f"bench:{args.model}:{b.name}:{gen.prompt_version}", "eval", {
            "arm": "base", "model": args.model, "benchmark": b.name, "limit": args.limit,
            "prompt_version": gen.prompt_version, "generation": {k: v for k, v in vars(gen).items()},
            "load_in_4bit": args.load_in_4bit, "results_file": res["path"], "source": "benchmarks.run"})
        store.add_responses(run, res["records"])
        s = res["summary"]
        metrics = {b.name: s.get("accuracy"), "error_distribution": s.get("error_distribution"),
                   "format_ok_share": s.get("format_ok_share"), "truncated_share": s.get("truncated_share"),
                   "n_examples": s.get("n_examples")}
        store.finish_run(run, metrics=metrics)
        out[b.name] = {"run_id": run, "file": res["path"], **metrics}
    return out


def main(argv: list[str] | None = None, runner_factory: RunnerFactory = _hf_runner) -> dict | list:
    args = _parser().parse_args(argv)
    if args.cmd == "list":
        out: dict | list = [b.info() for b in registry.available()]
    elif args.cmd == "run":
        out = cmd_run(args, runner_factory)
    else:
        from .importer import cmd_import

        out = cmd_import(args)
    print(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    main()
