"""Job command line (PLAN.md §7.5).

    python -m dreammachine.jobs run --preset explore --model Qwen/Qwen3-0.6B --limit 50
    python -m dreammachine.jobs run --preset explore --model Qwen/Qwen3-0.6B --target feature:n_carry --reuse-from <id>
    python -m dreammachine.jobs run --preset paper --model Qwen/Qwen3-1.7B
    python -m dreammachine.jobs run --kind benchmark_run --model Qwen/Qwen3-0.6B --limit 300 --no-diagnosis
    python -m dreammachine.jobs enqueue ...   (same flags; only queues the job for the worker)
    python -m dreammachine.jobs list | show <id> | cancel <id> | resume <id> | delete <id>
    python -m dreammachine.jobs.worker        (runs queued jobs, one step per subprocess)

`run` executes every step in this process, in the foreground (no worker). Explore runs are never paper evidence.
"""

from __future__ import annotations

import argparse
import json

from ..store.db import Store
from .errors import JobError
from .execute import run_job
from .requests import create_job_from_request


def _request(args) -> dict:
    if args.kind == "benchmark_run":
        req = {"model": args.model, "include_diagnosis": not args.no_diagnosis}
    elif args.preset == "paper":
        req = {"mode": "paper", "model": args.model}
    else:
        req = {"mode": "explore", "model": args.model}
        if args.target:
            if args.target.startswith("feature:"):
                req["target"] = {"strategy": "feature", "feature": args.target.split(":", 1)[1]}
            else:
                req["target"] = {"strategy": args.target}
        for flag, key in ((args.arms, "arms"), (args.ratio, "ratio"), (args.seeds, "seeds"),
                          (args.reuse_from, "reuse_from")):
            if flag is not None:
                req[key] = flag
    if args.name:
        req["name"] = args.name
    if args.preset != "paper" or args.kind == "benchmark_run":
        if args.benchmarks:
            req["benchmarks"] = args.benchmarks
        if args.full:
            req["benchmark_limit"] = None
        elif args.limit is not None:
            req["benchmark_limit"] = args.limit
    return req


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m dreammachine.jobs", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "enqueue"):
        s = sub.add_parser(name)
        s.add_argument("--kind", choices=["pipeline", "benchmark_run"], default="pipeline")
        s.add_argument("--preset", choices=["explore", "paper"], default="explore")
        s.add_argument("--model", required=True)
        s.add_argument("--name")
        s.add_argument("--benchmarks", nargs="+", help="default: gsm8k gsm_symbolic:main probes_train_families "
                                                       "probes_heldout_families; jsonl:<path> allowed here")
        s.add_argument("--limit", type=int, help="problems per benchmark (>= 10); default from configs/explore.yaml")
        s.add_argument("--full", action="store_true", help="no limit: full benchmark sets")
        s.add_argument("--target", help="auto | untargeted | feature:<name>")
        s.add_argument("--arms", nargs="+")
        s.add_argument("--ratio", type=int)
        s.add_argument("--seeds", type=int, nargs="+")
        s.add_argument("--reuse-from")
        s.add_argument("--no-diagnosis", action="store_true", help="benchmark_run only")
        s.add_argument("--db", default="runs/dreammachine.db")
        s.add_argument("--runs-root", default="runs/pipelines")
    for name, help_ in (("cancel", "cancel a queued or running job"), ("resume", "resume a failed or cancelled job"),
                        ("delete", "delete a finished explore job and its folder"), ("show", "show one job")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("job_id")
        s.add_argument("--db", default="runs/dreammachine.db")
    s = sub.add_parser("list", help="list jobs, newest first")
    s.add_argument("--db", default="runs/dreammachine.db")
    return p


def _summary(store: Store, job_id: str) -> dict:
    job = store.get_job(job_id)
    return {"job_id": job_id, "kind": job["kind"], "mode": job["mode"], "status": job["status"],
            "cancel_requested": job["cancel_requested"], "output_dir": job["output_dir"], "error": job["error"],
            "warnings": job["warnings"],
            "steps": [f"{s['idx']:>2} {s['key']:<34} {s['status']}" for s in store.get_steps(job_id)]}


def main(argv: list[str] | None = None) -> dict:
    from .control import cancel_job, delete_job, resume_job

    args = _parser().parse_args(argv)
    store = Store(args.db)
    try:
        if args.cmd == "list":
            out = {"jobs": [f"{j['id']}  {j['kind']:<13} {j['mode']:<7} {j['status']:<9} {j['name']}"
                            for j in store.list_jobs()]}
        elif args.cmd in ("cancel", "resume", "show"):
            if args.cmd != "show":
                (cancel_job if args.cmd == "cancel" else resume_job)(store, args.job_id)
            elif store.get_job(args.job_id) is None:
                raise JobError("not_found", f"no job {args.job_id!r}")
            out = _summary(store, args.job_id)
        elif args.cmd == "delete":
            out = delete_job(store, args.job_id)
        else:
            job_id = create_job_from_request(store, args.kind, _request(args), runs_root=args.runs_root,
                                             allow_local_models=True, allow_jsonl=True)
            if args.cmd == "run":
                run_job(store, job_id, log=lambda s: print(s, flush=True))
            out = _summary(store, job_id)
    except JobError as e:
        print(json.dumps({"error": e.to_dict()}, indent=2))
        raise SystemExit(2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
