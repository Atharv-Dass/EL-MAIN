"""Start the API and the worker together (PLAN.md §7.5).

    python -m dreammachine.serve [--port 8000] [--db runs/dreammachine.db] [--no-worker]

The API listens on 127.0.0.1 only (no authentication; never expose it to a network). The worker runs as a child
process; Ctrl+C stops both (the step that was running is marked "interrupted" and can be resumed).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

HOST = "127.0.0.1"   # fixed on purpose: the API has no authentication


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m dreammachine.serve", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--db", default="runs/dreammachine.db")
    p.add_argument("--no-worker", action="store_true", help="API only (start the worker separately)")
    args = p.parse_args(argv)

    import uvicorn

    from .api.app import create_app
    from .store.db import Store

    os.environ["DREAMMACHINE_DB"] = args.db
    worker = None
    if not args.no_worker:
        worker = subprocess.Popen([sys.executable, "-m", "dreammachine.jobs.worker", "--db", args.db])
        print(f"worker started (pid {worker.pid})", flush=True)
    print(f"API on http://{HOST}:{args.port}/api/v1  (docs: http://{HOST}:{args.port}/docs)", flush=True)
    try:
        uvicorn.run(create_app(store=Store(args.db)), host=HOST, port=args.port)
    finally:
        if worker is not None and worker.poll() is None:
            try:
                worker.wait(timeout=60)       # Ctrl+C reached the worker too; let it mark its step interrupted
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=30)
            print("worker stopped", flush=True)


if __name__ == "__main__":
    main()
