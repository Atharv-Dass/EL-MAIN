"""Benchmark registry (PLAN.md §8.2).

    register(benchmark)   add one (built-ins and plug-ins call this)
    get(name)             look one up; "jsonl:<path>" is resolved on the fly
    available()           every registered benchmark, in registration order

Plug-ins: any .py file in `dreammachine/benchmarks/contrib/` that calls `register(...)` at import time is
loaded on first use (see docs/BENCHMARKS.md).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Iterable

from ..data.loaders import load_jsonl
from .base import Benchmark

_REGISTRY: dict[str, Benchmark] = {}
_plugins_loaded = False
CONTRIB_DIR = Path(__file__).resolve().parent / "contrib"


def register(benchmark: Benchmark, replace: bool = False) -> Benchmark:
    if benchmark.name.startswith("jsonl:"):
        raise ValueError("names starting with 'jsonl:' are reserved for local files")
    if benchmark.name in _REGISTRY and not replace:
        raise ValueError(f"benchmark {benchmark.name!r} is already registered")
    _REGISTRY[benchmark.name] = benchmark
    return benchmark


def unregister(name: str) -> None:
    _REGISTRY.pop(name, None)


def _jsonl_loader(limit: int | None = None, seed: int = 0, path: str = "") -> list:
    return load_jsonl(path)[:limit]


def get(name: str) -> Benchmark:
    _ensure_loaded()
    if name.startswith("jsonl:"):
        path = name[len("jsonl:"):]
        if not path:
            raise KeyError("jsonl: needs a file path, e.g. jsonl:data/my_set.jsonl")
        return Benchmark(name=name, description=f"Local JSONL file {path}", kind="external",
                         loader=_jsonl_loader, options={"path": path})
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown benchmark {name!r}; available: {[b.name for b in available()]} "
                       "or jsonl:<path>") from None


def available() -> list[Benchmark]:
    _ensure_loaded()
    return list(_REGISTRY.values())


def load_plugins(paths: Iterable[Path] | None = None) -> list[str]:
    """Import plug-in files (default: every .py in contrib/ except __init__). Returns the module names."""
    files = sorted(CONTRIB_DIR.glob("*.py")) if paths is None else [Path(p) for p in paths]
    loaded = []
    for f in files:
        if f.name.startswith("_"):
            continue
        mod_name = f"dreammachine.benchmarks.contrib.{f.stem}"
        if mod_name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(mod_name, f)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            del sys.modules[mod_name]
            raise
        loaded.append(mod_name)
    return loaded


def _ensure_loaded() -> None:
    global _plugins_loaded
    from . import builtin  # noqa: F401  (registers the built-ins on import)

    if not _plugins_loaded:
        _plugins_loaded = True
        load_plugins()
