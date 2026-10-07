"""Benchmark registry and standalone runner (PLAN.md §8). See docs/BENCHMARKS.md."""

from .base import Benchmark
from .registry import available, get, load_plugins, register, unregister

__all__ = ["Benchmark", "available", "get", "load_plugins", "register", "unregister"]
