"""Built-in benchmarks (PLAN.md §8.3). They wrap the existing loaders; the logic stays in data/ and generator/.

Loader options (passed by `experiments.pipeline.eval_sets` from the config):
    gsm8k, gsm_symbolic:*      path=<local JSONL of Examples> (offline override), else Hugging Face
    probes_*                   grid=ProbeGrid, per_cell=int (probe seeds are fixed: 777 train, 778 held-out)

`limit`: GSM8K / GSM-Symbolic take the first `limit` problems (as before). Probe sets take a random subset of
`limit` problems (seeded by `seed`, original order kept), so every grid cell can still appear.
"""

from __future__ import annotations

import random

from ..data.loaders import Example, from_items, load_gsm8k, load_gsm_symbolic, load_jsonl
from ..generator.probes import ProbeGrid, factorial_probe_set
from .base import Benchmark
from .registry import _REGISTRY, register

# Full sizes, counted on the live Hugging Face datasets (2026-10-04).
GSM8K_TEST_SIZE = 1319
GSM_SYMBOLIC_SIZES = {"main": 5000, "p1": 5000, "p2": 2500}
PROBE_SEEDS = {"train": 777, "heldout": 778}   # unchanged from the pre-registry eval_sets


def _gsm8k(limit: int | None = None, seed: int = 0, path: str | None = None) -> list[Example]:
    exs = load_jsonl(path) if path else load_gsm8k("test", limit=limit)
    return exs[:limit]


def spread_templates(exs: list[Example]) -> list[Example]:
    """GSM-Symbolic lists its rows grouped by template (all ~50 instances of template 0, then template 1, ...), so the
    first N rows cover only N/50 templates (B8: the first 50 were one problem, gold answer always 20). Reorder
    round-robin: instance 0 of every template, then instance 1 of every template, ... so any limit covers as many
    templates as possible. Deterministic; rows without template metadata keep their order."""
    if not exs or not all("original_id" in e.meta and "instance" in e.meta for e in exs):
        return exs
    order = {}
    for e in exs:
        order.setdefault(e.meta["original_id"], len(order))       # templates in their original order
    return sorted(exs, key=lambda e: (int(e.meta["instance"]), order[e.meta["original_id"]]))


def _gsm_symbolic(variant: str):
    def loader(limit: int | None = None, seed: int = 0, path: str | None = None) -> list[Example]:
        exs = load_jsonl(path) if path else load_gsm_symbolic(variant)   # all rows, then spread across templates
        return spread_templates(exs)[:limit]
    return loader


def _probes(split: str):
    def loader(limit: int | None = None, seed: int = 0, grid: ProbeGrid | None = None,
               per_cell: int = 2) -> list[Example]:
        exs = from_items(factorial_probe_set(grid or ProbeGrid(), per_cell=per_cell, seed=PROBE_SEEDS[split],
                                             split=split))
        if limit is not None and limit < len(exs):
            keep = sorted(random.Random(seed).sample(range(len(exs)), limit))
            exs = [exs[i] for i in keep]
        return exs
    return loader


BUILTINS = [
    Benchmark("gsm8k", f"GSM8K test set ({GSM8K_TEST_SIZE} grade-school word problems).", "external", _gsm8k,
              size=GSM8K_TEST_SIZE),
    Benchmark("gsm_symbolic:main", "GSM-Symbolic: GSM8K templates with new names and numbers.", "external",
              _gsm_symbolic("main"), size=GSM_SYMBOLIC_SIZES["main"], version=2),
    Benchmark("gsm_symbolic:p1", "GSM-Symbolic P1: one extra clause per problem.", "external",
              _gsm_symbolic("p1"), size=GSM_SYMBOLIC_SIZES["p1"], version=2),
    Benchmark("gsm_symbolic:p2", "GSM-Symbolic P2: two extra clauses per problem.", "external",
              _gsm_symbolic("p2"), size=GSM_SYMBOLIC_SIZES["p2"], version=2),
    Benchmark("probes_train_families", "Generated probe problems in the story styles used for training.",
              "synthetic", _probes("train")),
    Benchmark("probes_heldout_families", "Generated probe problems in story styles never used for training.",
              "synthetic", _probes("heldout")),
]
for _b in BUILTINS:
    if _b.name not in _REGISTRY:
        register(_b)
