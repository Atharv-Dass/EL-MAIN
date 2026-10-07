"""Human-validation tooling for the error classifier (the '150 stratified errors' step).

1. `stratified_sample` picks errors balanced across predicted types.
2. Two annotators label them independently.
3. `cohen_kappa` measures annotator-vs-annotator and classifier-vs-consensus agreement.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Hashable, Sequence, TypeVar

T = TypeVar("T")


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label sequences")
    labels = sorted(set(a) | set(b), key=str)
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((list(a).count(l) / n) * (list(b).count(l) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def confusion_matrix(a: Sequence[Hashable], b: Sequence[Hashable]) -> dict[Hashable, dict[Hashable, int]]:
    labels = sorted(set(a) | set(b), key=str)
    m = {x: {y: 0 for y in labels} for x in labels}
    for x, y in zip(a, b):
        m[x][y] += 1
    return m


def stratified_sample(records: Sequence[T], key: Callable[[T], Hashable], n: int,
                      seed: int = 0) -> list[T]:
    """Round-robin over strata, so rare error types are represented, not swamped."""
    rng = random.Random(seed)
    strata: dict[Hashable, list[T]] = defaultdict(list)
    for r in records:
        strata[key(r)].append(r)
    for v in strata.values():
        rng.shuffle(v)
    out: list[T] = []
    keys = sorted(strata, key=str)
    while len(out) < n and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(out) < n:
                out.append(strata[k].pop())
    return out
