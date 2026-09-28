"""Q-matrix features: the exact skill/difficulty profile of a generated item.

These are the columns of the LLTM design matrix (see diagnosis/lltm.py).
They are computed from the DAG, never estimated.
"""

from __future__ import annotations

import math

from .dag import DAG, Op

FEATURES: tuple[str, ...] = (
    "steps",          # number of arithmetic operations (reasoning length)
    "log10_max",      # log10 of the largest number anywhere in the computation (magnitude)
    "n_mul",          # multiplication steps
    "n_div",          # division steps
    "n_carry",        # column carries (add) + borrows (sub) across all add/sub steps
    "n_distractors",  # irrelevant numeric sentences in the question
    "has_merge",      # 1 if two sub-computations are combined (tree, not chain)
)


def count_carries(a: int, b: int) -> int:
    """Number of column carries when adding two non-negative integers."""
    carries = carry = 0
    while a or b:
        s = a % 10 + b % 10 + carry
        carry = 1 if s >= 10 else 0
        carries += carry
        a //= 10
        b //= 10
    return carries


def count_borrows(a: int, b: int) -> int:
    """Number of column borrows when computing a - b for integers a >= b >= 0."""
    if a < b:
        raise ValueError("count_borrows expects a >= b")
    borrows = borrow = 0
    while a or b:
        d = a % 10 - b % 10 - borrow
        borrow = 1 if d < 0 else 0
        borrows += borrow
        a //= 10
        b //= 10
    return borrows


def compute_features(dag: DAG, n_distractors: int = 0, has_merge: bool = False) -> dict[str, float]:
    n_mul = n_div = n_carry = 0
    for node in dag.op_nodes():
        a, b = (dag[i].value for i in node.inputs)
        if node.op is Op.MUL:
            n_mul += 1
        elif node.op is Op.DIV:
            n_div += 1
        elif a.denominator == 1 and b.denominator == 1 and a >= 0 and b >= 0:
            if node.op is Op.ADD:
                n_carry += count_carries(int(a), int(b))
            elif a >= b:
                n_carry += count_borrows(int(a), int(b))
    max_abs = max(abs(n.value) for n in dag.nodes)
    return {
        "steps": float(dag.n_steps),
        "log10_max": math.log10(float(max_abs)) if max_abs > 0 else 0.0,
        "n_mul": float(n_mul),
        "n_div": float(n_div),
        "n_carry": float(n_carry),
        "n_distractors": float(n_distractors),
        "has_merge": 1.0 if has_merge else 0.0,
    }


def feature_vector(features: dict[str, float], names: tuple[str, ...] = FEATURES) -> list[float]:
    return [float(features[n]) for n in names]
