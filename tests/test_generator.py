import itertools
import random
import re
from fractions import Fraction

import pytest

from dreammachine.generator import (
    DAG, FAMILIES, GenerationError, GenSpec, Item, Op, count_borrows, count_carries, generate,
    generate_many,
)

ANNOT = re.compile(r"<<([^=<>]+)=([^<>]+)>>")


def _eval_simple(expr: str) -> Fraction:
    m = re.fullmatch(r"(\d+)([-+*/])(\d+)", expr)
    assert m, expr
    return Op(m.group(2)).apply(Fraction(m.group(1)), Fraction(m.group(3)))


# ----------------------------------------------------------------- DAG / math
def test_dag_exact_evaluation():
    d = DAG()
    a, b = d.leaf(7), d.leaf(3)
    c = d.op(Op.DIV, a, b)
    e = d.op(Op.MUL, c, d.leaf(3))
    assert d[c].value == Fraction(7, 3)
    assert d.answer == 7 and d.evaluate() == 7
    assert d.n_steps == 2 and d.depth() == 2
    assert DAG.from_dict(d.to_dict()).evaluate() == d.answer
    assert e == d.root


def test_dag_rejects_bad_input():
    d = DAG()
    with pytest.raises(IndexError):
        d.op(Op.ADD, 0, 1)
    z = d.leaf(0)
    with pytest.raises(ZeroDivisionError):
        d.op(Op.DIV, d.leaf(5), z)


@pytest.mark.parametrize("a,b,expected", [(0, 0, 0), (5, 4, 0), (5, 5, 1), (99, 1, 2), (999, 999, 3), (1234, 8766, 4)])
def test_count_carries(a, b, expected):
    assert count_carries(a, b) == expected


@pytest.mark.parametrize("a,b,expected", [(9, 3, 0), (10, 1, 1), (100, 1, 2), (1000, 999, 3), (52, 27, 1)])
def test_count_borrows(a, b, expected):
    assert count_borrows(a, b) == expected


def test_carry_counts_match_brute_force():
    rng = random.Random(0)
    for _ in range(2000):
        a, b = rng.randint(0, 99999), rng.randint(0, 99999)
        # brute force: a carry happens in column k iff (a mod 10^(k+1)) + (b mod 10^(k+1)) >= 10^(k+1)
        brute = sum(1 for k in range(7) if a % 10 ** (k + 1) + b % 10 ** (k + 1) >= 10 ** (k + 1))
        assert count_carries(a, b) == brute
        hi, lo = max(a, b), min(a, b)
        brute_b = sum(1 for k in range(7) if hi % 10 ** (k + 1) < lo % 10 ** (k + 1))
        assert count_borrows(hi, lo) == brute_b


# -------------------------------------------------------------- property test
SPECS = [
    GenSpec(steps=s, digits=d, n_distractors=k, merge=m, split=sp)
    for s, d, k, m, sp in itertools.product([1, 2, 4, 7], [1, 2, 3, 4], [0, 2], [False, True],
                                             ["train", "heldout"])
    if not (m and s < 2)
]


def _check_item(item: Item) -> None:
    dag = DAG.from_dict(item.dag)
    # 1. the stored answer is the exact DAG value
    assert dag.evaluate() == item.answer == dag.answer
    # 2. every value is a positive integer
    assert all(n.value.denominator == 1 and n.value > 0 for n in dag.nodes)
    # 3. solution annotations are arithmetically correct, match the DAG steps, and end with ####
    annots = ANNOT.findall(item.solution)
    assert len(annots) == dag.n_steps
    for (expr, val), node in zip(annots, dag.op_nodes()):
        assert _eval_simple(expr) == Fraction(val) == node.value
    assert item.solution.rstrip().endswith(f"#### {item.answer}")
    # 4. every leaf number is stated in the question; distractors are stated but unused
    q_numbers = [int(x) for x in re.findall(r"\d+", item.question)]
    for leaf in dag.leaves():
        assert int(leaf.value) in q_numbers
    used = {int(n.value) for n in dag.nodes}
    for m in item.distractor_numbers:
        assert m in q_numbers and m not in used
    # 5. features agree with the DAG
    assert item.features["steps"] == dag.n_steps
    assert item.features["n_distractors"] == len(item.distractor_numbers)
    # 6. the family's split is respected
    assert FAMILIES[item.family].split == item.split


def test_generator_property_10k():
    rng = random.Random(1234)
    count = 0
    while count < 10_000:
        for spec in SPECS:
            item = generate(spec, rng)
            _check_item(item)
            assert item.split == spec.split
            assert item.features["steps"] == spec.steps
            assert item.features["has_merge"] == float(spec.merge)
            assert item.features["n_distractors"] == spec.n_distractors
            count += 1


def test_leaf_magnitudes_respect_digits():
    for d in (1, 2, 3, 4):
        spec = GenSpec(steps=3, digits=d, op_weights=(1, 1, 0, 0))
        lo, hi = spec.leaf_range
        for item in generate_many(spec, 200, seed=d):
            dag = DAG.from_dict(item.dag)
            for leaf in dag.leaves():
                assert lo <= leaf.value <= hi


def test_op_weights_are_respected_exactly():
    # ops are planned, never substituted: a pure-op spec yields only that op
    for i, op in enumerate(Op):
        w = [0.0] * 4
        w[i] = 1.0
        items = generate_many(GenSpec(steps=3, digits=2, op_weights=tuple(w)), 50, seed=i)
        ops = {n.op for it in items for n in DAG.from_dict(it.dag).op_nodes()}
        assert ops == {op}


def test_mixed_op_frequencies_track_weights():
    items = generate_many(GenSpec(steps=4, digits=2, op_weights=(0.5, 0.5, 0, 0)), 400, seed=3)
    ops = [n.op for it in items for n in DAG.from_dict(it.dag).op_nodes()]
    assert 0.4 < ops.count(Op.SUB) / len(ops) < 0.6


def test_determinism_and_uniqueness():
    a = generate_many(GenSpec(steps=3), 100, seed=7)
    b = generate_many(GenSpec(steps=3), 100, seed=7)
    assert [x.to_dict() for x in a] == [y.to_dict() for y in b]
    assert len({x.id for x in a}) == 100
    assert Item.from_dict(a[0].to_dict()) == a[0]


def test_heldout_family_rejected_for_train_split():
    with pytest.raises(ValueError):
        generate(GenSpec(family="warehouse", split="train"), random.Random(0))


def test_spec_validation():
    with pytest.raises(ValueError):
        GenSpec(steps=0)
    with pytest.raises(ValueError):
        GenSpec(steps=1, merge=True)
    with pytest.raises(ValueError):
        GenSpec(op_weights=(0, 0, 0, 0))


def test_infeasible_spec_raises_generation_error():
    # pure division from single-digit starts dies quickly on primes; must fail loudly, not loop
    spec = GenSpec(steps=6, digits=1, op_weights=(0, 0, 0, 1), max_tries=20)
    with pytest.raises(GenerationError):
        generate_many(spec, 50, seed=0)
