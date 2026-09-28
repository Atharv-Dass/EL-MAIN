"""Constrained sampling of word problems that are correct by construction.

A `GenSpec` sets the difficulty knobs (steps, digits, op mix, distractors,
merge). `generate` builds a DAG that satisfies them and realises it as text,
with a GSM8K-style solution. Every intermediate value is a positive integer
and every division is exact. The answer is the DAG's exact value, never a
string computed from the text.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass, field
from fractions import Fraction

from .dag import DAG, Op
from .families import FAMILIES, NAMES, Family, families_for_split
from .features import compute_features

OPS: tuple[Op, ...] = (Op.ADD, Op.SUB, Op.MUL, Op.DIV)

# Solution phrasing per op. Family-agnostic, so the solution format stays uniform.
_SOLUTION = {
    Op.ADD: "{who} now has {expr}.",
    Op.SUB: "{who} then has {expr} left.",
    Op.MUL: "{who} then has {expr}.",
    Op.DIV: "One equal share is {expr}.",
}
_SOLUTION_COMBINE = {
    Op.ADD: "Together they have {expr}.",
    Op.SUB: "The difference is {expr}.",
}


class GenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class GenSpec:
    steps: int = 3
    digits: int = 2
    op_weights: tuple[float, float, float, float] = (0.35, 0.35, 0.15, 0.15)  # + - * /
    n_distractors: int = 0
    merge: bool = False
    split: str = "train"
    family: str | None = None
    max_tries: int = 500

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if self.merge and self.steps < 2:
            raise ValueError("merge requires steps >= 2")
        if not 1 <= self.digits <= 6:
            raise ValueError("digits must be in [1, 6]")
        if len(self.op_weights) != 4 or min(self.op_weights) < 0 or sum(self.op_weights) <= 0:
            raise ValueError("op_weights must be 4 non-negative numbers with a positive sum")
        if self.n_distractors < 0:
            raise ValueError("n_distractors must be >= 0")

    @property
    def leaf_range(self) -> tuple[int, int]:
        return max(2, 10 ** (self.digits - 1)), 10 ** self.digits - 1

    @property
    def mul_range(self) -> tuple[int, int]:
        return (2, 9) if self.digits == 1 else (2, 12)

    @property
    def div_range(self) -> tuple[int, int]:
        return 2, 12

    @property
    def value_cap(self) -> int:
        return 10 ** (self.digits + 2)


@dataclass
class Item:
    id: str
    family: str
    split: str
    question: str
    answer: int
    solution: str
    dag: dict
    features: dict[str, float]
    distractor_numbers: list[int]
    spec: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Item":
        return cls(**d)

    def to_training_example(self) -> dict[str, str]:
        return {"prompt": self.question, "completion": self.solution}


# ---------------------------------------------------------------- internals
@dataclass
class _Step:
    node: int
    op: Op
    who: str
    template: str | None  # None for the combine step (asked in the question)


def _plan_ops(spec: GenSpec, n_ops: int, rng: random.Random) -> list[Op]:
    """Sample the op sequence up front. Ops are never swapped later: if the plan
    cannot be realised the whole item is rejected, so the op mix stays exactly
    as specified (this matters for factorial probing)."""
    return rng.choices(OPS, weights=spec.op_weights, k=n_ops)


def _operand_for(op: Op, v: int, spec: GenSpec, rng: random.Random) -> int | None:
    """An operand n such that `v op n` is a positive integer within the cap, or None."""
    lo, hi = spec.leaf_range
    if op is Op.ADD:
        n = rng.randint(lo, hi)
        return n if v + n <= spec.value_cap else None
    if op is Op.SUB:
        top = min(hi, v - 1)
        return rng.randint(lo, top) if top >= lo else None
    if op is Op.MUL:
        mlo, mhi = spec.mul_range
        ok = [n for n in range(mlo, mhi + 1) if v * n <= spec.value_cap]
        return rng.choice(ok) if ok else None
    dlo, dhi = spec.div_range
    ok = [n for n in range(dlo, dhi + 1) if v % n == 0 and v // n >= 1]
    return rng.choice(ok) if ok else None


def _build_chain(dag: DAG, n_ops: int, who: str, fam: Family, spec: GenSpec,
                 rng: random.Random, steps: list[_Step]) -> int:
    lo, hi = spec.leaf_range
    cur = dag.leaf(rng.randint(lo, hi), role=f"start:{who}")
    steps.append(_Step(node=cur, op=Op.ADD, who=who, template=rng.choice(fam.start)))
    for op in _plan_ops(spec, n_ops, rng):
        n = _operand_for(op, int(dag[cur].value), spec, rng)
        if n is None:
            raise GenerationError(f"planned {op.name} is infeasible here")
        leaf = dag.leaf(n, role="operand")
        cur = dag.op(op, cur, leaf)
        steps.append(_Step(node=cur, op=op, who=who, template=rng.choice(fam.ops[op])))
    return cur


def _fmt(n: int | Fraction) -> str:
    return str(int(n))


def _expr(dag: DAG, node_id: int) -> str:
    node = dag[node_id]
    a, b = (_fmt(dag[i].value) for i in node.inputs)
    c = _fmt(node.value)
    sym = node.op.value  # type: ignore[union-attr]
    return f"{a} {sym} {b} = <<{a}{sym}{b}={c}>>{c}"


def _try_generate(spec: GenSpec, rng: random.Random) -> Item:
    fam = FAMILIES[spec.family] if spec.family else rng.choice(families_for_split(spec.split))
    if fam.split != spec.split:
        raise ValueError(f"family {fam.name!r} belongs to split {fam.split!r}, not {spec.split!r}")
    who, who2, other = rng.sample(NAMES, 3)

    dag = DAG()
    steps: list[_Step] = []
    if not spec.merge:
        _build_chain(dag, spec.steps, who, fam, spec, rng, steps)
        question = rng.choice(fam.question).format(who=who)
    else:
        rest = spec.steps - 1
        n_a = (rest + 1) // 2
        a = _build_chain(dag, n_a, who, fam, spec, rng, steps)
        b = _build_chain(dag, rest - n_a, who2, fam, spec, rng, steps)
        va, vb = dag[a].value, dag[b].value
        use_sub = spec.op_weights[1] > 0 and va != vb and (
            spec.op_weights[0] == 0 or rng.random() < 0.5)
        if use_sub:
            (hi_node, hi_who), (lo_node, lo_who) = sorted(
                [(a, who), (b, who2)], key=lambda t: dag[t[0]].value, reverse=True)
            root = dag.op(Op.SUB, hi_node, lo_node, role="combine")
            question = rng.choice(fam.question_diff).format(who=hi_who, who2=lo_who)
            steps.append(_Step(node=root, op=Op.SUB, who=hi_who, template=None))
        else:
            if dag[a].value + dag[b].value > spec.value_cap:
                raise GenerationError("combined value exceeds cap")
            root = dag.op(Op.ADD, a, b, role="combine")
            question = rng.choice(fam.question_sum).format(who=who, who2=who2)
            steps.append(_Step(node=root, op=Op.ADD, who=who, template=None))

    answer = dag.answer
    if answer.denominator != 1 or answer <= 0:
        raise GenerationError("answer must be a positive integer")
    if any(n.value.denominator != 1 or n.value <= 0 for n in dag.nodes):
        raise GenerationError("all intermediate values must be positive integers")

    # Distractors: numbers that appear nowhere in the computation.
    used = {int(n.value) for n in dag.nodes}
    lo, hi = spec.leaf_range
    distractor_numbers: list[int] = []
    for _ in range(spec.n_distractors):
        for _attempt in range(100):
            m = rng.randint(lo, hi)
            if m not in used and m not in distractor_numbers:
                distractor_numbers.append(m)
                break
        else:
            raise GenerationError("could not sample a distinct distractor number")
    if spec.n_distractors and not fam.distractors:
        raise GenerationError(f"family {fam.name!r} has no distractor templates")

    # Surface text.
    sentences: list[str] = []
    for st in steps:
        if st.template is None:
            continue
        n = int(dag[st.node].value) if dag[st.node].is_leaf else int(dag[dag[st.node].inputs[1]].value)
        sentences.append(st.template.format(who=st.who, n=n))
    for m in distractor_numbers:
        pos = rng.randint(1, len(sentences))  # never before the opening sentence
        sentences.insert(pos, rng.choice(fam.distractors).format(other=other, m=m))
    text = " ".join(sentences + [question])

    solution_lines = []
    for st in steps:
        node = dag[st.node]
        if node.is_leaf:
            continue
        tmpl = _SOLUTION_COMBINE[st.op] if st.template is None else _SOLUTION[st.op]
        solution_lines.append(tmpl.format(who=st.who, expr=_expr(dag, st.node)))
    solution_lines.append(f"#### {int(answer)}")
    solution = "\n".join(solution_lines)

    features = compute_features(dag, n_distractors=len(distractor_numbers), has_merge=spec.merge)
    item_id = hashlib.sha1(f"{text}|{int(answer)}".encode()).hexdigest()[:12]
    spec_dict = asdict(spec)
    spec_dict["family"] = fam.name
    return Item(
        id=item_id,
        family=fam.name,
        split=fam.split,
        question=text,
        answer=int(answer),
        solution=solution,
        dag=dag.to_dict(),
        features=features,
        distractor_numbers=distractor_numbers,
        spec=spec_dict,
    )


# ------------------------------------------------------------------ public
def generate(spec: GenSpec, rng: random.Random) -> Item:
    last: Exception | None = None
    for _ in range(spec.max_tries):
        try:
            return _try_generate(spec, rng)
        except GenerationError as e:
            last = e
    raise GenerationError(f"failed to generate after {spec.max_tries} tries: {last}")


def generate_many(spec: GenSpec, n: int, seed: int = 0, unique: bool = True) -> list[Item]:
    rng = random.Random(seed)
    out: list[Item] = []
    seen: set[str] = set()
    budget = n * 20
    while len(out) < n and budget > 0:
        budget -= 1
        item = generate(spec, rng)
        if unique and item.id in seen:
            continue
        seen.add(item.id)
        out.append(item)
    if len(out) < n:
        raise GenerationError(f"only {len(out)}/{n} unique items for spec {spec}")
    return out
