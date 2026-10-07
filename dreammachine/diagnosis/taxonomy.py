"""Annotation-free error taxonomy (docs/RESEARCH.md §3.6).

This replaces the Phase-1 taxonomy (REASONING_DEPTH / ARITHMETIC_MAGNITUDE /
...). Those were *item properties*, not *error causes*. Item properties now
live in the Q-matrix, and this module answers "what went wrong".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

from .align import ReferenceTrace, align
from .cot_parse import parse_equations
from .extract import answers_match, extract_answer


class ErrorType(str, Enum):
    CORRECT = "CORRECT"
    FORMAT_ERROR = "FORMAT_ERROR"          # no extractable answer, or the right value was not reported
    ARITHMETIC_SLIP = "ARITHMETIC_SLIP"    # a written equation is false
    DISTRACTOR_USE = "DISTRACTOR_USE"      # computed with an irrelevant number
    PLAN_ERROR = "PLAN_ERROR"              # consistent arithmetic, wrong computation graph
    UNVERIFIABLE = "UNVERIFIABLE"          # wrong answer with no parseable work


@dataclass
class Diagnosis:
    error_type: ErrorType
    predicted: str | None
    gold: str
    n_equations: int
    n_slips: int
    slip_index: int | None = None          # index of the first false equation
    slip_ops: list[str] = field(default_factory=list)
    slip_operand_digits: int | None = None
    divergence_step: int | None = None     # first reference step never reached
    progress: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["error_type"] = self.error_type.value
        return d


def classify(response: str, trace: ReferenceTrace) -> Diagnosis:
    pred = extract_answer(response)
    eqs = parse_equations(response)
    slips = [i for i, e in enumerate(eqs) if not e.correct]
    al = align(trace, eqs)
    base = dict(
        predicted=None if pred is None else str(pred),
        gold=str(trace.answer),
        n_equations=len(eqs),
        n_slips=len(slips),
        divergence_step=al.first_divergence,
        progress=al.progress,
    )

    if pred is None:
        return Diagnosis(ErrorType.FORMAT_ERROR, **base)
    if answers_match(pred, trace.answer):
        return Diagnosis(ErrorType.CORRECT, **base)
    if slips:
        e = eqs[slips[0]]
        digits = max((len(str(int(abs(x)))) for x in e.operands), default=None)
        return Diagnosis(ErrorType.ARITHMETIC_SLIP, slip_index=slips[0], slip_ops=e.ops,
                         slip_operand_digits=digits, **base)
    if eqs and answers_match(eqs[-1].rhs_value, trace.answer):
        return Diagnosis(ErrorType.FORMAT_ERROR, **base)
    if not eqs:
        return Diagnosis(ErrorType.UNVERIFIABLE, **base)

    legit = trace.relevant_numbers | set(trace.step_values)
    distractors = trace.distractor_numbers - legit
    if distractors and any(x in distractors and x not in legit for e in eqs for x in e.operands):
        return Diagnosis(ErrorType.DISTRACTOR_USE, **base)
    return Diagnosis(ErrorType.PLAN_ERROR, **base)


def error_distribution(diagnoses: list[Diagnosis], errors_only: bool = True) -> dict[str, float]:
    ds = [d for d in diagnoses if not errors_only or d.error_type is not ErrorType.CORRECT]
    if not ds:
        return {}
    out: dict[str, float] = {}
    for d in ds:
        out[d.error_type.value] = out.get(d.error_type.value, 0) + 1
    return {k: v / len(ds) for k, v in sorted(out.items())}
