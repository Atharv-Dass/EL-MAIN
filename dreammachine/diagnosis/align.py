"""Reference traces and alignment of a model's reasoning against them.

A ReferenceTrace is the gold computation in a source-independent form. It is
built from our DAG for synthetic items, or from calculator annotations
(`<<48/2=24>>`) for real GSM8K items, so the same classifier works on both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction

from ..generator.dag import DAG
from .cot_parse import Equation, safe_eval

_ANNOT = re.compile(r"<<([^=<>]+)=([^<>]+)>>")
_Q_NUM = re.compile(r"(?<![\w.])\$?(\d[\d,]*(?:\.\d+)?)")


@dataclass
class RefStep:
    expr: str
    value: Fraction
    ops: list[str]


@dataclass
class ReferenceTrace:
    answer: Fraction
    steps: list[RefStep]
    relevant_numbers: set[Fraction] = field(default_factory=set)    # numbers the solution uses
    distractor_numbers: set[Fraction] = field(default_factory=set)  # stated but unused

    @property
    def step_values(self) -> list[Fraction]:
        return [s.value for s in self.steps]

    @classmethod
    def from_dag(cls, dag: DAG, distractor_numbers: list[int] | None = None) -> "ReferenceTrace":
        steps = []
        for node in dag.op_nodes():
            a, b = (dag[i].value for i in node.inputs)
            steps.append(RefStep(expr=f"{a}{node.op.value}{b}", value=node.value, ops=[node.op.value]))
        return cls(
            answer=dag.answer,
            steps=steps,
            relevant_numbers={n.value for n in dag.leaves()},
            distractor_numbers={Fraction(m) for m in (distractor_numbers or [])},
        )

    @classmethod
    def from_item(cls, item) -> "ReferenceTrace":
        return cls.from_dag(DAG.from_dict(item.dag), item.distractor_numbers)

    @classmethod
    def from_gsm8k(cls, question: str, solution: str) -> "ReferenceTrace":
        """Build a trace from a GSM8K reference solution with calculator annotations.

        Limitation: numbers written as words ('twice', 'half', 'three') are invisible
        here, so distractor detection on real GSM8K is conservative.
        """
        steps: list[RefStep] = []
        used: set[Fraction] = set()
        for expr, val in _ANNOT.findall(solution):
            try:
                value = Fraction(val.replace(",", "").strip())
            except ValueError:
                try:
                    value = safe_eval(val)
                except Exception:
                    continue
            steps.append(RefStep(expr=expr, value=value, ops=re.findall(r"[-+*/]", expr)))
            used.update(Fraction(x) for x in re.findall(r"\d+(?:\.\d+)?", expr))
        ans_text = solution.split("####")[-1].strip().replace(",", "")
        answer = Fraction(ans_text)
        q_nums = {Fraction(x.replace(",", "")) for x in _Q_NUM.findall(question)}
        relevant = q_nums & used
        return cls(answer=answer, steps=steps, relevant_numbers=relevant,
                   distractor_numbers=q_nums - used)


@dataclass
class Alignment:
    reached: list[bool]           # per reference step: did the model produce this value?
    first_divergence: int | None  # index of the first reference step never reached
    progress: float               # fraction of reference steps reached

    @property
    def complete(self) -> bool:
        return self.first_divergence is None


def align(trace: ReferenceTrace, equations: list[Equation]) -> Alignment:
    """Match reference step values against the values the model *claims* (right-hand sides).

    Claimed values are used, not recomputed ones, because we want to know which plan
    the model followed, independently of whether its arithmetic was right.
    """
    produced = {e.rhs_value for e in equations}
    reached = [s.value in produced for s in trace.steps]
    first = next((i for i, r in enumerate(reached) if not r), None)
    progress = sum(reached) / len(reached) if reached else 0.0
    return Alignment(reached=reached, first_divergence=first, progress=progress)
