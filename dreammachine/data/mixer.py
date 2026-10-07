"""Training-set construction at an equal token budget.

Arms must be compared at the same number of training *tokens*, not examples.
Targeted items tend to be longer (more steps), so equal example counts would
quietly give the targeted arm more training signal.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable, Sequence

from .loaders import Example

_ANNOT = re.compile(r"<<[^<>]*>>")

TokenCounter = Callable[[str], int]


def whitespace_tokens(text: str) -> int:
    return len(text.split())


def clean_solution(solution: str) -> str:
    """Strip GSM8K calculator annotations: training targets should read like normal reasoning."""
    return _ANNOT.sub("", solution)


@dataclass
class TrainExample:
    prompt: str
    completion: str
    source: str
    id: str

    def to_dict(self) -> dict:
        return {"prompt": self.prompt, "completion": self.completion, "source": self.source, "id": self.id}


def to_train(ex: Example) -> TrainExample:
    return TrainExample(prompt=ex.question, completion=clean_solution(ex.solution), source=ex.source, id=ex.id)


def example_tokens(ex: TrainExample, count: TokenCounter = whitespace_tokens) -> int:
    return count(ex.prompt) + count(ex.completion)


def take_by_token_budget(examples: Sequence[TrainExample], budget: int,
                         count: TokenCounter = whitespace_tokens) -> list[TrainExample]:
    """Take examples in the given order until the next one would exceed the budget."""
    out, used = [], 0
    for ex in examples:
        t = example_tokens(ex, count)
        if used + t > budget:
            break
        out.append(ex)
        used += t
    return out


def total_tokens(examples: Sequence[TrainExample], count: TokenCounter = whitespace_tokens) -> int:
    return sum(example_tokens(e, count) for e in examples)


def mix(real: Sequence[TrainExample], synthetic: Sequence[TrainExample], ratio: float, budget: int,
        seed: int = 0, count: TokenCounter = whitespace_tokens) -> list[TrainExample]:
    """Combine at synthetic:real = ratio:1 by tokens, within `budget` tokens in total.

    ratio=0 means real only. Raises if either source is too small for its share,
    because silently short arms would break the equal-budget comparison.
    """
    if ratio < 0:
        raise ValueError("ratio must be >= 0")
    syn_budget = int(budget * ratio / (1 + ratio))
    real_budget = budget - syn_budget
    rng = random.Random(seed)
    real_s, syn_s = list(real), list(synthetic)
    rng.shuffle(real_s)
    rng.shuffle(syn_s)
    r = take_by_token_budget(real_s, real_budget, count)
    s = take_by_token_budget(syn_s, syn_budget, count) if syn_budget else []
    for name, got, want, pool in (("real", r, real_budget, real_s), ("synthetic", s, syn_budget, syn_s)):
        if want and total_tokens(got, count) < 0.95 * want and len(got) == len(pool):
            from ..errors import InsufficientData

            hint = ("raise data.pool_size or data.pool_max, or lower data.token_budget" if name == "synthetic"
                    else "lower data.token_budget or add real training data")
            raise InsufficientData(f"not enough {name} data: {total_tokens(got, count)} < {want} tokens; {hint}")
    out = r + s
    rng.shuffle(out)
    return out
