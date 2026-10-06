"""Self-distillation of training solutions (PLAN.md D14; approved by the user for a dev-set test, 2026-10-06).

Why: fine-tuning on terse reference solutions (GSM8K's, and our code-written ones) made the model copy that style and
lowered GSM8K accuracy by 0.20-0.24 for every setting tried (docs/RUNNING.md §10).

What changes: only the *solution text*. Problems, gold answers and Q-matrix features stay made by code. For each
training item the base model writes `samples` answers (project prompt); the first one is kept if
    - its final answer equals the gold answer,
    - it writes at least one equation and every written equation is arithmetically correct,
    - it was not cut off by the token limit.
Otherwise the original solution is kept. Model answers are longer, so the arm is trimmed back to its token budget,
synthetic and real parts separately (the synthetic:real ratio by tokens stays as configured).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from ..diagnosis.cot_parse import parse_equations
from ..diagnosis.extract import answers_match, extract_answer, strip_reasoning_tags
from .loaders import final_answer
from .mixer import TrainExample, example_tokens, take_by_token_budget


@dataclass
class DistillConfig:
    samples: int = 2              # answers generated per item
    temperature: float = 0.7      # Qwen3 model card, non-thinking mode
    top_p: float = 0.8
    max_new_tokens: int = 512
    batch_size: int = 32          # sequences per batch (questions x samples)
    chunk: int = 64               # questions per generate() call; the stop check runs between chunks


@dataclass
class DistillStats:
    n_items: int = 0
    n_distilled: int = 0
    n_kept_original: int = 0
    by_source: dict = field(default_factory=dict)   # "synthetic" / "real" -> [distilled, total]

    def to_dict(self) -> dict:
        return {"n_items": self.n_items, "n_distilled": self.n_distilled, "n_kept_original": self.n_kept_original,
                "distilled_share": self.n_distilled / self.n_items if self.n_items else None,
                "by_source": self.by_source}


def accept(text: str, gold: str, truncated: bool = False) -> bool:
    """Keep a model-written solution only if it is right and its written work checks out."""
    if truncated:
        return False
    eqs = parse_equations(text)
    return bool(eqs) and all(e.correct for e in eqs) and answers_match(extract_answer(text), gold)


def _kind(row: TrainExample) -> str:
    return "synthetic" if row.source.startswith("synthetic") else "real"


def distill_arm(rows: list[TrainExample], runner, cfg: DistillConfig, budget: int, ratio: float, seed: int,
                progress: Callable[[int, int], None] | None = None) -> tuple[list[TrainExample], DistillStats]:
    """Distill an arm's rows in file order, then trim it back to `budget` tokens (see `rebalance`).

    Accepted model answers are longer than the reference solutions, so fewer items fit the budget: rows are
    distilled chunk by chunk and the loop stops once both the real and the synthetic share are full, because rows
    after that point would be cut by `rebalance` anyway. Gold answer = the row's `#### n` line."""
    syn_budget = int(budget * ratio / (1 + ratio))
    need = {"real": budget - syn_budget, "synthetic": syn_budget}
    used = {"real": 0, "synthetic": 0}
    left = {k: sum(1 for r in rows if _kind(r) == k) for k in need}
    stats = DistillStats()
    out: list[TrainExample] = []
    for start in range(0, len(rows), cfg.chunk):
        if all(used[k] >= need[k] or not left[k] for k in need):
            break
        batch = rows[start:start + cfg.chunk]
        answers = runner.generate([r.prompt for r in batch])
        cut = getattr(runner, "last_truncated", None)
        for i, (row, samples) in enumerate(zip(batch, answers)):
            gold = final_answer(row.completion)
            pick = next((s for k, s in enumerate(samples)
                         if accept(s, gold, bool(cut[i][k]) if cut else False)), None)
            kind = _kind(row)
            tally = stats.by_source.setdefault(kind, [0, 0])
            tally[1] += 1
            stats.n_items += 1
            if pick is None:
                stats.n_kept_original += 1
            else:
                stats.n_distilled += 1
                tally[0] += 1
                row = TrainExample(row.prompt, strip_reasoning_tags(pick).strip(), row.source, row.id)
            out.append(row)
            used[kind] += example_tokens(row)
            left[kind] -= 1
        if progress:
            progress(start + len(batch), len(rows))
    return rebalance(out, budget, ratio, seed), stats


def rebalance(rows: list[TrainExample], budget: int, ratio: float, seed: int) -> list[TrainExample]:
    """Trim an arm back to its token budget after its solutions changed length: real and synthetic parts are cut
    separately to their shares (ratio = synthetic:real by tokens), in file order, then shuffled as mix() does."""
    syn_budget = int(budget * ratio / (1 + ratio))
    real = take_by_token_budget([r for r in rows if _kind(r) == "real"], budget - syn_budget)
    syn = take_by_token_budget([r for r in rows if _kind(r) == "synthetic"], syn_budget) if syn_budget else []
    out = real + syn
    random.Random(seed).shuffle(out)
    return out
