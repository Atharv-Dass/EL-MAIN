"""Evaluation/training examples from GSM8K, GSM-Symbolic, JSONL files and generated items.

The Hugging Face `datasets` import is lazy, so the rest of the package works
without the ML stack installed. If you have no internet access, export the
datasets to JSONL once and use `load_jsonl`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from ..diagnosis.align import ReferenceTrace
from ..generator.dag import DAG
from ..generator.sampler import Item


@dataclass
class Example:
    id: str
    source: str               # "gsm8k", "gsm_symbolic:p1", "synthetic:train", ...
    question: str
    answer: str               # gold final answer as a string
    solution: str = ""        # reference solution (GSM8K format), used for training and traces
    features: dict[str, float] = field(default_factory=dict)  # Q-matrix (synthetic only)
    meta: dict = field(default_factory=dict)

    def trace(self) -> ReferenceTrace:
        """Gold computation: from the DAG for synthetic items, else from GSM8K annotations."""
        if "dag" in self.meta:
            return ReferenceTrace.from_dag(DAG.from_dict(self.meta["dag"]),
                                           self.meta.get("distractor_numbers", []))
        return ReferenceTrace.from_gsm8k(self.question, self.solution)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Example":
        return cls(**d)


def final_answer(solution: str) -> str:
    return solution.split("####")[-1].strip().replace(",", "")


def from_items(items: Iterable[Item]) -> list[Example]:
    return [
        Example(
            id=it.id, source=f"synthetic:{it.split}", question=it.question, answer=str(it.answer),
            solution=it.solution, features=dict(it.features),
            meta={"dag": it.dag, "family": it.family, "split": it.split,
                  "distractor_numbers": it.distractor_numbers, "spec": it.spec},
        )
        for it in items
    ]


def _row_id(r: dict, i: int, id_prefix: str) -> str:
    """GSM8K rows have no id (use the row index). GSM-Symbolic's `id` is the *template* id, shared by
    all ~50 instances of a template (checked on the live dataset, 2026-10-04), so pair it with `instance`."""
    if "id" not in r:
        return f"{id_prefix}-{i}"
    if "instance" in r:
        return f"{id_prefix}-{r['id']}-{r['instance']}"
    return str(r["id"])


def _from_gsm_rows(rows: Iterable[dict], source: str, id_prefix: str) -> list[Example]:
    out = []
    for i, r in enumerate(rows):
        sol = r["answer"]
        out.append(Example(id=_row_id(r, i, id_prefix), source=source,
                           question=r["question"], answer=final_answer(sol), solution=sol,
                           meta={k: r[k] for k in ("original_id", "instance") if k in r}))
    return out


def load_gsm8k(split: str = "test", limit: int | None = None) -> list[Example]:
    from datasets import load_dataset  # lazy

    ds = load_dataset("openai/gsm8k", "main", split=split)
    rows = ds.select(range(min(limit, len(ds)))) if limit else ds
    return _from_gsm_rows(rows, "gsm8k", f"gsm8k-{split}")


def load_gsm_symbolic(variant: str = "main", limit: int | None = None) -> list[Example]:
    """variant: 'main', 'p1' (one extra clause) or 'p2' (two extra clauses)."""
    from datasets import load_dataset  # lazy

    ds = load_dataset("apple/GSM-Symbolic", name=variant, split="test")
    rows = ds.select(range(min(limit, len(ds)))) if limit else ds
    return _from_gsm_rows(rows, f"gsm_symbolic:{variant}", f"gsmsym-{variant}")


def save_jsonl(examples: Iterable[Example], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
    return p


def load_jsonl(path: str | Path) -> list[Example]:
    with Path(path).open(encoding="utf-8") as f:
        return [Example.from_dict(json.loads(line)) for line in f if line.strip()]
