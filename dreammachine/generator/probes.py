"""Factorial probe sets and generation pools.

- `factorial_probe_set`: fully crossed design over the difficulty knobs, used to
  fit the LLTM. Crossing makes the factor effects identifiable and causal.
- `natural_spec`: a GSM8K-like prior over knobs. This is the "untargeted" data
  distribution, and the reference for normalising factor ranges.
- `build_pool`: a large candidate pool to *select* targeted, matched and
  untargeted items from (generate-then-select).
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

from .sampler import GenerationError, GenSpec, Item, generate

OP_PROFILES: dict[str, tuple[float, float, float, float]] = {
    "addsub": (0.5, 0.5, 0.0, 0.0),
    "mixed": (0.35, 0.35, 0.15, 0.15),
    "muldiv": (0.2, 0.2, 0.3, 0.3),
}


@dataclass(frozen=True)
class ProbeGrid:
    steps: tuple[int, ...] = (2, 3, 4, 5, 6, 7)
    digits: tuple[int, ...] = (1, 2, 3, 4)
    distractors: tuple[int, ...] = (0, 1, 2)
    op_profiles: tuple[str, ...] = ("addsub", "mixed", "muldiv")
    merge: tuple[bool, ...] = (False,)

    def cells(self) -> list[dict]:
        return [
            dict(steps=s, digits=d, n_distractors=k, op_profile=o, merge=m)
            for s, d, k, o, m in itertools.product(self.steps, self.digits, self.distractors,
                                                   self.op_profiles, self.merge)
            if not (m and s < 2)
        ]


def spec_for_cell(cell: dict, split: str = "train") -> GenSpec:
    return GenSpec(steps=cell["steps"], digits=cell["digits"], n_distractors=cell["n_distractors"],
                   op_weights=OP_PROFILES[cell["op_profile"]], merge=cell["merge"], split=split)


def factorial_probe_set(grid: ProbeGrid = ProbeGrid(), per_cell: int = 3, seed: int = 0,
                        split: str = "train") -> list[Item]:
    rng = random.Random(seed)
    items: list[Item] = []
    seen: set[str] = set()
    for cell in grid.cells():
        spec = spec_for_cell(cell, split)
        got = 0
        for _ in range(per_cell * 20):
            if got == per_cell:
                break
            item = generate(spec, rng)
            if item.id in seen:
                continue
            seen.add(item.id)
            item.spec["cell"] = cell
            items.append(item)
            got += 1
        if got < per_cell:
            raise GenerationError(f"cell {cell} produced only {got}/{per_cell} unique items")
    return items


def natural_spec(rng: random.Random, split: str = "train") -> GenSpec:
    """A GSM8K-like prior: mostly 2-5 steps, 1-3 digit numbers, few distractors."""
    steps = rng.choices([1, 2, 3, 4, 5, 6, 7, 8], weights=[2, 14, 24, 22, 16, 11, 7, 4])[0]
    digits = rng.choices([1, 2, 3, 4], weights=[30, 45, 20, 5])[0]
    distractors = rng.choices([0, 1, 2], weights=[70, 22, 8])[0]
    profile = rng.choices(list(OP_PROFILES), weights=[30, 50, 20])[0]
    merge = steps >= 3 and rng.random() < 0.25
    return GenSpec(steps=steps, digits=digits, n_distractors=distractors,
                   op_weights=OP_PROFILES[profile], merge=merge, split=split)


def build_pool(n: int, seed: int = 0, split: str = "train", grid: ProbeGrid | None = None,
               exclude_ids: set[str] | None = None) -> list[Item]:
    """n unique items. With a grid: uniform over its cells (wide coverage, for selection).
    Without one: drawn from `natural_spec` (the untargeted distribution)."""
    rng = random.Random(seed)
    cells = grid.cells() if grid else None
    exclude = set(exclude_ids or ())
    out: list[Item] = []
    for _ in range(n * 20):
        if len(out) == n:
            break
        spec = spec_for_cell(rng.choice(cells), split) if cells else natural_spec(rng, split)
        try:
            item = generate(spec, rng)
        except GenerationError:
            continue
        if item.id in exclude:
            continue
        exclude.add(item.id)
        out.append(item)
    if len(out) < n:
        raise GenerationError(f"pool has only {len(out)}/{n} unique items")
    return out
