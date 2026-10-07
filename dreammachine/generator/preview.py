"""Print sample problems for manual review.

    python -m dreammachine.generator.preview --n 20 --steps 3 --digits 2 --distractors 1
"""

from __future__ import annotations

import argparse

from .sampler import GenSpec, generate_many


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--digits", type=int, default=2)
    p.add_argument("--distractors", type=int, default=0)
    p.add_argument("--merge", action="store_true")
    p.add_argument("--split", default="train", choices=["train", "heldout"])
    p.add_argument("--family", default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    spec = GenSpec(steps=a.steps, digits=a.digits, n_distractors=a.distractors, merge=a.merge,
                   split=a.split, family=a.family)
    for i, item in enumerate(generate_many(spec, a.n, seed=a.seed), 1):
        print(f"[{i}] ({item.family}) {item.question}")
        print("    " + item.solution.replace("\n", "\n    "))
        print()


if __name__ == "__main__":
    main()
