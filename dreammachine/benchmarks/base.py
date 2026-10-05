"""What a benchmark is (PLAN.md §8.1).

A benchmark is a named set of test problems. `external` = real problems we did not write (GSM8K,
GSM-Symbolic); `synthetic` = problems from our generator (probes). Each one loads `Example`s and scores a
model response with the shared error taxonomy, so every benchmark reports the same error types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from ..data.loaders import Example
from ..diagnosis.taxonomy import Diagnosis, classify

Kind = Literal["external", "synthetic"]
Loader = Callable[..., list[Example]]   # loader(limit=..., seed=..., **options) -> examples


@dataclass
class Benchmark:
    name: str
    description: str
    kind: Kind
    loader: Loader
    size: int | None = None              # full size if known (shown by the API); None = depends on config
    options: dict[str, Any] = field(default_factory=dict)   # default loader options

    def load(self, limit: int | None = None, seed: int = 0, **options: Any) -> list[Example]:
        """Up to `limit` examples (None = all). `options` override the defaults (e.g. a local file path)."""
        return self.loader(limit=limit, seed=seed, **{**self.options, **options})

    def score(self, example: Example, response: str) -> Diagnosis:
        return classify(response, example.trace())

    def info(self) -> dict:
        return {"name": self.name, "kind": self.kind, "description": self.description, "size": self.size}
