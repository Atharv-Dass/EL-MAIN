"""Run a model over examples, diagnose every response, and summarise.

`evaluate` is model-agnostic: it takes any Runner, including the EchoRunner
test double. That keeps the whole pipeline testable without a GPU.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from ..data.loaders import Example
from ..diagnosis.extract import extract_lastline, extract_lastline_v0, lastline_format_ok
from ..diagnosis.taxonomy import ErrorType, classify
from ..generator.features import FEATURES
from ..models.runner import Runner


@dataclass
class ResponseRecord:
    example_id: str
    source: str
    sample: int
    text: str
    correct: bool
    error_type: str
    diagnosis: dict
    features: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(runner: Runner, examples: list[Example], chunk: int = 64,
             progress: Callable[[int, int], None] | None = None) -> list[ResponseRecord]:
    records: list[ResponseRecord] = []
    for start in range(0, len(examples), chunk):
        batch = examples[start:start + chunk]
        outputs = runner.generate([ex.question for ex in batch])
        # Runners that know it (HFRunner) report which answers hit max_new_tokens; a cut-off answer
        # has no final line, so its error label says more about the token limit than the model.
        cut = getattr(runner, "last_truncated", None)
        for i, (ex, samples) in enumerate(zip(batch, outputs)):
            trace = ex.trace()
            for k, text in enumerate(samples):
                d = classify(text, trace)
                diag = d.to_dict()
                diag.update(lastline_fields(text))
                if cut is not None:
                    diag["truncated"] = bool(cut[i][k])
                records.append(ResponseRecord(
                    example_id=ex.id, source=ex.source, sample=k, text=text,
                    correct=d.error_type is ErrorType.CORRECT, error_type=d.error_type.value,
                    diagnosis=diag, features=dict(ex.features)))
        if progress:
            progress(min(start + chunk, len(examples)), len(examples))
    return records


def _str(x) -> str | None:
    return None if x is None else str(x)


def lastline_fields(text: str) -> dict:
    """Final-line readings stored next to the classifier's answer (`predicted`, from extract_answer),
    for comparing prompt versions (PLAN.md §8.5). They do not change the error label."""
    return {"lastline": _str(extract_lastline(text)), "lastline_v0": _str(extract_lastline_v0(text)),
            "format_ok": lastline_format_ok(text)}


def summarize(records: list[ResponseRecord]) -> dict:
    if not records:
        return {"n_examples": 0, "n_responses": 0}
    by_source: dict[str, list[bool]] = defaultdict(list)
    errors: dict[str, int] = defaultdict(int)
    for r in records:
        by_source[r.source].append(r.correct)
        if not r.correct:
            errors[r.error_type] += 1
    n_err = sum(errors.values())
    out = {
        "n_examples": len({r.example_id for r in records}),
        "n_responses": len(records),
        "accuracy": float(np.mean([r.correct for r in records])),
        "accuracy_by_source": {s: float(np.mean(v)) for s, v in sorted(by_source.items())},
        "error_distribution": {k: v / n_err for k, v in sorted(errors.items())} if n_err else {},
    }
    ok = [r.diagnosis["format_ok"] for r in records if "format_ok" in r.diagnosis]
    if ok:
        out["format_ok_share"] = float(np.mean(ok))
    flags = [r.diagnosis["truncated"] for r in records if "truncated" in r.diagnosis]
    if flags:
        out["truncated_share"] = float(np.mean(flags))
    return out


def item_counts(records: list[ResponseRecord], names: tuple[str, ...] = FEATURES
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Aggregate responses into (Q, successes, trials, example_ids) for the LLTM.
    Only examples that carry Q-matrix features (synthetic items) are included."""
    agg: dict[str, list] = {}
    for r in records:
        if not r.features:
            continue
        a = agg.setdefault(r.example_id, [r.features, 0, 0])
        a[1] += int(r.correct)
        a[2] += 1
    ids = sorted(agg)
    Q = np.array([[agg[i][0][n] for n in names] for i in ids], dtype=float).reshape(len(ids), len(names))
    s = np.array([agg[i][1] for i in ids], dtype=float)
    n = np.array([agg[i][2] for i in ids], dtype=float)
    return Q, s, n, ids


def slice_accuracy(records: list[ResponseRecord], feature: str, threshold: float) -> dict:
    """Accuracy above vs at-or-below a feature threshold (e.g. the target-factor slice)."""
    hi = [r.correct for r in records if r.features and r.features[feature] > threshold]
    lo = [r.correct for r in records if r.features and r.features[feature] <= threshold]
    return {
        "feature": feature, "threshold": threshold,
        "above": {"n": len(hi), "accuracy": float(np.mean(hi)) if hi else None},
        "at_or_below": {"n": len(lo), "accuracy": float(np.mean(lo)) if lo else None},
    }
