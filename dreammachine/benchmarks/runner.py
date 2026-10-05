"""Standalone benchmark runner (PLAN.md §8.4-8.5): no training code involved.

Results go to an append-only JSONL file, one per (model, benchmark, prompt version), so:
- an interrupted run resumes: rows already written (key = benchmark, example id, sample) are skipped;
- prompt versions never mix (the version is in the file name and checked on every row).
Every row carries its provenance: model, prompt version, generation settings, batch time, git commit, packages.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from ..data.loaders import Example
from ..experiments.evaluate import ResponseRecord, lastline_fields, summarize
from ..diagnosis.taxonomy import ErrorType
from ..models.runner import GenConfig, Runner
from ..provenance import hf_revision, provenance
from .base import Benchmark

# Settings that must match for rows of one file to be comparable (resume refuses to mix them).
_MUST_MATCH = ("model_id", "prompt_version", "gen_settings")


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def output_path(out_dir: str | Path, model: str, benchmark: str, prompt_version: str) -> Path:
    return Path(out_dir) / f"{_safe(model)}__{_safe(benchmark)}__{_safe(prompt_version)}.jsonl"


def read_rows(path: Path) -> list[dict]:
    """All complete rows; a line cut off by a sudden stop is skipped (that answer is generated again)."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _drop_partial_tail(path: Path) -> None:
    """Cut a half-written last line (no trailing newline) so the next append starts on a clean line."""
    if not path.exists() or path.stat().st_size == 0:
        return
    data = path.read_bytes()
    if not data.endswith(b"\n"):
        path.write_bytes(data[:data.rfind(b"\n") + 1])


def _gen_settings(gen: GenConfig) -> dict:
    return {k: v for k, v in asdict(gen).items() if k != "prompt_version"}


def run_benchmark(benchmark: Benchmark, examples: list[Example], runner: Runner, model: str, gen: GenConfig,
                  out_dir: str | Path, chunk: int = 32, log: Callable[[str], None] = print) -> dict:
    """Generate, score and append. Returns {"path", "records" (ResponseRecords for these examples), "summary"}."""
    path = output_path(out_dir, model, benchmark.name, gen.prompt_version)
    path.parent.mkdir(parents=True, exist_ok=True)
    expect = {"model_id": model, "prompt_version": gen.prompt_version, "gen_settings": _gen_settings(gen)}
    _drop_partial_tail(path)
    rows = read_rows(path)
    for r in rows:
        bad = [k for k in _MUST_MATCH if r.get(k) != expect[k]]
        if bad:
            raise ValueError(f"{path} already holds rows with different {bad}; use another --out folder "
                             "or delete the file")
    done = {(r["benchmark"], r["example_id"], r["sample"]) for r in rows}
    todo = [ex for ex in examples
            if any((benchmark.name, ex.id, k) not in done for k in range(gen.n_samples))]
    log(f"{benchmark.name}: {len(examples) - len(todo)}/{len(examples)} already done in {path.name}")
    prov = provenance()
    revision = hf_revision(model)
    n_done, t_all = 0, time.time()
    for start in range(0, len(todo), chunk):
        batch = todo[start:start + chunk]
        t0 = time.time()
        outputs = runner.generate([ex.question for ex in batch])
        secs = round(time.time() - t0, 3)
        cut = getattr(runner, "last_truncated", None)
        with path.open("a", encoding="utf-8") as f:
            for i, (ex, samples) in enumerate(zip(batch, outputs)):
                for k, text in enumerate(samples):
                    if (benchmark.name, ex.id, k) in done:
                        continue
                    d = benchmark.score(ex, text)
                    diag = {**d.to_dict(), **lastline_fields(text)}
                    if cut is not None:
                        diag["truncated"] = bool(cut[i][k])
                    f.write(json.dumps({
                        "benchmark": benchmark.name, "example_id": ex.id, "sample": k, "source": ex.source,
                        "text": text, "gold": ex.answer, "correct": d.error_type is ErrorType.CORRECT,
                        "error_type": d.error_type.value, "diagnosis": diag, "features": dict(ex.features),
                        **expect, "model_revision": revision, "batch_secs": secs, "batch_size": len(batch),
                        "git_commit": prov["git"]["commit"], "git_dirty": prov["git"]["dirty"],
                        "packages": prov["packages"], "gpu": prov["gpu"],
                    }, ensure_ascii=False) + "\n")
        n_done += len(batch)
        rate = n_done / max(time.time() - t_all, 1e-9)
        log(f"{benchmark.name}: {len(examples) - len(todo) + n_done}/{len(examples)} "
            f"({rate:.3f} problems/s, last batch {secs:.1f}s)")
    wanted = {ex.id for ex in examples}
    records = [ResponseRecord(r["example_id"], benchmark.name, r["sample"], r["text"], r["correct"],
                              r["error_type"], r["diagnosis"], r.get("features") or {})
               for r in read_rows(path) if r["example_id"] in wanted and r["benchmark"] == benchmark.name]
    return {"path": str(path), "records": records, "summary": summarize(records)}
