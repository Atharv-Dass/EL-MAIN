"""Import an existing results JSONL (e.g. the friend's Colab GSM8K run) and re-score it (PLAN.md §8.5).

Each answer is re-read with three extractors and compared with the gold answer:
    extract_answer        the project's extractor (#### > \\boxed > "answer is" > last number); used by classify
    extract_lastline      last number on the last non-empty line (tolerant), plus format_ok
    extract_lastline_v0   first number on the last line (the friend's original behaviour)
Disagreements between them are reported, and every answer is classified against the benchmark's reference
solution (GSM8K: the <<a*b=c>> calculator annotations). The rows are stored as one eval run.

The file's field names are detected from common names; pass --id-field / --text-field if detection fails.
"""

from __future__ import annotations

import collections
import json
from fractions import Fraction
from pathlib import Path

from ..diagnosis.extract import answers_match, extract_answer, extract_lastline, extract_lastline_v0
from ..diagnosis.taxonomy import ErrorType
from ..experiments.evaluate import ResponseRecord, lastline_fields, summarize
from . import registry
from .runner import read_rows

ID_FIELDS = ("example_id", "id", "idx", "index", "i")
TEXT_FIELDS = ("text", "response", "output", "completion", "generation", "model_output")
EXTRACTORS = {"extract_answer": extract_answer, "extract_lastline": extract_lastline,
              "extract_lastline_v0": extract_lastline_v0}


def _detect(rows: list[dict], candidates: tuple[str, ...], given: str | None, what: str) -> str:
    keys = set().union(*(r.keys() for r in rows[:50]))
    if given:
        if given not in keys:
            raise ValueError(f"--{what}-field {given!r} is not in the rows; row keys: {sorted(keys)}")
        return given
    found = [c for c in candidates if c in keys]
    if len(found) != 1:
        raise ValueError(f"cannot tell which field holds the {what} (candidates found: {found}); "
                         f"row keys: {sorted(keys)}. Pass --{what}-field.")
    return found[0]


def _single(rows: list[dict], key: str, override):
    if override is not None:
        return override
    values = {json.dumps(r.get(key), sort_keys=True) for r in rows if r.get(key) is not None}
    return json.loads(values.pop()) if len(values) == 1 else (None if not values else "mixed")


def import_results(path: str | Path, benchmark: str, model: str | None = None, prompt_version: str | None = None,
                   limit: int | None = None, id_field: str | None = None, text_field: str | None = None,
                   load_options: dict | None = None) -> dict:
    """Re-score a results file. Returns {"records", "report", "meta"} (nothing is stored here)."""
    rows = read_rows(Path(path))
    if not rows:
        raise ValueError(f"{path}: no rows")
    idf = _detect(rows, ID_FIELDS, id_field, "id")
    txf = _detect(rows, TEXT_FIELDS, text_field, "text")
    bench = registry.get(benchmark)
    examples = bench.load(limit=limit, **(load_options or {}))
    by_id = {e.id: e for e in examples}

    def example_for(raw):
        if isinstance(raw, int) or (isinstance(raw, str) and raw.isdigit()):   # a row index into the benchmark
            i = int(raw)
            return examples[i] if 0 <= i < len(examples) else None
        return by_id.get(str(raw))

    latest: dict[str, tuple[dict, object]] = {}
    unmatched, duplicates = [], 0
    for r in rows:
        ex = example_for(r.get(idf))
        if ex is None:
            unmatched.append(r.get(idf))
            continue
        duplicates += ex.id in latest
        latest[ex.id] = (r, ex)          # resumable files may repeat an index; keep the last row

    records, correct = [], collections.Counter()
    no_extract = collections.Counter()
    pairs = [("extract_answer", "extract_lastline"), ("extract_answer", "extract_lastline_v0"),
             ("extract_lastline", "extract_lastline_v0")]
    disagree = {f"{a} vs {b}": [] for a, b in pairs}
    for ex_id, (r, ex) in sorted(latest.items()):
        text = str(r.get(txf) or "")
        preds: dict[str, Fraction | None] = {name: fn(text) for name, fn in EXTRACTORS.items()}
        for name, p in preds.items():
            correct[name] += answers_match(p, ex.answer)
            no_extract[name] += p is None
        for a, b in pairs:
            if preds[a] != preds[b]:
                disagree[f"{a} vs {b}"].append(ex_id)
        d = bench.score(ex, text)
        diag = {**d.to_dict(), **lastline_fields(text)}
        records.append(ResponseRecord(ex.id, bench.name, 0, text, d.error_type is ErrorType.CORRECT,
                                      d.error_type.value, diag, dict(ex.features)))
    n = len(records)
    summ = summarize(records)
    report = {
        "file": str(path), "benchmark": bench.name, "id_field": idf, "text_field": txf,
        "n_rows": len(rows), "n_scored": n, "n_duplicates": duplicates, "n_unmatched": len(unmatched),
        "unmatched_examples": unmatched[:10],
        "accuracy": {name: (correct[name] / n if n else None) for name in EXTRACTORS},
        "no_extract": dict(no_extract),
        "disagreements": {k: {"n": len(v), "examples": v[:10]} for k, v in disagree.items()},
        "error_distribution": summ.get("error_distribution", {}), "format_ok_share": summ.get("format_ok_share"),
    }
    meta = {"model": _single(rows, "model_id", model), "prompt_version": _single(rows, "prompt_version", prompt_version),
            "gen_settings": _single(rows, "gen_settings", None)}
    return {"records": records, "report": report, "meta": meta}


def cmd_import(args) -> dict:
    from ..experiments.pipeline import Config, benchmark_options
    from ..store.db import Store

    opts = {}
    if args.config:
        opts = {k: v for k, v in benchmark_options(Config.load(args.config), args.benchmark).items() if k != "limit"}
    res = import_results(args.file, args.benchmark, args.model, args.prompt_version, args.limit,
                         getattr(args, "id_field", None), getattr(args, "text_field", None), opts)
    meta, rep = res["meta"], res["report"]
    store = Store(args.db)
    run = store.create_run(f"import:{meta['model']}:{args.benchmark}:{meta['prompt_version']}", "eval", {
        "arm": "base", "model": meta["model"], "benchmark": args.benchmark, "prompt_version": meta["prompt_version"],
        "generation": meta["gen_settings"], "source": "benchmarks.import-jsonl", "results_file": str(args.file)})
    store.add_responses(run, res["records"])
    store.put_artifact(run, "import_report", rep)
    store.finish_run(run, metrics={args.benchmark: rep["accuracy"]["extract_answer"], **{
        f"accuracy_{k}": v for k, v in rep["accuracy"].items()}})
    return {"run_id": run, **meta, **rep}
