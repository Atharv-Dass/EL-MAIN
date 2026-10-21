# Benchmarks

A **benchmark** is a named set of test problems. The registry in `dreammachine/benchmarks/` is the one place
that knows them; the pipeline (`eval.benchmarks` in a config), the standalone runner and (from B7) the API
all read it. Every benchmark is scored with the same error taxonomy (`diagnosis/taxonomy.py`).

| Name | Kind | Size | Source |
|---|---|---|---|
| `gsm8k` | external | 1319 | GSM8K test (Hugging Face `openai/gsm8k`) |
| `gsm_symbolic:main` / `:p1` / `:p2` | external | 5000 / 5000 / 2500 | `apple/GSM-Symbolic` (sizes counted on the live dataset, 2026-10-04) |
| `probes_train_families` | synthetic | depends on grid | generator probes, training story styles (seed 777) |
| `probes_heldout_families` | synthetic | depends on grid | generator probes, held-out story styles (seed 778) |
| `jsonl:<path>` | external | file | a local JSONL of `Example` rows (CLI / configs only) |

`limit`: GSM8K / GSM-Symbolic take the first N problems; probe sets take a seeded random subset of N (original
order kept). In a config, `eval.benchmarks` picks the list; without it the original list is used
(`gsm8k`, every `gsm_symbolic` variant in `eval.gsm_symbolic`, both probe sets), with identical examples.

## Run a benchmark (no training involved)
```powershell
python -m dreammachine.benchmarks list
python -m dreammachine.benchmarks run --model Qwen/Qwen3-0.6B --benchmark gsm8k --limit 200
python -m dreammachine.benchmarks run --model Qwen/Qwen3-1.7B --benchmark gsm8k --benchmark gsm_symbolic:main `
    --limit 200 --prompt-version v2_700
```
Defaults: greedy, `--max-new-tokens 512`, `--batch-size 32` (sequences per batch; measured, see
`docs/RUNNING.md` §6), bf16 (`--load-in-4bit` for NF4). `--config configs/main.yaml` reuses a config's local data
overrides and probe grid.

**Output:** `runs/benchmarks/<model>__<benchmark>__<prompt version>.jsonl`, append-only, one row per answer:
`benchmark, example_id, sample, source, text, gold, correct, error_type, diagnosis, features, model_id,
prompt_version, gen_settings, model_revision, batch_secs, batch_size, git_commit, git_dirty, packages, gpu`.
`diagnosis` holds the classifier's result plus `lastline`, `lastline_v0`, `format_ok` and (HF runner) `truncated`.
Each benchmark is also stored as one `eval` run in the SQLite store (`--db`, default `runs/dreammachine.db`).

**Resume:** run the same command again. Rows already in the file (key = benchmark + example id + sample) are
skipped; a half-written last line from a sudden stop is dropped and redone. A file never mixes models, prompt
versions or generation settings: the runner refuses and asks for another `--out` folder.

## Prompt versions (`dreammachine/models/prompts.py`)
| Version | Placement | Notes |
|---|---|---|
| `dm_v1` (default) | instruction + `Problem: …` in the user turn | asks for written equations and a final `#### n` |
| `v2_700` | **system** message; user turn = the question | the friend's prompt, byte-exact (pinned by sha256 in `tests/test_prompts.py`) |
| `qwen_boxed` | after the question in the user turn | the Qwen3 model card's math prompt; answer in `\boxed{}` |

If a chat template has no system role (it raises "System role not supported"), `v2_700` falls back to the user
turn exactly like `dm_v1` (`<prompt>\n\nProblem: <question>`). A project uses **one** prompt version everywhere
(`prompt_version:` at the top level of a config; PLAN.md D10).

## Answer extractors (`dreammachine/diagnosis/extract.py`)
| Function | Reads | Used for |
|---|---|---|
| `extract_answer` | `####` > `\boxed{}` > "answer is" > last number | the error classifier (correct / error type) |
| `extract_lastline` | last number on the last non-empty line; ignores `**`, `$`, thousands commas, a trailing `.` | final-line prompts such as v2_700 |
| `lastline_format_ok` | is the last non-empty line strictly a bare number (`18`, `-2.5`)? | format compliance (`18.` → false) |
| `extract_lastline_v0` | the **first** number on the last line (`3 * 6 = 18` → 3) | reproduces the friend's extractor, to measure disagreement only |

## Import existing results (e.g. the friend's Colab GSM8K run)
```powershell
python -m dreammachine.benchmarks import-jsonl friend_gsm8k.jsonl --benchmark gsm8k
```
Matches each row to a benchmark problem (a numeric index or an example id), re-scores the answer with all three
extractors, reports accuracy per extractor, "no extract" counts and disagreements, classifies every answer
against the GSM8K reference solution, and stores one `eval` run with an `import_report` artifact.
The id and answer fields are detected from common names (`idx`, `index`, `id`, … / `response`, `output`,
`text`, …); if detection fails, the error lists the row's keys and you pass `--id-field` / `--text-field`.
**The friend's file has not been received yet**: the importer is tested on a fixture only.

## Write a plug-in benchmark
Put a `.py` file in `dreammachine/benchmarks/contrib/`. It is imported on first registry use and must call
`register(...)` once. The loader receives `limit` and `seed` (plus any options you pass) and returns `Example`s.
`Example.trace()` must work for scoring: give GSM8K-style `<<a+b=c>>` annotations in `solution`, or use
`from_items` for generator items.

```python
# dreammachine/benchmarks/contrib/svamp_local.py  (worked example)
from dreammachine.benchmarks import Benchmark, register
from dreammachine.data import load_jsonl

def _load(limit=None, seed=0, path="data/svamp.jsonl"):
    return load_jsonl(path)[:limit]          # rows saved with dreammachine.data.save_jsonl

register(Benchmark(
    name="svamp_local",
    description="SVAMP word problems from a local JSONL file.",
    kind="external",            # "external" = real problems we did not write; "synthetic" = our generator
    loader=_load,
    size=None,                  # full size if known (shown by the API)
))
```
Rules: unique `name` (not starting with `jsonl:`); no network calls at import time; `kind` is `external` or
`synthetic`. Override `score(example, response)` only if the shared taxonomy cannot score your problems.
Check it with `python -m dreammachine.benchmarks list` and `pytest tests/test_benchmarks.py`.
