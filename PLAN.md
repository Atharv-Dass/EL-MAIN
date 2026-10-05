# DreamMachine — Master Plan (Phase 2: backend)

> **Read this whole file before you write any code.**
> This is the single source of truth for *what* to build and *in what order*.
> The API that the frontend will use is defined separately in `docs/API_CONTRACT.md`.
> Version 1.1 — 2026-10-04 (review changes listed in section 15). Repo: `https://github.com/yorudamn12/EL-MAIN`, branch `claude/wonderful-hypatia-pqs6t9`.

---

## 0. Rules for the AI agent (read these first)

1. **Order of truth.** (1) The code on the branch shows what exists. (2) This plan shows what to build.
   (3) `docs/API_CONTRACT.md` is the law for anything the frontend sees. (4) `docs/RESEARCH.md` is the
   science. If the plan and the code disagree, **stop and ask** before you pick one.
2. **Do not invent.** No made-up function names, file paths, dataset field names, model names, prompt
   texts or numbers. If you are not sure, open the file. Section 12 lists what exists and what does not.
3. **Backend only.** Do not write any frontend (no React, Vite, Next.js, HTML pages). The frontend will be
   built later by another tool (Manus AI) using only `docs/API_CONTRACT.md` and the running API.
4. **Protected files.** Do not change `docs/RESEARCH.md`, or any component marked `CONFIRMED` in
   `docs/COMPONENTS.md`, without the user's explicit OK in chat.
5. **Status labels.** You may set a component to `IMPLEMENTED`. Only a team member sets `CONFIRMED`.
6. **CPU vs GPU honesty.** CPU tests prove that the plumbing works. They do not prove the method works
   on a real model. Always write "CPU-verified" or "GPU-verified". Never claim a GPU result you did not run.
7. **Explore runs are never paper evidence.** See section 6.
8. **Windows first.** Native Windows (no WSL) is the main target; Linux must keep working. Use `pathlib`,
   explicit `encoding="utf-8"`, `python -m ...` entry points, and PowerShell commands in docs.
9. **Dependencies.** Ask before adding any new package. Already allowed: everything in `pyproject.toml`,
   plus `pydantic` (it comes with FastAPI).
10. **Contract discipline.** Every change to an API route or response shape updates
    `docs/API_CONTRACT.md`, bumps its version, adds a changelog line, and regenerates `docs/openapi.json`
    **in the same commit**. The contract test (section 11) must pass.
11. **Stop at every milestone** (section 10). Run `pytest -q`, commit, print the milestone report, and wait
    for the user's go-ahead.

---

## 1. What DreamMachine is (plain English)

DreamMachine finds out **where a small language model (≤ 2B parameters) makes mistakes** on multi-step
arithmetic word problems, **creates training problems aimed at that weakness**, fine-tunes the model with
QLoRA on one 8 GB laptop GPU, and then **tests it again** to see what changed.

**The research question (the novelty):** *Is it the targeting, or just the difficulty?*

Earlier papers show that training data aimed at a model's weakness beats random data. But aimed data is
also *harder* than random data. So nobody can tell whether the *aiming* helped, or whether *any* hard
practice would have helped the same. DreamMachine adds a **difficulty-matched control**: training data
that is **just as hard** for the model, but hard for a **different reason**. If aimed data beats the
matched control, aiming really helps. If they tie, earlier gains were mostly a difficulty effect.
Both answers are publishable.

Why only we can build this control: our problems are **made by code**, so we know exactly what makes each
problem hard (number of steps, carries, distractors, …). Problems written by an LLM do not come with
exact labels like that.

**Setting (not the headline):** everything runs on one RTX 4060 laptop (8 GB), with no big model and no
paid API anywhere in the loop.

---

## 2. Where the project stands today (2026-10-04)

### 2.1 What is built (milestones M0–M5)

| Area | Path | What it does |
|---|---|---|
| Problem generator | `dreammachine/generator/` | Builds a small calculation graph (DAG), turns it into a word problem using story templates ("families"), and computes exact features. |
| Diagnosis | `dreammachine/diagnosis/` | Extracts the answer, checks every written equation, labels the error type, fits the LLTM, ranks weaknesses, and builds the aimed / matched / untargeted selections. |
| Data | `dreammachine/data/` | Loads GSM8K / GSM-Symbolic / JSONL; mixes synthetic and real data under an equal token budget. |
| Models | `dreammachine/models/` | One shared prompt format; batched Hugging Face inference (`HFRunner`); `EchoRunner` test double. |
| Training | `dreammachine/train/qlora.py` | QLoRA (or plain LoRA) fine-tuning, loss only on the answer part, resume support, `train_manifest.json`. |
| Pipeline stages + CLI | `dreammachine/experiments/` | `screen`, `diagnose`, `build_data`, `train_arm`, `evaluate_model`, `report`; CLI in `run.py`. |
| Results store | `dreammachine/store/db.py` | SQLite tables `runs`, `responses`, `artifacts`. |
| API (read-only) | `dreammachine/api/app.py` | `/health`, `/meta`, `/components`, `/generate`, `/diagnose`, `/lltm/fit`, `/runs...` (no `/api/v1` prefix yet). |

### 2.2 How it was tested — and what that does NOT prove

All tests run on CPU. No real model has ever run in this repo. The tests use two stand-ins:

1. **A tiny fake model** (`tests/conftest.py`, fixture `tiny_model_dir`): a ~100k-parameter Llama with
   **random weights** and a homemade tokenizer. It knows nothing. It only proves that prompts format,
   training saves an adapter, and generation returns text.
2. **A simulated model** (`SimulatedRunner` in `tests/test_pipeline.py`): not a neural network. It answers
   correctly with probability `sigmoid(3 − q·η)` using a **planted** weakness (carrying). The test checks that
   diagnosis finds "carrying". In this test, trained-model answers do not depend on training at all (the
   random seed comes from the adapter name), so the comparison numbers it produces are noise by design.

**So: there is zero evidence yet that the method works on a real model.** Milestone B1 (GPU smoke run)
is the first real test.

### 2.3 What changed since Phase 1

The Phase 1 idea was a **local LLM that writes training problems**. That was **dropped**. Training problems
now come from the **code generator** because (a) answers are correct by construction and (b) every problem
has exact feature labels, which the matched control and the LLTM need. **Do not reintroduce LLM-written
training data** without the user's explicit OK.

---

## 3. Decisions (locked unless the user changes them)

| # | Decision |
|---|---|
| D1 | **Novelty = targeting vs difficulty**, tested with the difficulty-matched control. "Small models on a laptop" is the setting, not the headline. |
| D2 | Training problems are **made by code** (section 2.3). |
| D3 | **One pipeline from benchmark to final comparison.** A single "pipeline run" goes: preflight → baseline benchmark → diagnose → choose target → build data → train → re-benchmark → compare. No manual steps in between. The old per-stage CLI commands stay for debugging only. |
| D4 | **Backend only for now.** Frontend later, built by Manus AI from `docs/API_CONTRACT.md`. Manus builds in the cloud and cannot reach `127.0.0.1` on the laptop, so it builds against the contract's **mock mode**; real data is only seen when the frontend runs on the same laptop as the backend (contract §2). |
| D5 | **Compute = one laptop** (RTX 4060, 8 GB). API, worker and GPU all on `127.0.0.1`. No Colab, no cloud. |
| D6 | **Two modes.** *Explore*: free choices, logged, never paper evidence. *Paper*: fixed protocol from `configs/main.yaml` (4 arms incl. matched control, 3 seeds, equal token budget), nothing can be changed except the model and the run name. |
| D7 | The matched control is **mandatory in paper mode** and **on by default in explore mode** (the user may turn it off). |
| D8 | **PROPOSED — needs team sign-off:** the headline result is *aimed − matched control* on **GSM8K test and GSM-Symbolic** (real problems our generator did not write). The target-weakness slice and the change in η are secondary results. Reason: the aimed arm trains on many more problems with the target feature, so it will almost surely win on that slice. **Compute all of them; do not edit `docs/RESEARCH.md` §3.7 until the team confirms.** |
| D9 | The friend's benchmark code is adopted through the benchmark module (section 8), with the answer-extraction bug fixed. |
| D10 | **One prompt for the whole project** (benchmark, diagnosis, training, evaluation). Two candidate versions are registered (`dm_v1` = ours, `v2_700` = friend's); **DECIDED 2026-10-05 by the user: `qwen_boxed`** (the Qwen3 model card's math prompt; best on the dev slice, docs/RUNNING.md §7). Training answers end with `\boxed{n}` (converted at tokenization; data files keep `#### n`). `dm_v1` and `v2_700` stay registered for comparisons. |
| D11 | Native Windows, no WSL. Public GitHub release later, so the README must hold a complete, safe install guide. |
| D13 | **PROPOSED (2026-10-05, user delegated the choice):** keep the two Qwen3 models as the core (clean size comparison within one family). Add `meta-llama/Llama-3.2-1B-Instruct` as a third, cross-family model **if** a dev-slice benchmark (dm_v1 + boxed, ~45 min GPU) shows it is usable — it answers "is it Qwen-specific?" (Qwen models train on a lot of math data). Its paper run (full or a reduced 12-training protocol without the ratio sweep) is decided after B8. `Qwen/Qwen2.5-1.5B-Instruct` stays out of the paper (same vendor, size overlaps Qwen3-1.7B); the importer covers the friend's results without re-running it. |
| D12 | **Two research models: `Qwen/Qwen3-0.6B` and `Qwen/Qwen3-1.7B`** (decided by the user, 2026-10-05), so results can be compared across model size. Each model gets its own paper run (paper mode already takes `model` per pipeline). `docs/RESEARCH.md` §4 still says "choose the model" (singular): the team must update it (protected file). |

---

## 4. Words you must use exactly (glossary)

Use these exact names in code, configs, the database and the API. Do not create synonyms.

| Term | Meaning in plain English | Where in code |
|---|---|---|
| **feature** | One measurable thing that can make a problem hard. There are exactly 7: `steps`, `log10_max`, `n_mul`, `n_div`, `n_carry`, `n_distractors`, `has_merge`. | `generator/features.py::FEATURES` |
| **Q-matrix** | Table of feature values, one row per problem. Exact, computed from the DAG. | `diagnosis/targeting.py::design_matrix` |
| **probes** | Test problems made by the generator on a full grid (steps × digits × distractors × op profile), so every feature varies in a controlled way. | `generator/probes.py::factorial_probe_set` |
| **families** | Story templates. 8 are `train`, 2 are `heldout`. Held-out families are **never** used for training. | `generator/families.py` |
| **LLTM** | A simple statistical model: chance of a correct answer = `sigmoid(θ − Σ q_k·η_k)`. **θ** = the model's overall ability. **η_k** = how much one unit of feature *k* lowers the odds of being right. Bigger η = bigger weakness. | `diagnosis/lltm.py` |
| **weakness ranking** | Each feature's η × its typical range (p90 − p10 on natural-looking problems) = **effect**. Ranked by (significant, effect). | `targeting.py::rank_weaknesses` |
| **target** | The feature we aim at. Auto choice = the significant weakness with the biggest effect (`choose_target`). | `targeting.py::choose_target` |
| **threshold** | Problems with the target feature **above** this value count as "rich in the target"; at or below = "baseline". | `targeting.py::baseline_threshold` |
| **arm** | One training-data recipe. Exactly 4 names: `real_only`, `untargeted`, `matched_control`, `targeted`. Plus `base` = the untrained model (evaluation only). | `experiments/pipeline.py::ARMS` |
| **targeted** ("aimed") | Synthetic problems rich in the target feature, with predicted chance of success between 0.3 and 0.7 (`p_band`). | `select_targeted` |
| **matched_control** | Synthetic problems **low** in the target feature, picked one-by-one to have the **same predicted difficulty** (nearest predicted logit) as each targeted problem. Needs the targeted selection first. | `select_matched_control` → `MatchReport` |
| **untargeted** | Synthetic problems sampled from the natural-looking pool, no aiming. | `select_untargeted` |
| **real_only** | Only real GSM8K training problems. Ratio is always 0. | `build_data` |
| **ratio** | synthetic : real by tokens. Ratio 3 = 3 parts synthetic to 1 part real (75% synthetic). | `data/mixer.py::mix` |
| **token budget** | Every arm gets the same total number of (whitespace) tokens, so arms are fair. | `data.token_budget` |
| **seed** | Repeat number. Paper mode uses seeds 0, 1, 2. | `cfg.seeds` |
| **arm key** | `"{arm}_r{ratio:g}_s{seed}"`, e.g. `targeted_r3_s0`, `real_only_r0_s0`. Used for file names, adapter folders, step keys. | `_arm_path`, `adapter_dir` |
| **benchmark** | A named test set. Exact names: `gsm8k`, `gsm_symbolic:main`, `gsm_symbolic:p1`, `gsm_symbolic:p2`, `probes_train_families`, `probes_heldout_families`, `jsonl:<path>`. These are already the metric keys written by `evaluate_model`; keep them. | `pipeline.py::eval_sets` |
| **external / synthetic benchmark** | External = real problems we did not write (GSM8K, GSM-Symbolic). Synthetic = problems from our generator (probes). | M6 `Benchmark.kind` |
| **target slice** | Accuracy on probe problems above vs at-or-below the threshold of the target feature. | `evaluate.py::slice_accuracy` |
| **error types** | Exactly 6: `CORRECT`, `FORMAT_ERROR`, `ARITHMETIC_SLIP`, `DISTRACTOR_USE`, `PLAN_ERROR`, `UNVERIFIABLE`. | `diagnosis/taxonomy.py::ErrorType` |
| **paired bootstrap** | Compare two arms on the **same** problems: resample problems 10,000 times, report mean difference, 95% CI and p-value. | `pipeline.py::paired_bootstrap` |
| **regression** | A trained arm got **worse** than `base` on a benchmark, and the whole 95% CI of the difference is below 0. | M8 |
| **adapter** | The small set of LoRA weights produced by training. Base model + adapter = fine-tuned model. | `adapters/<arm key>/adapter` |
| **pipeline run** | One job that runs every stage from baseline to comparison (D3). | M7 |
| **benchmark run** | A job that only benchmarks (+ optionally diagnoses) a model, no training. Its results can be reused by a later explore pipeline. | M7 |
| **mode** | `explore` or `paper` (D6). | M7/M8 |
| **step** | One unit of work inside a job (e.g. `train:targeted_r3_s0`). Runs in its own subprocess. | M7 |

---

## 5. The single pipeline (stage by stage)

```
 POST /api/v1/pipelines  (or: python -m dreammachine.jobs run ...)
            |
            v
 [S0 preflight] -> [S1 baseline] -> [S2 diagnose] -> [S3 choose_target] -> [S4 build_data]
                                                                                |
            +-------------------------------------------------------------------+
            v
   for each (arm, ratio, seed):   [S5 train:<arm key>] -> [S6 evaluate:<arm key>]
            |
            v
 [S7 compare]  -> results.json + report.md  -> GET /api/v1/pipelines/{id}/results
```

**Every pipeline run gets its own folder and its own name.** Output folder:
`runs/pipelines/<pipeline_id>/`. Internal config name (`Config.name`): `pl_<pipeline_id>`.
This matters: the existing stages pass data through files in `cfg.out` (`diagnosis.json`,
`data/manifest.json`, `evals/*.json`), and `report()` picks eval runs by the name prefix `f"{cfg.name}:"`.
A shared folder or name would mix up runs.

The orchestrator **reuses the existing stage functions**. Do not rewrite their logic.

### S0 — preflight
- **What:** check things that would otherwise fail hours later.
- **Checks:** config resolves and validates; model id is in `configs/models.yaml`; if a GPU is needed,
  `torch.cuda.is_available()`; free disk space under `runs/` and the Hugging Face cache; gated model access
  (try `huggingface_hub` model info; on 401/403 → error code `model_access_denied`); required datasets
  loadable (GSM8K, GSM-Symbolic) or local JSONL overrides present.
- **Disk rule:** fail with `disk_low` if free space on the drive holding `runs/` is below `preflight.min_free_gb`
  (default **10**, a starting guess — adjust after B1 measures real run sizes).
- **Writes:** `runs/pipelines/<id>/config.resolved.yaml`, `provenance.json` (section 9.4).
- **Fails with:** `gpu_unavailable`, `model_access_denied`, `dataset_unavailable`, `disk_low`, `validation_error`.

### S1 — baseline (benchmark the untrained model)
- **What:** run the base model on every selected benchmark and diagnose every answer.
- **Reuse:** `evaluate_model(cfg, store, factory, "base")`. After M6 it reads benchmarks from
  `cfg.eval["benchmarks"]` through the registry.
- **Note:** greedy decoding (temperature 0), one sample per problem.
- **Writes:** one `eval` run (arm `base`), `evals/base.json`, per-response rows in `responses`.
- **Skipped when** `reuse_from` points to a compatible earlier run (section 6.3); the step is then marked
  `skipped` and links the reused run ids.

### S2 — diagnose (find the weaknesses)
- **What:** run the base model on the factorial probe grid **several times per problem with sampling**
  (`n_samples: 3`, `temperature: 0.7`), fit the LLTM with bootstrap CIs, rank weaknesses.
- **Why sampling here but greedy in S1:** the LLTM needs a success *rate* per problem, so each problem is
  answered several times. Do not "fix" this difference.
- **Reuse:** `diagnose(cfg, store, factory)`. It also writes `annotate_errors.csv` for the human κ check.
- **Writes:** `diagnosis.json` (keys: `run_id, model, summary, lltm, weaknesses, target, threshold, probe_ids`)
  and the same keys as artifacts on the diagnose run.
- **Important:** `probe_ids` are excluded from all training pools later. Keep that.

### S3 — choose_target
- **What:** decide which feature the training data aims at.
- **Strategies (from the request):**
  - `auto` — use `diagnosis.json["target"]` (from `choose_target`). If it is `null`, fail the step with
    code `no_significant_weakness` and a message telling the user to start an explore run with a named
    feature or `untargeted` (they can reuse this baseline).
  - `feature` — a named feature from the 7. Allowed **only in explore mode**. Compute its threshold with
    `baseline_threshold`. If the feature is not significant in the ranking, continue but add a warning.
  - `untargeted` — no target. Only arms `untargeted` and `real_only` are allowed (the matched control needs
    a target). Explore mode only.
- **Paper mode:** must be `auto`.
- **Writes:** `target.json` = `{strategy, feature|null, threshold|null, eta, ci_low, ci_high, significant, warnings}`.
  The `build_data` step reads the target from here, **not** from `diagnosis.json["target"]` (refactor needed,
  see S4).

### S4 — build_data (make the training sets)
- **What:** generate big candidate pools with the code generator, select targeted / matched / untargeted
  problems, mix each with real GSM8K under the **same token budget**, write one JSONL per arm key.
- **Reuse:** `build_data(cfg, store)` for paper mode. For explore mode, split out
  `build_arm(cfg, arm, ratio, seed, target, threshold, lltm) -> dict` that builds one arm key. Paper mode keeps
  calling the same code path so its output stays identical.
- **Refactor rule:** after the split, a test must show that `build_data` on a fixed config produces
  **byte-identical** JSONL files to the version before the refactor (same seeds).
- **Gotchas:**
  - `matched_control` is built by pairing with the targeted selection. If the user asks only for
    `matched_control`, still compute the targeted selection internally (do not train on it).
  - Pools use `exclude_ids = probe_ids` so no diagnosis probe leaks into training.
  - Record `MatchReport` per seed (`n_requested, n_matched, mean_abs_logit_diff, max_abs_logit_diff,
    ks_statistic, ks_pvalue`). If `n_matched < n_requested` or the match looks poor, add a **warning**.
    Hard thresholds for failing are an open question for the team (section 14) — do not invent them.
  - If fewer targeted items than needed are found, the existing code adds a warning ("increase
    data.pool_size"). Surface warnings in the API.
- **Writes:** `data/<arm key>.jsonl`, `data/manifest.json`, one `build-data` run with the manifest artifact.

### S5 — train:<arm key>
- **What:** QLoRA fine-tune of the base model on one arm key's JSONL.
- **Reuse:** `train_arm(cfg, store, arm, ratio, seed)`. Output: `adapters/<arm key>/adapter` and
  `train_manifest.json`.
- **Add:** a `TrainerCallback` that writes `(step, total_steps)` progress and the training loss
  (every `logging_steps`) to the step row, so the API can show a progress bar and a loss curve.
- **Windows:** `dataloader_num_workers=0`; no `torch.compile`; no Triton. If `load_in_4bit` is set but
  bitsandbytes cannot import or has no CUDA, raise a clear error suggesting `load_in_4bit: false`
  for ≤ 1.7B models (bf16 LoRA fits in 8 GB).
- **Resume:** the trainer already resumes from `checkpoint-*` folders. A resumed pipeline must reuse them,
  but **ignore any `checkpoint-*` folder without `trainer_state.json`** (it was cut off mid-write) — delete it and
  resume from the newest complete one.
- **Cancel:** the same TrainerCallback reads `cancel_requested` from the job row every `logging_steps` and sets
  `control.should_training_stop = True`, so training stops cleanly at a step boundary (section 7.3).
- **Cleanup:** after the final adapter is saved, delete the `checkpoint-*` folders unless
  `train.keep_checkpoints: true` (default `false`). They are only needed to resume an unfinished run.

### S6 — evaluate:<arm key>
- **What:** run the fine-tuned model (base + adapter) on the **same benchmarks** as S1.
- **Reuse:** `evaluate_model(cfg, store, factory, arm, ratio, seed)`. It also refits the LLTM on
  `probes_train_families` and stores `eta`, and computes `…:target_slice` for probe benchmarks.
- **Rule:** inference precision (`eval.load_in_4bit`) must be identical for base and fine-tuned models.
- **Order:** the step list interleaves train → evaluate per arm key, so results appear early.
- **Cancel:** the `progress` callback also checks `cancel_requested` between chunks and raises a `Cancelled`
  exception, so evaluation stops cleanly between chunks (section 7.3).

### S7 — compare (the last stage)
- **What:** build the final comparison: before vs after for every arm, arm vs arm, regressions, η change,
  error-type shift, and the primary result block (D8).
- **Reuse:** `paired_bootstrap`; the pairing logic in `report()`.
- **Writes:** `results.json` (shape fixed by `docs/API_CONTRACT.md` → `PipelineResults`), `report.md`,
  and for paper mode also the existing `report.json`. Details in section 9.

### Screening is not part of the pipeline
`screen()` compares candidate models once to pick the research model. It stays CLI-only for now
(`python -m dreammachine.experiments.run screen --config configs/main.yaml`).

---

## 6. Modes, presets and reuse

### 6.1 Explore mode
| Field | Allowed | Default |
|---|---|---|
| `model` | any id in `configs/models.yaml` | — (required) |
| `benchmarks` | any registered names | `gsm8k`, `gsm_symbolic:main`, `probes_train_families`, `probes_heldout_families` |
| `benchmark_limit` | int ≥ 10, or null (= full) | from `configs/explore.yaml` |
| `target` | `auto`, `feature` (+ name), `untargeted` | `auto` |
| `arms` | subset of the 4 arms (rules in S3) | `["targeted", "matched_control"]` |
| `ratio` | one of 1, 3, 9 | 3 |
| `seeds` | 1–3 seeds from {0,1,2} | `[0]` |
| `reuse_from` | id of an earlier explore pipeline or benchmark run | null |

Create `configs/explore.yaml` as a copy of `configs/main.yaml` with smaller defaults. **Its numbers are
starting guesses; replace them with measured values after B1** and say so in a comment.

### 6.2 Paper mode
- Uses `configs/main.yaml` exactly. Only `model`, `name` may be set. Any other field → `422 paper_mode_locked`.
- `target` must be `auto`. No reuse.
- **Step count with the current `main.yaml`:** per seed = `real_only_r0` + `untargeted_r3` +
  `matched_control_r3` + `targeted_r1` + `targeted_r3` + `targeted_r9` = 6 trainings and 6 evaluations;
  × 3 seeds = **18 trainings + 18 evaluations**, plus preflight, baseline, diagnose, choose_target,
  build_data, compare. (This follows `experiments/run.py::_jobs`. Note: `docs/RUNNING.md` says "16 jobs";
  that is wrong — fix the doc.)
- How long it takes on the 4060 is **unknown** until B1 measures it.

### 6.3 Reuse (benchmark once, try several weaknesses)
The friend's flow is: test the model → see weaknesses → pick one → train. With a single pipeline, do it like
this: run a **benchmark run** (baseline + diagnosis) or an explore pipeline, then start a new explore pipeline
with `reuse_from: <that id>`. S1 and S2 are then marked `skipped` and their run ids are linked.
Reuse uses **two hashes**, both stored on every job:
- `baseline_hash` = model + HF model revision + benchmark list + benchmark_limit + prompt version + generation
  settings + eval precision (`eval.load_in_4bit`).
- `diagnosis_hash` = model + HF model revision + prompt version + the whole `diagnose` config block (probe grid,
  `probe_per_cell`, `n_samples`, `temperature`, `bootstrap`, seed) + eval precision.

Rules:
- S1 is skipped only if `baseline_hash` matches. S2 is skipped only if `diagnosis_hash` matches **and** the source
  job has a finished diagnosis.
- If the source is a benchmark run made **without** diagnosis (or with different diagnosis settings), the
  baseline is reused and **diagnosis runs fresh** (S2 is not skipped). No error.
- If `baseline_hash` does not match → `422 reuse_mismatch` (nothing worth reusing).
- Paper pipelines never reuse.

---

## 7. Job system (M7): queue, worker, cancel, resume

### 7.1 Database (add to `store/db.py`, with safe migrations)
- On connect: `PRAGMA journal_mode=WAL;` and `PRAGMA busy_timeout=5000;` (API, worker and step subprocesses
  share the file).
- **Migration rule:** `CREATE TABLE IF NOT EXISTS` for new tables; for new columns on `runs`, check
  `PRAGMA table_info` and `ALTER TABLE ... ADD COLUMN` only if missing. Never drop data.
- New table `jobs`: `id TEXT PK, kind TEXT ('pipeline'|'benchmark_run'), name TEXT, mode TEXT,
  status TEXT, request TEXT(JSON), resolved TEXT(JSON), baseline_hash TEXT, diagnosis_hash TEXT, config_hash TEXT,
  output_dir TEXT, reuse_from TEXT NULL, cancel_requested INTEGER DEFAULT 0, warnings TEXT(JSON),
  error TEXT(JSON) NULL, provenance TEXT(JSON), created_at REAL, started_at REAL NULL, finished_at REAL NULL`.
- New table `job_steps`: `job_id TEXT, idx INTEGER, key TEXT, stage TEXT, params TEXT(JSON), status TEXT,
  progress_done INTEGER, progress_total INTEGER, progress_unit TEXT, run_ids TEXT(JSON), metrics TEXT(JSON),
  started_at REAL NULL, finished_at REAL NULL, error TEXT(JSON) NULL, PRIMARY KEY (job_id, idx)`.
- New table `worker_state`: one row with `pid, heartbeat_at, current_job_id, current_step_idx`.
- New columns on `runs`: `job_id TEXT NULL`, `mode TEXT NULL`.
- Store times as epoch floats (as today); the API converts to ISO 8601 UTC strings.

### 7.2 Statuses (exact strings)
- Job: `queued`, `running`, `done`, `failed`, `cancelled`.
- Step: `pending`, `running`, `done`, `failed`, `skipped`, `cancelled`.

### 7.3 Worker
- Start: `python -m dreammachine.jobs.worker`. One worker, one job at a time (one GPU).
- Loop: pick the oldest `queued` job → mark `running` → for each step in order whose status is `pending`:
  spawn `python -m dreammachine.jobs.execute --job <id> --step <idx>` with `subprocess.Popen`, redirecting
  stdout/stderr to `runs/pipelines/<id>/logs/job.log` (append, utf-8).
- **Why a subprocess per step:** GPU memory is fully freed after each step, and a crash fails one step,
  not the worker.
- **Cancel (cooperative first, hard kill last):** the API sets `cancel_requested=1`. The running step checks this
  flag itself at safe points (training: every `logging_steps` in the TrainerCallback; evaluation and diagnosis:
  between chunks) and exits cleanly with status `cancelled`. The worker waits up to
  `worker.cancel_grace_s` (default 120 s) for that. Only if the step is still alive after the grace period does the
  worker call `Popen.kill()`.
  **Why:** on Windows, `Popen.terminate()` is the same as `kill()` (an instant `TerminateProcess`), so "terminate,
  then wait" gives no grace at all and can cut a checkpoint in half. No POSIX signals.
- **Heartbeat:** update `worker_state.heartbeat_at` every ~5 s. The API reports the worker as down if the
  heartbeat is older than 30 s.
- **Crash recovery:** on worker start, any step left `running` becomes `failed` with code `interrupted`,
  and its job becomes `failed`. The user can resume it.
- **Resume:** `failed` or `cancelled` jobs can be resumed: failed/cancelled/pending steps go back to `pending`,
  `done` and `skipped` steps are kept, job goes back to `queued`.

### 7.4 Step execution
- `dreammachine/jobs/steps.py`: turns a validated request into the ordered step list (section 5).
- `dreammachine/jobs/execute.py`: loads the job's resolved config, runs exactly one step by calling the
  existing stage function, writes progress/metrics/run ids to `job_steps`, exits 0 on success, non-zero on failure
  after writing a structured error `{code, message, details}`.
- Map known failures to error codes: CUDA out of memory → `out_of_memory` (message lists the knobs:
  `generation.batch_size`, `train.per_device_batch_size`, `train.max_seq_len`); missing GPU → `gpu_unavailable`;
  gated model → `model_access_denied`; dataset download error → `dataset_unavailable`; anything else →
  `internal_error` with the traceback in the log, not in the API message.
- Progress: S1/S6 use `evaluate(progress=...)` (already exists) → unit `examples`. S5 uses the new
  `TrainerCallback` → unit `steps`. S2 → `examples`. Others → 0/1.
- Timing: every step that loads a model records `load_s` (model + adapter loading) and `run_s` (the actual work)
  separately in `job_steps.metrics`. Each step is a new subprocess, so the model is loaded again every time; we
  want to see how much of the total that costs.
- ETA: `eta_s` for a job = for each unfinished step, the average duration of finished steps of the same stage in
  this job (or, if none, in the most recent finished job with the same model), summed. `null` if no data yet.
  It is a rough guide, not a promise.

### 7.5 Commands
- `python -m dreammachine.jobs run --preset explore --model Qwen/Qwen3-0.6B [...]` — run a pipeline in the
  foreground without the worker (for development).
- `python -m dreammachine.jobs enqueue ...` — add to the queue.
- `python -m dreammachine.jobs.worker` — the worker.
- `python -m dreammachine.serve` — convenience: starts the API (uvicorn on 127.0.0.1:8000) and the worker
  together; Ctrl+C stops both.

### 7.6 Windows notes for the job system
- Keep the repo and `runs/` **outside OneDrive-synced folders** (e.g. `C:\dev\EL-MAIN`). Sync tools can lock or
  corrupt SQLite WAL files.
- Use `sys.executable` to spawn subprocesses (the venv's Python).

### 7.7 GPU telemetry (laptop thermals)
One laptop does everything, so heat can slow the GPU down and distort timing numbers. While a step runs, the
worker samples `nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used
--format=csv,noheader,nounits` every 30 s into `runs/pipelines/<id>/logs/gpu.csv` (with timestamp and step key),
and writes a summary per step into `job_steps.metrics.gpu` = `{max_temp_c, min_sm_clock_mhz, max_memory_used_mb}`.
No new dependency. If `nvidia-smi` is missing, skip silently and set the summary to `null`.

### 7.8 Disk use and cleanup
- Checkpoints are deleted after training finishes (S5 cleanup). Final adapters, data files, logs and results are kept.
- The API reports `output_bytes` per job and free disk space in `GET /system`.
- `DELETE` on a job removes its output folder and its rows (`jobs`, `job_steps`, and `runs`/`responses`/`artifacts`
  linked by `job_id`). Allowed only for **explore pipelines and benchmark runs that are not queued or running**.
  **Paper jobs cannot be deleted through the API** (`409 paper_protected`); delete them by hand if you really mean it.
- A job that is the `reuse_from` source of another job can still be deleted; the other job keeps its own copies of
  the reused run ids in its steps, but drill-down links to the deleted runs will return 404.

---

## 8. Benchmark module (M6) and the friend's code

### 8.1 Interface — `dreammachine/benchmarks/base.py`
`Benchmark` protocol with: `name: str`, `description: str`, `kind: "external" | "synthetic"`,
`load(limit: int | None, seed: int) -> list[Example]` (reuse `Example` from `data/loaders.py`), and an
optional `score(example, response) -> Diagnosis` defaulting to
`diagnosis.taxonomy.classify(response, example.trace())`.

### 8.2 Registry — `registry.py`
`register(benchmark)`, `get(name)`, `available()`. `jsonl:<path>` is resolved on the fly.

### 8.3 Built-ins — wrap the existing loaders (do not move their logic yet)
`gsm8k` → `load_gsm8k("test")`; `gsm_symbolic:main|p1|p2` → `load_gsm_symbolic(v)`;
`probes_train_families` → `factorial_probe_set(grid, per_cell, seed=777)`;
`probes_heldout_families` → same with `split="heldout"`, `seed=778` (these seeds match today's `eval_sets`).
Then change `eval_sets(cfg)` to read `cfg.eval["benchmarks"]` through the registry, **defaulting to today's
exact list** when the key is missing, so old configs behave the same.

### 8.4 Standalone runner and plug-ins
- `python -m dreammachine.benchmarks run --model X --benchmark gsm8k --limit 200` → accuracy, error types,
  stored run. No dependency on training code.
- Friend's code goes in `dreammachine/benchmarks/contrib/<name>.py` and calls `register(...)`. Contract and one
  worked example in `docs/BENCHMARKS.md`.

### 8.5 What we adopt from the friend's Colab benchmark (Qwen2.5-1.5B-Instruct, full GSM8K test)
Known facts about it: 4-bit NF4 with fp16 compute; system prompt version `v2_700` ("final line = numeric answer
only"); greedy; `max_new_tokens=700`; batch 16; resumable append-only JSONL that skips done indices; per-row
`model_id`, `prompt_version`, `gen_settings`, `batch_secs`; reports accuracy and a no-extract count.
- **Adopt:** resumable append-only JSONL (resume key = benchmark + example id + sample); per-record provenance
  (model_id, prompt_version, gen_settings, batch_secs, git commit, package versions); prompt version in output
  file names; throughput logging.
- **Fix the extractor:** their `extract_pred` takes the **first** number on the last line, so `3 * 6 = 18`
  gives 3. Add `extract_lastline` (last number on the last non-empty line) and keep `extract_lastline_v0`
  (first number — reproduces theirs, for measuring disagreement only). No bare `except:`.
- **Prompts:** in `models/prompts.py` register `PROMPTS = {"dm_v1": <current INSTRUCTION>, "v2_700": ...}` and
  add `prompt_version` (default `"dm_v1"`) to `format_prompt`/`build_messages`, config, and provenance.
  **Update 2026-10-04:** the exact `v2_700` text was received (open question 4 resolved); it is in `CLAUDE.md`
  under "Prompt versions". Copy it byte-for-byte; never write your own version of it.
- **Choosing the prompt (on GPU, B1/B8):** run both on the same 200 GSM8K items; compare FORMAT_ERROR rate and
  UNVERIFIABLE rate. `dm_v1` is favoured because the error classifier needs written equations, unless it costs
  accuracy. The team decides.
- **Import existing results (CPU):** `python -m dreammachine.benchmarks import-jsonl <file> --benchmark gsm8k`
  stores the 1319 responses as a run, re-scores with `extract_answer`, `extract_lastline`,
  `extract_lastline_v0`, reports disagreements, and runs `classify` against GSM8K reference solutions. We have
  **not** received this file yet — build and test it with a fixture.
- Add `Qwen/Qwen2.5-1.5B-Instruct` to `configs/models.yaml` and to `configs/main.yaml` candidates.
- **Unverified:** GSM-Symbolic field names in `load_gsm_symbolic` (registry note C12). Check against the live
  dataset on the first GPU/network run.

---

## 9. Results, comparison, regressions and provenance (M8)

### 9.1 Before vs after (per arm)
For each trained arm key, average each problem's correctness over seeds (same as `report()` does), then
`paired_bootstrap(arm, base)` on the shared problem ids, per benchmark → `vs_base`.
**Regression** = `vs_base.ci_high < 0`.

### 9.2 Arm vs arm
Use exactly the pairs in `report()`: `targeted − matched_control`, `targeted − untargeted`,
`matched_control − untargeted`, `targeted − real_only` — only when both arms exist at the same ratio
(`real_only` is ratio 0 and pairs with any ratio). Per benchmark.

### 9.3 Other outputs
- **η change:** `eta_after − eta_before` per feature (η from `evaluate_model`'s refit on `probes_train_families`).
  **Sign:** η is a *cost*, so a **negative** change means that feature hurts **less** after training.
- **Error-type shift:** share of each error type among wrong answers, after minus before.
- **Target slice:** before/after accuracy above and at-or-below the threshold on both probe benchmarks.
- **Primary block (D8, status "proposed"):** the `targeted − matched_control` comparisons on `gsm8k` and
  `gsm_symbolic:main` (and p1/p2 if present).
- **Generic compare:** `compare_runs(before_id, after_id)` for any two eval runs (used by `GET /compare`).

### 9.4 Provenance and paper hygiene
- Every job and every run stores: `mode`, git commit + dirty flag (`git rev-parse HEAD`, "unknown" if git is
  missing), Python and package versions (torch, transformers, peft, bitsandbytes, datasets), seeds,
  `prompt_version`, GPU name, OS, and the Hugging Face model revision (commit sha; from `huggingface_hub`
  or the loaded model config).
- Precision is part of provenance: `train.load_in_4bit`, `eval.load_in_4bit`, and whether bitsandbytes imported
  with CUDA. There is **no automatic fallback** from QLoRA to plain LoRA: if 4-bit fails, the step fails with a clear
  message (S5) and the user changes the config on purpose. Paper mode uses one fixed config, so all arms always share
  the same precision. `compare_runs` adds a warning when the two runs used different precision.
- `paper_eligible = (mode == "paper") and all steps done and no overrides and dirty == False`.
  If the working tree is dirty, the paper run still runs but is flagged `paper_eligible: false` with a warning.
- Any "paper report" command must refuse explore jobs.

---

## 10. Work order (milestones) and "done" criteria

Do these in order. **Stop after each one** and print the milestone report (section 10.1).
CPU-only milestones can be done without the GPU; B1 and B8 need the user at the laptop.

| # | Milestone | Done when |
|---|---|---|
| **B0** | **Windows hardening + project memory.** Add `encoding="utf-8"` to every `read_text`/`write_text`/`open` (today missing in `experiments/pipeline.py` lines with `screen.json`, `diagnosis.json`, `manifest.json`, `evals/*.json`, `report.json`, `report.md`, `Config.load`; and `train/qlora.py` `train_manifest.json`). `dataloader_num_workers=0`. Clear bitsandbytes error (S5). Regression test that writes `report.md` with `±` and `Δ` while Python's default encoding is forced to cp1252. Root `CLAUDE.md` (project memory: novelty, rules from section 0, branch, Windows conventions, "explore ≠ paper", pointers to docs, **the current API contract version and endpoint list** — updated whenever the contract changes). Replace WSL steps in `docs/RUNNING.md` with native Windows steps; fix "16 jobs" → 18. | `pytest -q` passes on Linux; user runs it on Windows. |
| **B1** | **GPU smoke run (user present).** `configs/smoke.yaml` end to end with the existing CLI. Fix what breaks on Windows (bitsandbytes 4-bit path, real model output format, GSM-Symbolic fields). Record measured throughput (examples/s for eval, seconds/step for training, peak VRAM, **model-load time separately from run time**, GPU max temperature and lowest SM clock) and the disk size of one run in `docs/RUNNING.md`, replacing guesses. | Smoke run finishes; numbers written down. **No new features until this passes** for anything GPU-dependent. |
| **B2** | **Benchmark module (M6)** + importer + extractors + prompt versions (section 8). `docs/BENCHMARKS.md`. | Tests: registry, plug-in loading, standalone CLI with `EchoRunner`, importer + 3 extractors + resume on a fixture, prompt-version separation. |
| **B3** | **Store changes** (section 7.1) with migrations. Also: `get_responses` filters (`correct`, `error_type`, `source`) plus a matching count for pagination; `list_runs` filter by `job_id`. | Tests: new tables, migration on an old DB file adds columns without losing rows, WAL on, filters + counts. |
| **B4** | **Pipeline orchestrator** (`dreammachine/jobs/steps.py`, `execute.py`, CLI `run`): step list, per-pipeline folder/name, S0–S7, reuse, `build_arm` split, `target.json`, TrainerCallback progress. | Tests with `SimulatedRunner`-style fake runner + tiny model: explore pipeline end to end; paper step list = 18 train + 18 eval; reuse skips S1–S2 only when both hashes match; reuse of a benchmark run without diagnosis skips S1 but runs S2; baseline mismatch rejected; `untargeted` strategy forbids matched_control; `build_data` output byte-identical to before. |
| **B5** | **Worker** (queue, subprocess per step, cancel, resume, heartbeat, crash recovery, logs). | Tests with a fake executor: queued → running → done; failure → failed; cooperative cancel (step sees the flag and exits `cancelled`); a step that ignores the flag is hard-killed after the grace period; resume continues after the failed step; incomplete `checkpoint-*` folder is ignored on resume; stale `running` step becomes `interrupted` on worker start; delete refuses paper and running jobs. |
| **B6** | **Comparison + results** (section 9) → `results.json`, `report.md`, `compare_runs`, provenance, `paper_eligible`. | Tests on stored fake eval runs: regression flag, pairs, η sign, explore never paper-eligible. |
| **B7** | **API v1** exactly as `docs/API_CONTRACT.md` (incl. 202-with-warning when the worker is down, `DELETE` rules from 7.8, dataset download, UTF-8-safe log chunks): all routes under `/api/v1`, pydantic response models, error envelope, CORS, ISO times, `python -m dreammachine.api.export_openapi` → `docs/openapi.json`, `python -m dreammachine.serve`. Move the old unprefixed routes under `/api/v1` with the contract's names (`/generate` → `/tools/generate`, `/diagnose` → `/tools/classify`, `/lltm/fit` → `/tools/lltm-fit`; no frontend depends on the old ones yet) and update `tests/test_api.py`. | Contract test passes; every endpoint has at least one TestClient test; `docs/openapi.json` is up to date. |
| **B8** | **Real end-to-end on GPU (user present).** Start `serve`, `POST /api/v1/pipelines` explore run for `Qwen/Qwen3-0.6B` with `benchmark_limit: 50`; watch progress; read results. Also the prompt comparison (8.5). | Pipeline `done` on the GPU; results readable via API; numbers recorded. |

After B8 (not in this phase): the paper run, then the frontend (Manus AI).

### 10.1 Milestone report (print this at the end of every milestone)
```
MILESTONE <id> — <title>
What changed: <files, 1 line each>
Tests: <pytest summary line>  | CPU-verified / GPU-verified
Components: <ids moved to IMPLEMENTED>
Contract: <version> — <changes or "no change">
Open questions for the team: <list or "none">
CONTRACT DIGEST (paste to advisor):
  version: <x.y.z>
  base: http://127.0.0.1:8000/api/v1
  endpoints: <METHOD path, one per line>
  changed since last digest: <list>
```

---

## 11. Testing rules

- Every new module gets tests. Use `EchoRunner`, a simulated runner, or the `tiny_model_dir` fixture. No network
  in tests (use local JSONL fixtures). Mark real-GPU tests `@pytest.mark.gpu`, downloads `@pytest.mark.slow`.
- **Contract test** `tests/test_api_contract.py`:
  1. Parse the endpoint table between `<!-- ENDPOINTS:START -->` and `<!-- ENDPOINTS:END -->` in
     `docs/API_CONTRACT.md`; every `(method, path)` must exist in `create_app().openapi()`, and every route in
     the app under `/api/v1` must be in the table.
  2. `docs/openapi.json` must equal `create_app().openapi()` (so it was regenerated).
  3. The contract version in `docs/API_CONTRACT.md` must equal the app's `api_version`.
- A test passing on CPU never upgrades a component note to "GPU-verified".

---

## 12. Fact sheet — what exists and what does not (check before you assume)

**Exists today (verified 2026-10-04):**
- Stage functions: `screen`, `diagnose`, `build_data`, `train_arm`, `evaluate_model`, `report`, `paired_bootstrap`,
  `eval_sets`, `adapter_dir`, `hf_runner_factory` in `dreammachine/experiments/pipeline.py`. `ARMS` tuple.
- CLI: `python -m dreammachine.experiments.run <screen|diagnose|build-data|train|evaluate|report> --config ...`.
- `evaluate(runner, examples, chunk=64, progress=None)`, `summarize`, `item_counts`, `slice_accuracy`.
- `Store`: `create_run`, `finish_run`, `get_run`, `list_runs`, `add_responses`, `get_responses(limit, offset)`,
  `put_artifact`, `get_artifact`, `list_artifacts`. Tables `runs`, `responses`, `artifacts` only.
- Prompt: one `INSTRUCTION` in `models/prompts.py`; Qwen3 thinking disabled via `enable_thinking=False`.
- Extraction: `extract_answer` (`####` > `\boxed{}` > "answer is" > last number), strips `<think>`.
- Configs: `configs/smoke.yaml`, `configs/main.yaml`. Version `0.1.0`.

**Does NOT exist yet (you will build it):**
- `dreammachine/benchmarks/`, `dreammachine/jobs/`, `dreammachine/serve.py`, `configs/explore.yaml`,
  `configs/models.yaml`, `docs/BENCHMARKS.md`, `docs/openapi.json`, `CLAUDE.md`.
- Tables `jobs`, `job_steps`, `worker_state`; columns `runs.job_id`, `runs.mode`.
- `/api/v1` prefix, CORS, error envelope, any job/pipeline endpoint.
- `build_arm`, `target.json`, `compare_runs`, `results.json`, TrainerCallback progress.
- Prompt registry, `v2_700` text (must come from the user), `extract_lastline`, `extract_lastline_v0`, importer.
- **Any GPU result. Any measured timing.**

---

## 13. Windows laptop setup (for docs and for you)

- User installs: NVIDIA driver, Git for Windows, Python 3.11 (python.org, "Add to PATH"), Claude Code.
- Clone outside OneDrive, e.g. `C:\dev\EL-MAIN`; `git config core.longpaths true`.
- PowerShell venv: `python -m venv .venv`; if activation is blocked:
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`; then `.\.venv\Scripts\Activate.ps1`.
- CUDA torch from the official PyTorch index URL, then `pip install -e ".[dev,api,ml]"`.
- GPU check: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.
- Hugging Face login for gated models: `huggingface-cli login` (never put tokens in configs or commits).
- Optional: set `HF_HOME` to a drive with ~20 GB free.
- The API binds to `127.0.0.1` only and has no authentication. Do not expose it to a network.

---

## 14. Open questions (do not decide these alone — ask the user)

| # | Question | Needed before |
|---|---|---|
| 1 | D8: confirm the headline result (aimed − matched on GSM8K / GSM-Symbolic) before editing `RESEARCH.md`. | the paper run (the backend computes all versions anyway) |
| 2 | Match-quality thresholds that should **fail** a paper run (today: warnings only). | the paper run |
| 3 | ~~Which research model after screening.~~ **RESOLVED 2026-10-05 (D12):** both Qwen3-0.6B and Qwen3-1.7B, to study the effect of model size. Dev-slice measurements in `docs/RUNNING.md` §7. | — |
| 4 | ~~Exact `v2_700` prompt text from the friend.~~ **RESOLVED — text received 2026-10-04**, recorded verbatim in `CLAUDE.md` ("Prompt versions"); registered in B2 as `PROMPTS["v2_700"]` (SYSTEM message). | — |
| 5 | `docs/RESEARCH.md` cites "O'Grady & Ramlan (2026), arXiv 2607.18266". It could not be found online — the team must check it before citing, or remove it. | any paper draft |
| 6 | Before going public: LICENSE, `CITATION.cff`, GitHub Actions CI (Windows + Ubuntu running `pytest`). | the public release |

The model-size limit is **not** open: `docs/RESEARCH.md` already fixes it at ≤ 2B parameters.

---

## 15. Changes in version 1.1 (after an outside review)

- Frontend reality: Manus cannot reach `127.0.0.1`; it builds against mock mode (D4, contract §2).
- Cooperative cancel, because `Popen.terminate()` is a hard kill on Windows (7.3, S5, S6); incomplete checkpoints are
  ignored on resume (S5).
- Reuse now checks a separate `diagnosis_hash`; reusing a benchmark run without diagnosis runs diagnosis fresh (6.3).
- Precision recorded in provenance; no automatic QLoRA → LoRA fallback (9.4).
- Disk: `preflight.min_free_gb`, checkpoint cleanup, sizes in the API, delete for explore/benchmark jobs only (S0, 7.8).
- Timing: `load_s` vs `run_s` per step, a simple ETA (7.4); GPU temperature/clock logging (7.7).
- `CLAUDE.md` holds the contract version and endpoint list (B0); B1 records load time, thermals and run size.
- Open questions now say which milestone they block (14).
