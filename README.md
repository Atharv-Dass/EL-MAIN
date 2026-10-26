# EL MAIN: DreamMachine

DreamMachine finds **where a small language model fails at multi-step arithmetic word problems**, writes new
training problems **by code** aimed at exactly that weakness, fine-tunes the model on them, and tests it again.

The research question: *does aiming the training data at the weakness help, or does any hard data help just as
much?* To answer it, the **targeted** training set is compared with a **matched control** that is just as hard
but hard for a *different* reason.

This README is written so that someone who has never seen the project can install it, run every feature and read
the results. Everything is done on your own computer; nothing is sent to a cloud service.

**Contents**
1. [What you get (and what we found)](#1-what-you-get-and-what-we-found)
2. [What you need](#2-what-you-need)
3. [Install (Windows)](#3-install-windows) · [Install (Linux)](#31-install-linux)
4. [Check that it works](#4-check-that-it-works)
5. [Try it without a GPU](#5-try-it-without-a-gpu)
6. [Start the program (the API)](#6-start-the-program-the-api)
7. [Every feature, step by step](#7-every-feature-step-by-step)
8. [The same features from the command line](#8-the-same-features-from-the-command-line)
9. [Reading the results](#9-reading-the-results)
10. [Explore vs paper runs](#10-explore-vs-paper-runs)
11. [Reference tables](#11-reference-tables)
12. [Problems and fixes](#12-problems-and-fixes)
13. [Backups, safety, rules](#13-backups-safety-rules)
14. [Where everything is](#14-where-everything-is)

---

## 1. What you get (and what we found)

| Feature | What it does | Needs a GPU? |
|---|---|---|
| **Problem generator** | Makes arithmetic word problems whose answer is correct by construction, with known properties (number of steps, digits, carries, ...). | No |
| **Answer checker / error classifier** | Reads a model's answer and says if it is right, and if wrong, why (format, arithmetic slip, distractor, wrong plan). | No |
| **Weakness statistics (LLTM)** | Finds which problem property (for example "more than 4 steps") makes the model fail. | No (needs model answers first) |
| **Benchmark run** | Tests a model on GSM8K, GSM-Symbolic and generated problems, and optionally diagnoses its weakness. | Yes |
| **Pipeline** | The whole experiment: test, diagnose, pick the target, build training data for each arm, train (QLoRA), re-test, compare. | Yes |
| **Results and report** | Before/after accuracy per training arm, with confidence intervals, plus a readable `report.md`. | No (reads saved results) |
| **Web API** | All of the above over HTTP at `http://127.0.0.1:8000/api/v1`. A separate frontend will use it. | Only to run jobs |

Models: `Qwen/Qwen3-0.6B` and `Qwen/Qwen3-1.7B` (the research models). Training uses 4-bit QLoRA, which fits on one
RTX 4060 laptop GPU (8 GB).

**What we found (both paper runs are finished):** on real test problems (GSM8K, GSM-Symbolic) there is **no
difference between targeted and matched-control training** for either model (differences of about one point, all
confidence intervals include 0). Every trained model also got worse on real test problems (about 14-24 points),
because the training solutions are short code-written texts. On generated test problems, all generated training data
helped a lot. Details: [`docs/RUNNING.md`](docs/RUNNING.md) §14-15 and [`docs/weakness.md`](docs/weakness.md).

---

## 2. What you need

| Need | Why |
|---|---|
| **Windows 11** (main target) or Linux | Windows needs no WSL. |
| **Python 3.10 or newer** (3.11 recommended) | Tick "Add to PATH" when installing from python.org. |
| **Git** | To download the project. |
| **NVIDIA GPU with 8 GB or more, plus the NVIDIA driver** | Only for benchmark runs, pipelines and training. Without a GPU you can still use the generator, the classifier, the tests and read saved results. |
| **About 20 GB free disk** | Model and dataset downloads from Hugging Face (set `HF_HOME` to choose the drive). A pipeline needs at least 10 GB free; a paper run writes a few GB into `runs/`. |
| **Internet** (first run) | Downloads models and the GSM8K / GSM-Symbolic datasets. See [offline use](#12-problems-and-fixes) if you have none. |

Do **not** clone into a OneDrive (or other synced) folder: sync tools can lock the SQLite database files.

---

## 3. Install (Windows)

Open **PowerShell** and run these one at a time.

```powershell
git config --global core.longpaths true
git clone https://github.com/yorudamn12/EL-MAIN C:\dev\EL-MAIN
cd C:\dev\EL-MAIN
git checkout claude/wonderful-hypatia-pqs6t9

python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
If PowerShell refuses to activate: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then try again.
After activation your prompt starts with `(.venv)`. **Do this activation in every new PowerShell window.**

Install the packages. Pick **one** of the two lines:

```powershell
# A) Everything, including GPU training (needs an NVIDIA GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev,api,ml]"

# B) No GPU: generator, classifier, API, tests only
pip install -e ".[dev,api]"
```

Models that need a licence (Llama, Gemma) also need `huggingface-cli login`. The two Qwen research models do not.
**Never put a Hugging Face token in a config file or a commit.**

### 3.1 Install (Linux)
Same steps; replace the activation line:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,api,ml]"      # or ".[dev,api]" without a GPU
```

---

## 4. Check that it works

```powershell
python -m pytest -q
```
About 200 tests; takes roughly a minute and a half and needs no GPU. Expected last line: `200 passed`.

With the GPU stack installed, also check the GPU is seen:
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
It should print `True` and your GPU name. If it prints `False`, see [Problems and fixes](#12-problems-and-fixes).

---

## 5. Try it without a GPU

**Generate example problems** (watch the question, the worked solution and the answer):
```powershell
python -m dreammachine.generator.preview --n 5 --steps 4 --digits 2 --distractors 1
```
Options: `--n` how many, `--steps` number of calculation steps, `--digits` size of numbers, `--distractors` how many
irrelevant number sentences to add, `--merge` join two chains into one answer, `--split train|heldout` story
styles, `--family NAME`, `--seed N`.

**List the available test sets:**
```powershell
python -m dreammachine.benchmarks list
```

---

## 6. Start the program (the API)

All features are available through one local web API. Start it (and the background worker that runs GPU jobs) with:

```powershell
cd C:\dev\EL-MAIN
.\.venv\Scripts\Activate.ps1
python -m dreammachine.serve
```
Leave this window open. **Ctrl+C stops both** (a running step is marked "interrupted" and can be resumed later).

| Option | Meaning |
|---|---|
| `--port 8000` | Port (default 8000). |
| `--db runs/dreammachine.db` | Where results are stored. |
| `--no-worker` | API only. Start the worker yourself in another window: `python -m dreammachine.jobs.worker`. |

Now open **http://127.0.0.1:8000/docs** in a browser. This page lists every endpoint and lets you try each one
with a button ("Try it out"), so you do not have to type any code. Base URL for programs:
`http://127.0.0.1:8000/api/v1`.

Safety: the API listens on **127.0.0.1 only** and has **no password**. Never expose it to a network.

Quick check that it is alive:
```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health
Invoke-RestMethod http://127.0.0.1:8000/api/v1/system     # GPU, worker status
```
Only **one worker** can run at a time, and it runs one job step at a time. Jobs wait in a queue.

---

## 7. Every feature, step by step

The examples use PowerShell. In the browser docs page (`/docs`) the same calls are buttons. The full description of
every field is in [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md).

### 7.1 See what you can choose
```powershell
$api = "http://127.0.0.1:8000/api/v1"
Invoke-RestMethod "$api/models"        # models you may pick
Invoke-RestMethod "$api/benchmarks"    # test sets you may pick
Invoke-RestMethod "$api/presets"       # defaults and allowed values for explore and paper
Invoke-RestMethod "$api/meta"          # features, error types, arms, stages with plain-English labels
Invoke-RestMethod "$api/components"    # which parts of the project are built / confirmed
Invoke-RestMethod "$api/queue"         # what the worker is doing and what is waiting
```

### 7.2 Test a model without training (benchmark run)
Measures how accurate a model is and, optionally, which problem property makes it fail.
```powershell
$body = @{ name="qwen06-check"; model="Qwen/Qwen3-0.6B"; benchmarks=@("gsm8k","gsm_symbolic:main");
           benchmark_limit=300; include_diagnosis=$true } | ConvertTo-Json
$job = Invoke-RestMethod -Method Post -Uri "$api/benchmark-runs" -Body $body -ContentType "application/json"
$job.id
```
Then **poll every few seconds** until `status` is `done`, `failed` or `cancelled`:
```powershell
Invoke-RestMethod "$api/benchmark-runs/$($job.id)" | Select-Object status, current_step, progress
Invoke-RestMethod "$api/benchmark-runs/$($job.id)/baseline"     # accuracy per benchmark
Invoke-RestMethod "$api/benchmark-runs/$($job.id)/diagnosis"    # weakness ranking
```
`benchmark_limit` is the number of problems per test set (10 or more, or `null` for the full sets).

### 7.3 Run the whole experiment (pipeline)
The pipeline runs these stages in order: **preflight** (checks) → **baseline** (test the untrained model) →
**diagnose** (find the weakness) → **choose_target** → **build_data** (training problems for each arm) →
**train** → **evaluate** → **compare**.

Explore run (a practice run; you choose the settings):
```powershell
$body = @{ mode="explore"; name="qwen06-try1"; model="Qwen/Qwen3-0.6B";
           benchmarks=@("gsm8k","gsm_symbolic:main","probes_train_families","probes_heldout_families");
           benchmark_limit=300; target=@{strategy="auto"}; arms=@("targeted","matched_control");
           ratio=3; seeds=@(0) } | ConvertTo-Json -Depth 5
$p = Invoke-RestMethod -Method Post -Uri "$api/pipelines" -Body $body -ContentType "application/json"
$p.id
```
Paper run (the fixed official protocol; **only** `mode`, `name`, `model` are allowed):
```powershell
$body = @{ mode="paper"; name="paper-qwen17"; model="Qwen/Qwen3-1.7B" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$api/pipelines" -Body $body -ContentType "application/json"
```
A paper run takes **days** of GPU time (see [section 10](#10-explore-vs-paper-runs)).

The create call answers `202`. If the response contains a warning `worker_not_running`, the job is queued but no
worker is running: start it (`python -m dreammachine.jobs.worker`) or use `python -m dreammachine.serve`.

What the request fields mean:

| Field | Meaning |
|---|---|
| `target.strategy` | `auto` = aim at the weakest property found by the diagnosis; `feature` = you pick it with `target.feature`; `untargeted` = no target. |
| `arms` | Which training sets to compare: `real_only`, `untargeted`, `matched_control`, `targeted`. |
| `ratio` | Generated problems per real problem in the mix: 1, 3 or 9 (counted in tokens). |
| `seeds` | Random starting points, 0, 1, 2. More seeds show how much results vary by chance. |
| `reuse_from` | Id of an earlier explore pipeline or benchmark run of the **same** model and settings; skips the baseline (and diagnosis). |

Follow progress (poll every 2 seconds while `queued` or `running`):
```powershell
Invoke-RestMethod "$api/pipelines/$($p.id)" | Select-Object status, current_step, progress, eta_s, warnings
```
Look at each stage's output as soon as it exists:
```powershell
Invoke-RestMethod "$api/pipelines/$($p.id)/baseline"      # untrained model's accuracy
Invoke-RestMethod "$api/pipelines/$($p.id)/diagnosis"     # weakness ranking and the chosen target
Invoke-RestMethod "$api/pipelines/$($p.id)/data"          # training sets per arm, sizes, how well arms match
Invoke-RestMethod "$api/pipelines/$($p.id)/data/targeted_r3_s0/samples"   # example training problems (paged)
Invoke-RestMethod "$api/pipelines/$($p.id)/training"      # training progress and loss curves
Invoke-RestMethod "$api/pipelines/$($p.id)/results"       # before/after comparison
Invoke-RestMethod "$api/pipelines/$($p.id)/report"        # readable report (Markdown text)
Invoke-RestMethod "$api/pipelines/$($p.id)/log"           # the job log
```
(In `/data/{arm_key}/samples` the arm key looks like `targeted_r3_s0`; `/data/{arm_key}/download` downloads that
arm's training set as JSONL.)

### 7.4 Stop, restart, delete
```powershell
Invoke-RestMethod -Method Post "$api/pipelines/$($p.id)/cancel"   # stops at the next safe point (up to ~2 min)
Invoke-RestMethod -Method Post "$api/pipelines/$($p.id)/resume"   # after failed/cancelled: finished steps are kept
Invoke-RestMethod -Method Delete "$api/pipelines/$($p.id)"        # explore runs only; not while running
```
Paper runs cannot be deleted (they are protected). Benchmark runs use the same calls under `/benchmark-runs/{id}`.

### 7.5 Look at individual answers (drill-down)
```powershell
Invoke-RestMethod "$api/runs"                                  # all stored runs
Invoke-RestMethod "$api/runs/<run_id>"                         # one run and its stored artifacts
Invoke-RestMethod "$api/runs/<run_id>/responses?correct=false&error_type=ARITHMETIC_SLIP&limit=20"
Invoke-RestMethod "$api/compare?before=<eval_run_id>&after=<eval_run_id>"   # compare two evaluation runs
```
Each response shows the model's text, whether it was right, the error type, and checks such as `format_ok`
(final line is a bare number) and `truncated` (hit the token limit).

### 7.6 Playground tools (fast, CPU only, no GPU job)
Generate problems:
```powershell
$body = @{ n=3; seed=0; steps=3; digits=2; n_distractors=0 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$api/tools/generate" -Body $body -ContentType "application/json"
```
Classify one answer (compare a model answer with a reference):
```powershell
$body = @{ response="3 + 4 = 7. The answer is \boxed{7}"; question="Tom has 3 apples and buys 4 more. How many?";
           reference_solution="3+4=<<3+4=7>>7 #### 7" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$api/tools/classify" -Body $body -ContentType "application/json"
```
Fit the weakness model on your own data (`Q` = one row of property values per problem, `successes` = how many times
it was answered correctly, `trials` = how many tries):
`POST /tools/lltm-fit` with `{ "Q": [[...]], "successes": [...], "trials": [...], "feature_names": [...], "l2": 0.01 }`.
Limits: `n` 1-500, `steps` 1-12, `digits` 1-6, `n_distractors` 0-5.

### 7.7 Errors
Every failed call returns the same error shape: an `error` with a `code` and a plain `message` (for example
`unknown_model`, `paper_mode_locked`, `reuse_mismatch`, `not_found`, `conflict`). Full list:
[`docs/API_CONTRACT.md`](docs/API_CONTRACT.md) §3.1.

---

## 8. The same features from the command line

You do not need the API. These commands do the same work in the terminal. All commands run from the project
folder with the virtual environment active.

### 8.1 Jobs (pipelines and benchmark runs)
```powershell
# Run in the foreground (this window works until the job is finished)
python -m dreammachine.jobs run --preset explore --model Qwen/Qwen3-0.6B --limit 50
python -m dreammachine.jobs run --kind benchmark_run --model Qwen/Qwen3-1.7B --limit 300
python -m dreammachine.jobs run --preset explore --model Qwen/Qwen3-1.7B --target feature:n_carry --reuse-from <job id>
python -m dreammachine.jobs run --preset paper --model Qwen/Qwen3-0.6B            # the full protocol: days

# Or queue jobs and let a worker run them in the background
python -m dreammachine.jobs enqueue --preset explore --model Qwen/Qwen3-0.6B --limit 50
python -m dreammachine.jobs.worker          # keep this window open; Ctrl+C stops it

python -m dreammachine.jobs list            # all jobs, newest first
python -m dreammachine.jobs show <job id>   # steps and status of one job
python -m dreammachine.jobs cancel <job id>
python -m dreammachine.jobs resume <job id> # failed/cancelled -> queued; finished steps kept
python -m dreammachine.jobs delete <job id> # explore jobs only
```
Options of `run` / `enqueue`:

| Option | Meaning |
|---|---|
| `--preset explore\|paper` | Practice run (settings from `configs/explore.yaml`) or the locked protocol (`configs/main.yaml`). |
| `--kind pipeline\|benchmark_run` | Whole experiment (default) or just test + diagnose. |
| `--model` | Required. A model id from `configs/models.yaml`. |
| `--name` | A label for the job. |
| `--benchmarks a b c` | Test sets (see [reference](#11-reference-tables)). |
| `--limit N` / `--full` | Problems per test set (10 or more) / all of them. |
| `--target auto\|untargeted\|feature:<name>` | What to aim the training at. |
| `--arms ...`, `--ratio N`, `--seeds ...` | Which training sets, the mix ratio, the random seeds. |
| `--reuse-from <job id>` | Skip the baseline of an earlier matching run. |
| `--no-diagnosis` | Benchmark run without the diagnosis step. |

### 8.2 Benchmark a model directly
```powershell
python -m dreammachine.benchmarks run --model Qwen/Qwen3-0.6B --benchmark gsm8k --limit 200
python -m dreammachine.benchmarks run --model Qwen/Qwen3-0.6B --benchmark gsm8k --benchmark gsm_symbolic:main --limit 200
python -m dreammachine.benchmarks import-jsonl results.jsonl --benchmark gsm8k --model Qwen/Qwen2.5-1.5B-Instruct
```
`run` saves answers in `runs/benchmarks/` (append-only; run it again to continue) and the scores in the database.
Useful options: `--max-new-tokens 512`, `--batch-size 32`, `--load-in-4bit`, `--n-samples`, `--temperature`.
`import-jsonl` re-scores answers you got elsewhere (for example from a notebook).

### 8.3 The step-by-step experiment commands
For people who want to run each stage themselves with a config file (`configs/smoke.yaml` is a tiny test,
`configs/main.yaml` is the paper protocol):
```powershell
python -m dreammachine.experiments.run screen     --config configs/smoke.yaml
python -m dreammachine.experiments.run diagnose   --config configs/smoke.yaml
python -m dreammachine.experiments.run build-data --config configs/smoke.yaml
python -m dreammachine.experiments.run train      --config configs/smoke.yaml --arm targeted --ratio 3 --seed 0
python -m dreammachine.experiments.run evaluate   --config configs/smoke.yaml --arm base
python -m dreammachine.experiments.run evaluate   --config configs/smoke.yaml --arm targeted --ratio 3 --seed 0
python -m dreammachine.experiments.run report     --config configs/smoke.yaml
```
Run the smoke config first: it proves your GPU setup works end to end. More detail and measured timings:
[`docs/RUNNING.md`](docs/RUNNING.md).

### 8.4 Other commands
```powershell
python -m dreammachine.api.export_openapi    # rewrite docs/openapi.json after changing the API
python -m dreammachine.generator.preview --help
```

---

## 9. Reading the results

Every job gets its own folder: `runs/pipelines/<job id>/`.

| File / folder | What is in it |
|---|---|
| `report.md` | **Start here.** The readable report: baseline, each training arm vs the untrained model, targeted vs matched control, error-type shifts. |
| `results.json` | The same results as data (what the API's `/results` returns). |
| `diagnosis.json` | Every weakness found, how much it hurts (`eta`), its confidence interval, whether it is significant. |
| `target.json` | The one weakness chosen for training, for example `{"feature": "steps", "threshold": 4.0}` = "more than 4 steps". |
| `data/` | The training problems, one file per arm and seed. |
| `adapters/` | The trained add-ons (LoRA adapters). |
| `evals/` | Accuracy after each training and the baseline. |
| `logs/` | `job.log` and `gpu.csv` (temperature, clock, power, memory every 30 seconds). |
| `annotate_errors.csv` | A sample of wrong answers for humans to label. |

The database `runs/dreammachine.db` (SQLite) holds every single answer of every run.

**How to read the main number.** The headline is **targeted minus matched control, on real test problems**
(GSM8K and GSM-Symbolic). A positive number whose 95% confidence interval stays above 0 means targeting helps beyond
difficulty. If the interval includes 0 (as in both finished paper runs), there is no evidence of a targeting benefit.
Differences are given as fractions: 0.03 = 3 points. "Probes" are generated test problems with known properties;
they show whether the weakness itself improved. A "regression" means accuracy went down.

Plain-words write-ups of each model's weaknesses and whether training fixed them: [`docs/weakness.md`](docs/weakness.md).

### The human-validation step
The automatic error labels are checked by two people:
1. Open `runs/pipelines/<job id>/annotate_errors.csv`. Hide the `classifier_label` column.
2. Two people fill `annotator_1` and `annotator_2` independently with one of: `FORMAT_ERROR`, `ARITHMETIC_SLIP`,
   `DISTRACTOR_USE`, `PLAN_ERROR`, `UNVERIFIABLE`.
3. Compute agreement (kappa; 0.6 or more is usually called "substantial"):
```python
import csv
from dreammachine.diagnosis import cohen_kappa
rows = list(csv.DictReader(open("runs/pipelines/<job id>/annotate_errors.csv", encoding="utf-8")))
print("human-human", cohen_kappa([r["annotator_1"] for r in rows], [r["annotator_2"] for r in rows]))
print("classifier-human", cohen_kappa([r["classifier_label"] for r in rows], [r["annotator_1"] for r in rows]))
```

---

## 10. Explore vs paper runs

| | Explore | Paper |
|---|---|---|
| Purpose | Practice, trying settings | The official experiment |
| Settings | You choose (defaults in `configs/explore.yaml`) | Locked (`configs/main.yaml`); you only give model and name |
| Training | Usually 1 seed, 2 arms | 5 arm setups x 3 seeds = 15 trainings per model, 600,000 tokens each |
| Time (RTX 4060 laptop) | About 2.6 hours for Qwen3-0.6B with 50 problems per test set | Days (both models together about 4 days of GPU time) |
| Evidence | **Never** counts as paper evidence | Counts only if `paper_eligible` is true |
| Can be deleted | Yes | No (protected) |

A paper run is `paper_eligible` only if every step finished, nothing was overridden, and **the git working tree was
clean when it started**: commit all changes before starting one. Keep the laptop plugged in and not sleeping
(a pause only delays a run; it resumes). Windows can spill GPU memory into system RAM instead of failing; if the
laptop runs low on memory, stop the worker and `resume` the job.

---

## 11. Reference tables

**Models** (`GET /models`, `configs/models.yaml`)

| Id | Notes |
|---|---|
| `Qwen/Qwen3-0.6B` | Research model. |
| `Qwen/Qwen3-1.7B` | Research model. |
| `meta-llama/Llama-3.2-1B-Instruct` | Gated: needs the licence and `huggingface-cli login`. |
| `google/gemma-3-1b-it` | Gated. |
| `Qwen/Qwen2.5-1.5B-Instruct` | Used by an outside benchmark (prompt `v2_700`). |

**Test sets (benchmarks)**

| Name | What it is |
|---|---|
| `gsm8k` | The 1,319 grade-school GSM8K test problems. |
| `gsm_symbolic:main`, `:p1`, `:p2` | GSM-Symbolic: GSM8K templates with new names and numbers (`p1`/`p2` are harder variants). |
| `probes_train_families` | Generated problems in the story styles used for training. |
| `probes_heldout_families` | Generated problems in unseen story styles. |
| `jsonl:<path>` | Your own file (command line only). |

**Training arms**

| Arm | Meaning |
|---|---|
| `real_only` | Real GSM8K training problems only (the plain baseline). |
| `untargeted` | Generated problems not aimed at anything. |
| `matched_control` | Generated problems just as hard as targeted, but hard for a different reason. |
| `targeted` | Generated problems aimed at the weakness. |

**Problem properties (features)**: `steps` (calculation steps), `log10_max` (size of the largest number), `n_mul`
(multiplications), `n_div` (divisions), `n_carry` (carries and borrows in additions/subtractions),
`n_distractors` (irrelevant number sentences), `has_merge` (two chains joined).

**Error types**: `CORRECT`, `FORMAT_ERROR` (answer not readable), `ARITHMETIC_SLIP` (a calculation was wrong),
`DISTRACTOR_USE` (used an irrelevant number), `PLAN_ERROR` (wrong approach), `UNVERIFIABLE` (cannot tell).

**Job states**: `queued`, `running`, `done`, `failed`, `cancelled`. Stage names: `preflight`, `baseline`, `diagnose`,
`choose_target`, `build_data`, `train`, `evaluate`, `compare`.

**Prompt**: all runs use the `qwen_boxed` prompt (the Qwen3 model card's math prompt); training answers end with
`\boxed{n}`. Thinking mode is turned off by the code.

**Words**

| Word | Meaning |
|---|---|
| Baseline | The untrained model's answers: the "before". |
| Ratio 3:1 | 3 parts generated problems to 1 part real problems, counted in tokens. |
| Seed | The random starting point of a training; several seeds show chance variation. |
| QLoRA / adapter | Cheap fine-tuning: a small add-on is trained instead of the whole model, with the model stored in 4 bits. |
| LLTM | The statistics that find which problem property makes the model fail. |
| CI | 95% confidence interval: the range the true difference probably lies in. |

---

## 12. Problems and fixes

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'numpy'` (or another module) | The virtual environment is not active or packages are not installed. Run `.\.venv\Scripts\Activate.ps1`, then `pip install -e ".[dev,api]"`. |
| PowerShell blocks `Activate.ps1` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `torch.cuda.is_available()` prints `False` | Install the NVIDIA driver and the CUDA build of torch (`pip install torch --index-url https://download.pytorch.org/whl/cu128`). |
| 4-bit training stops with a clear error | 4-bit needs `bitsandbytes` with CUDA. There is deliberately no silent fallback. Models up to about 1.7B also fit in bf16 with `train.load_in_4bit: false`. |
| Out of memory | Lower `generation.batch_size` (testing) or set `train.per_device_batch_size: 2` and `grad_accum: 8` (training). Keep batches inside 8 GB: Windows may silently use system RAM, which is slow. |
| Training was interrupted | Run the same command again, or `resume` the job. Training continues from the last checkpoint. |
| API answers 202 with `worker_not_running` | Start the worker: `python -m dreammachine.jobs.worker` (or use `python -m dreammachine.serve`). |
| Port 8000 is busy | `python -m dreammachine.serve --port 8001`. |
| Browser frontend cannot call the API | Allowed origins are set by `DREAMMACHINE_CORS_ORIGINS` (default: ports 5173 and 3000 on localhost). Example: `$env:DREAMMACHINE_CORS_ORIGINS = "http://localhost:5173"`. |
| No internet / Hugging Face blocked | Export the datasets once elsewhere and point a config at the files (`eval_data: gsm8k_test: data/gsm8k_test.jsonl`, ...). Format: see `dreammachine.data.save_jsonl`. |
| Database locked | The project folder is in a OneDrive-synced folder. Move it elsewhere. |
| Paper run is not `paper_eligible` | The git tree was not clean at the start, a step failed, or something was overridden. Commit everything and start again. |
| A job failed | `python -m dreammachine.jobs show <job id>` shows the failing step; `runs/pipelines/<job id>/logs/job.log` has details; fix it, then `resume`. |

---

## 13. Backups, safety, rules

- **`runs/` is not in git** (it is too big). It holds all results and the database and exists only on your machine.
  Copy it somewhere safe after important runs (stop the worker first so the database is not being written).
- Never expose the API to a network (no password, `127.0.0.1` only).
- Never commit tokens or secrets.
- Training data is generated by code only; no language model writes training data.
- Explore runs are never paper evidence. Label results as measured on **GPU** or only checked by **CPU** tests.
- Run `python -m pytest -q` before committing code changes.

---

## 14. Where everything is

```
dreammachine/
  generator/    problem generator (exact arithmetic, property labels)
  diagnosis/    answer extraction, error classifier, LLTM statistics, targeting
  data/         GSM8K / GSM-Symbolic loaders, data mixing
  benchmarks/   test-set registry and runner
  models/       model runner and prompts            (GPU)
  train/        QLoRA fine-tuning                    (GPU)
  experiments/  pipeline stages and the step-by-step commands
  jobs/         queue, worker, job commands
  store/        SQLite results store
  api/          the web API (FastAPI)
configs/        explore.yaml, main.yaml (paper), smoke.yaml, models.yaml, tuning configs
tests/          the automatic tests
docs/           documentation (below)
frontend_handoff/  files for building the frontend, and the log of backend changes it must follow
runs/           results and database (created when you run; not in git)
```

| Document | What it is for |
|---|---|
| [`docs/RUNNING.md`](docs/RUNNING.md) | Detailed run guide and every measured number (timings, accuracies, paper-run tables). |
| [`docs/WORK.md`](docs/WORK.md) | Plain-words work log and next steps. |
| [`docs/weakness.md`](docs/weakness.md) | Each model's weaknesses and whether training fixed them. |
| [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md) | The exact API: every endpoint, field and type. `docs/openapi.json` is the machine-readable copy. |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | The test sets and answer extractors. |
| [`docs/RESEARCH.md`](docs/RESEARCH.md) | The science: novelty, mathematics, experiment design. |
| [`docs/COMPONENTS.md`](docs/COMPONENTS.md) | Which parts are built and which the team has confirmed. |
| [`PLAN.md`](PLAN.md) | The build plan and open questions. |
| [`CLAUDE.md`](CLAUDE.md) | Project rules for the AI coding assistant. |
| [`frontend_handoff/`](frontend_handoff/README.md) | Everything the frontend builder needs. |

Project status: the backend (milestones B0-B8) is complete and both paper runs are finished. The frontend will be built separately using the API contract. There is no `LICENSE` file yet: ask the team before reusing the code.
