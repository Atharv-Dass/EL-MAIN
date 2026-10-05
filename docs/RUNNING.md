# Running the experiment on your GPU (RTX 4060 Laptop, 8 GB)

## 0. One-time setup (native Windows, PowerShell)
No WSL needed: torch (CUDA build) and bitsandbytes 4-bit both run natively on Windows.

Install first: the NVIDIA driver, Git for Windows, Python 3.11 (python.org, tick "Add to PATH").
Clone **outside OneDrive-synced folders** (sync tools can lock the SQLite files).

```powershell
git config --global core.longpaths true
git clone https://github.com/yorudamn12/EL-MAIN C:\dev\EL-MAIN
cd C:\dev\EL-MAIN
git checkout claude/wonderful-hypatia-pqs6t9
python -m venv .venv
# If activation is blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA build of torch
pip install -e ".[dev,api,ml]"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m pytest -q                                 # should all pass
huggingface-cli login                               # only for the gated Llama / Gemma models
```
Never put Hugging Face tokens in configs or commits. Optional: set `HF_HOME` to a drive with ~20 GB free.

Linux (bash) is the same with `python3 -m venv .venv && source .venv/bin/activate`.

**Letting Claude drive the GPU directly:** open this folder in Claude Code on the laptop; that session can use the GPU.

## 1a. One pipeline, one command (B4)
Every stage from the baseline benchmark to the comparison, in the foreground (the worker for queued jobs comes in B5).
Each job gets its own folder `runs/pipelines/<job id>/`. **Explore runs are never paper evidence.**
```powershell
python -m dreammachine.jobs run --preset explore --model Qwen/Qwen3-0.6B --limit 50
python -m dreammachine.jobs run --kind benchmark_run --model Qwen/Qwen3-1.7B --limit 300          # baseline + diagnosis only
python -m dreammachine.jobs run --preset explore --model Qwen/Qwen3-1.7B --target feature:n_carry --reuse-from <job id>
python -m dreammachine.jobs run --preset paper --model Qwen/Qwen3-0.6B                             # the full protocol (days)
```
`--reuse-from` skips the baseline (and the diagnosis, if its settings match) of an earlier explore pipeline or benchmark
run. Explore defaults: `configs/explore.yaml` (starting guesses until B8 measures them).

**Queue + worker (B5)** — queue jobs, and let one worker run them in the background (one step per subprocess,
so GPU memory is freed after every step):
```powershell
python -m dreammachine.jobs enqueue --preset explore --model Qwen/Qwen3-0.6B --limit 50
python -m dreammachine.jobs.worker                  # keep this window open; Ctrl+C stops it (the running step fails
                                                    # as "interrupted" and can be resumed)
python -m dreammachine.jobs list
python -m dreammachine.jobs show <job id>
python -m dreammachine.jobs cancel <job id>         # stops at the next safe point; force-killed after 120 s
python -m dreammachine.jobs resume <job id>         # failed/cancelled -> queued; finished steps are kept
python -m dreammachine.jobs delete <job id>         # explore jobs only, not while queued/running
```
**Results (B6):** the last step writes `runs/pipelines/<job id>/results.json` (the API's `PipelineResults`) and a
readable `report.md`: baseline, the primary result (targeted − matched control; "proposed" until the team confirms
D8), each arm vs the untrained model with regressions, arm vs arm, change in η (negative = the feature hurts less),
error-type shift, and provenance. Paper runs are flagged `paper_eligible` only if every step finished, nothing was
overridden and the git working tree was clean when the run started — commit before starting a paper run.

Logs: `runs/pipelines/<job id>/logs/job.log`; GPU readings every 30 s: `logs/gpu.csv` (temperature, SM clock, power,
utilisation, memory), summarised per step. Only one worker can run at a time.

## 1. Smoke run (proves your setup works end to end)
```powershell
python -m dreammachine.experiments.run screen     --config configs/smoke.yaml
python -m dreammachine.experiments.run diagnose   --config configs/smoke.yaml
python -m dreammachine.experiments.run build-data --config configs/smoke.yaml
python -m dreammachine.experiments.run train      --config configs/smoke.yaml --arm targeted --ratio 3 --seed 0
python -m dreammachine.experiments.run evaluate   --config configs/smoke.yaml --arm base
python -m dreammachine.experiments.run evaluate   --config configs/smoke.yaml --arm targeted --ratio 3 --seed 0
python -m dreammachine.experiments.run report     --config configs/smoke.yaml
```

## 2. Main experiment (`configs/main.yaml`)
| Stage | Command | Rough cost on a 4060 | Human checkpoint |
|---|---|---|---|
| Screen | `run screen` | ~1–2 h for 4 models | Pick `model` from `runs/main/screen.json`, then CONFIRM it in COMPONENTS.md |
| Diagnose | `run diagnose` | ~1–2 h | Annotate `runs/main/annotate_errors.csv` (2 people, blind), compute κ, CONFIRM the target factor |
| Build data | `run build-data` | minutes (CPU) | Check `data/manifest.json`: equal tokens per arm, match KS small |
| Train | `run train --all` | 18 trainings × ~30–60 min | — |
| Evaluate | `run evaluate --arm base` then `run evaluate --all` | ~20–40 min per evaluation | — |
| Report | `run report` | seconds | Read `runs/main/report.md` |

`--all` = per seed `real_only_r0` + `untargeted_r3` + `matched_control_r3` + `targeted_r1/r3/r9` = 6, × 3 seeds = 18
trainings and 18 evaluations (plus the base evaluation).
Estimates are [Guessing]-grade until the smoke run gives real throughput numbers (section 6).

**If you run out of memory:** lower `generation.batch_size` (inference), or lower
`train.per_device_batch_size` to 2 and raise `grad_accum` to 8 (same effective batch).
Training resumes from the last checkpoint if interrupted: rerun the same command.

**4-bit:** `train.load_in_4bit: true` (QLoRA) needs bitsandbytes with CUDA. If it cannot run, the step stops
with a clear error; there is no silent fallback. Models up to ~1.7B also fit in bf16 with `load_in_4bit: false`.

## 3. The human-validation step (the 150 errors)
1. Hide the `classifier_label` column. Two teammates fill `annotator_1` / `annotator_2`
   independently, using the labels FORMAT_ERROR, ARITHMETIC_SLIP, DISTRACTOR_USE, PLAN_ERROR,
   UNVERIFIABLE.
2. Then compute agreement:
```python
import csv
from dreammachine.diagnosis import cohen_kappa
rows = list(csv.DictReader(open("runs/main/annotate_errors.csv", encoding="utf-8")))
print("human-human", cohen_kappa([r["annotator_1"] for r in rows], [r["annotator_2"] for r in rows]))
print("classifier-human", cohen_kappa([r["classifier_label"] for r in rows], [r["annotator_1"] for r in rows]))
```
κ ≥ 0.6 between the classifier and humans is the usual bar for "substantial" agreement.

## 4. Offline datasets
If Hugging Face is unreachable, export once elsewhere and point the config at the files:
```yaml
eval_data:
  gsm8k_test: data/gsm8k_test.jsonl
  gsm8k_train: data/gsm8k_train.jsonl
  gsm_symbolic_p1: data/gsm_symbolic_p1.jsonl
```
(The format is `Example` rows: see `dreammachine.data.save_jsonl`.)

## 5. API (for the frontend, later)
```powershell
$env:DREAMMACHINE_DB = "runs/dreammachine.db"
uvicorn dreammachine.api.app:app --host 127.0.0.1 --port 8000
# http://127.0.0.1:8000/docs
```

## 6. Measured numbers (RTX 4060 Laptop, 8 GB) — GPU-verified, B1 smoke run, 2026-10-04
Setup: native Windows 11, Python 3.11.9, torch 2.11.0+cu128, transformers 5.18.0, peft 0.21.2,
bitsandbytes 0.50.2, driver 581.29, on AC power, "Balanced" power plan. Model `Qwen/Qwen3-0.6B`, `configs/smoke.yaml`
(eval in bf16, `max_new_tokens: 256`, `batch_size: 16`; training 4-bit QLoRA, batch 4, 30 steps).
One measurement on one laptop: treat as a rough guide, not a benchmark.

| Stage | Wall time | Model load | Work | Throughput | Peak VRAM (allocated / reserved) |
|---|---|---|---|---|---|
| screen (66 problems, greedy) | 200 s | 18.8 s | 177 s | 0.37 problems/s | 2.0 / 2.3 GB |
| diagnose (216 probes × 2 samples, T=0.7) | 570 s | 15.5 s | 554 s | 0.39 problems/s = 0.78 answers/s | 2.9 / 3.3 GB |
| build-data (CPU) | 12 s | — | — | — | — |
| train targeted_r3_s0 (4-bit QLoRA, 30 steps) | 80 s | ~25 s (load + tokenise + save) | 54.6 s | **1.82 s/step**, 2.2 samples/s | 4.0 / 6.7 GB |
| evaluate base (82 problems) | 242 s | 13.9 s | 223 s | 0.35–0.42 problems/s | 2.0 / 2.3 GB |
| evaluate targeted (base + LoRA adapter) | 338 s | 19.3 s | 309 s | 0.22–0.54 problems/s | 2.0 / 2.4 GB |
| report | < 5 s | — | — | — | — |

- **Whole smoke run:** ~24 min. Disk: 109 MB under `runs/smoke/` (adapter 49.5 MB, `checkpoint-30` 58.6 MB,
  data 0.4 MB) + 0.7 MB SQLite. Device memory seen by `nvidia-smi` peaked at 7.5 GB of 8 GB (during training, incl.
  other processes).
- **Thermals:** max GPU temperature 62 °C; SM clock while busy min 210 / mean 1193 / max 2655 MHz; mean power 19.5 W
  (max 54 W). No HW thermal or power-brake slowdown was recorded.
- **Inference is CPU-bound, not GPU-bound:** mean GPU utilisation while generating was only ~34 %. With a 0.6B model,
  Hugging Face `generate` spends most of each token step in Python/kernel-launch overhead, so larger
  `generation.batch_size` (e.g. 32–64) should raise throughput almost linearly at small VRAM cost — not yet measured.
- **Token limit and batch size (measured after B1, dev slice = last 200 GSM8K *train* items, Qwen3-0.6B, dm_v1,
  greedy, bf16):**

  | max_new_tokens | batch_size | accuracy | answers cut off | problems/s | peak VRAM |
  |---|---|---|---|---|---|
  | 256 | 16 | 0.495 | 17.5 % | 0.40 | 2.0 GB |
  | 512 | 16 | 0.560 | 0.5 % | 0.29 | 2.4 GB |
  | **512** | **32** | **0.565** | **0 %** | **0.52** | 3.5 GB |

  The configs now use 512 / 32. `batch_size` counts generated sequences (questions × `n_samples`), so the
  diagnose stage (2–3 samples per question) stays within the same memory. Each answer records `truncated`
  (hit the token limit); evaluation metrics include `<benchmark>:truncated_share`.
- **Batch 64 does not fit:** at 512 tokens it filled 7.75 of 8.19 GB and was still unfinished after 26 min
  (batch 32: 6.5 min). See "GPU memory spill" below.
- **Extrapolation (a guess, not measured):** at ~0.4 problems/s, the full GSM8K test (1319) takes ~55 min per model
  evaluation at 256 new tokens, longer at 700.
- Smoke config gap: the eval grid has `steps ∈ {2, 4}`, so with the smoke target `steps` (threshold 4.0) the
  "above threshold" slice is empty (`accuracy: null`). Plumbing only; the main config's grid covers it.

## 7. Accuracy exploration: models, prompts, decoding — GPU-verified, EXPLORE ONLY (2026-10-04/05)
**Not paper evidence.** Run with throwaway scripts (kept in `runs/dev_explore/`, git-ignored), no provenance.
For the paper's settings-selection table, re-run through the benchmark module (B2/B8) with provenance.

Dev set: the last 200 GSM8K **train** problems (never GSM8K test). Scored with the repo's own
`classify` / `extract_answer`. bf16, RTX 4060 Laptop, native Windows. "Boxed" = the prompt recommended in the
Qwen3 model card (Best Practices): *"Please reason step by step, and put your final answer within \boxed{}."*
appended to the question in the user turn. Thinking mode uses the model card's sampling (T=0.6, top-p 0.95,
top-k 20; greedy is discouraged there). Majority vote = 5 samples at T=0.7, top-p 0.8, top-k 20.

| Model | Prompt | Decoding | Accuracy | Cut off | Mean tokens | Time / 200 |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | dm_v1 | greedy, 512 tok | 0.565 | 0 % | 191 | 6 min |
| Qwen3-0.6B | **boxed** | greedy, 512 tok | **0.705** | 2 % | 254 | 8 min |
| Qwen3-0.6B | dm_v1 | thinking, 2048 tok | 0.590 | 14 % | 934 | 120 min |
| Qwen3-0.6B | boxed | thinking, 2048 tok | 0.720 | 27.5 % | 1325 | 129 min |
| Qwen3-0.6B | dm_v1 | majority of 5, 512 tok | 0.635 | 0.5 % | 193 | 37 min |
| Qwen3-1.7B | dm_v1 | greedy, 512 tok | 0.790 | 1.5 % | 242 | 11 min |
| Qwen3-1.7B | **boxed** | greedy, 512 tok | **0.860** | 2.5 % | 293 | 13 min |
| Qwen3-1.7B | dm_v1 | thinking, 2048 tok | 0.840 | 16 % | 1031 | 224 min |
| Qwen3-1.7B | boxed | thinking, 2048 tok | 0.785 | 31.5 % | 1540 | 266 min |
| Qwen3-1.7B | dm_v1 | majority of 5, 512 tok | 0.840 | 2 % | 241 | 67 min |

Paired bootstrap on the same 200 problems (difference, 95% CI):
- boxed − dm_v1 (greedy): 0.6B **+0.140** [+0.075, +0.205]; 1.7B **+0.070** [+0.015, +0.125].
- 1.7B − 0.6B (greedy): dm_v1 +0.225 [+0.155, +0.295]; boxed +0.155 [+0.090, +0.225].
- thinking − greedy: 0.6B dm_v1 +0.025 [−0.055, +0.105]; 0.6B boxed +0.015 [−0.055, +0.085];
  1.7B dm_v1 +0.050 [−0.015, +0.115]; 1.7B boxed **−0.075** [−0.135, −0.015] (31.5 % of answers hit the limit).
- majority-of-5 − greedy (dm_v1): 0.6B +0.070 [+0.015, +0.125]; 1.7B +0.050 [+0.015, +0.085].
- boxed greedy − dm_v1 majority-of-5: 0.6B +0.070 [+0.005, +0.135]; 1.7B +0.020 [−0.030, +0.070].

Reading (for the team to decide; nothing in the configs was changed for this):
- **The boxed prompt is the biggest cheap gain** on both models, at ~1.2–1.3× the time of dm_v1.
- **Thinking mode is not worth it here:** no significant gain anywhere, a loss on 1.7B + boxed, ~15–20× the
  time, and the error classifier cannot see the stripped thinking text.
- **Majority voting helps dm_v1 but costs 5×**, and boxed greedy is as good or better.
- Switching the project prompt to "boxed" is a D10 decision: training completions would then end with
  `\boxed{n}` instead of `#### n`, and dm_v1's request to write every calculation as an equation (which the
  error classifier relies on) would be gone. UNVERIFIABLE shares did not rise with boxed in these runs.
- Model choice (open question 3): 1.7B is more accurate but leaves fewer errors to diagnose and fix.

## 8. GPU memory spill on Windows (measured)
When a batch needs more than the 8 GB of dedicated VRAM, Windows does **not** raise out-of-memory: it moves the
overflow into shared system RAM and the run becomes many times slower (seen: thinking mode, Qwen3-0.6B, batch 16,
2048 tokens → 4.3 GB in shared memory, no result after ~2 h; batch 64 at 512 tokens likewise). Check with
`Get-Counter '\GPU Process Memory(*)\Shared Usage'`. Remedies used here: smaller batches (thinking: 0.6B batch 8,
1.7B batch 4, both stayed in VRAM) and `torch.cuda.set_per_process_memory_fraction(0.92)` so an oversized batch
fails with a clear OOM instead. Proposed for B4/B5: the same cap in step subprocesses, so a pipeline step fails fast
rather than crawling. (Alternative: NVIDIA Control Panel → "CUDA - Sysmem Fallback Policy" → "Prefer No Sysmem
Fallback"; a system setting for the user.)
