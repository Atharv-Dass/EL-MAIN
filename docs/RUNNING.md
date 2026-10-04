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
- **Extrapolation (a guess, not measured):** at ~0.4 problems/s, the full GSM8K test (1319) takes ~55 min per model
  evaluation at 256 new tokens, longer at 700.
- Smoke config gap: the eval grid has `steps ∈ {2, 4}`, so with the smoke target `steps` (threshold 4.0) the
  "above threshold" slice is empty (`accuracy: null`). Plumbing only; the main config's grid covers it.
