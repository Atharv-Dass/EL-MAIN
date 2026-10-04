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

## 6. Measured numbers (RTX 4060 Laptop, 8 GB)
Not measured yet — filled in by milestone B1 (smoke run).
