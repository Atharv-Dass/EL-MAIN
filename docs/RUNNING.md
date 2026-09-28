# Running the experiment on your GPU (RTX 4060 Laptop, 8 GB)

## 0. One-time setup
**Windows:** use **WSL2 + Ubuntu**. bitsandbytes and most of the CUDA tooling are far less painful
there than on native Windows. Install the NVIDIA Windows driver; WSL2 picks it up automatically.

```bash
git clone https://github.com/yorudamn12/EL-MAIN && cd EL-MAIN
git checkout claude/wonderful-hypatia-pqs6t9
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA build of torch
pip install -e ".[dev,api,ml]"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pytest -q                                        # should all pass
huggingface-cli login                            # needed for the gated Llama / Gemma models
```

**Letting Claude drive the GPU directly:** open this folder in the Claude Desktop app, or run
`claude remote-control` from inside it. That session runs on your laptop, so it can use the GPU.

## 1. Smoke run (≈15–30 min): proves your setup works end to end
```bash
python -m dreammachine.experiments.run screen     --config configs/smoke.yaml
python -m dreammachine.experiments.run diagnose   --config configs/smoke.yaml
python -m dreammachine.experiments.run build-data --config configs/smoke.yaml
python -m dreammachine.experiments.run train      --config configs/smoke.yaml --arm targeted --ratio 3 --seed 0
python -m dreammachine.experiments.run evaluate   --config configs/smoke.yaml --arm targeted --ratio 3 --seed 0
```

## 2. Main experiment (`configs/main.yaml`)
| Stage | Command | Rough cost on a 4060 | Human checkpoint |
|---|---|---|---|
| Screen | `run screen` | ~1–2 h for 4 models | Pick `model` from `runs/main/screen.json`, then CONFIRM it in COMPONENTS.md |
| Diagnose | `run diagnose` | ~1–2 h | Annotate `runs/main/annotate_errors.csv` (2 people, blind), compute κ, CONFIRM the target factor |
| Build data | `run build-data` | minutes (CPU) | Check `data/manifest.json`: equal tokens per arm, match KS small |
| Train | `run train --all` | 16 jobs × ~30–60 min | — |
| Evaluate | `run evaluate --arm base` then `run evaluate --all` | ~20–40 min per job | — |
| Report | `run report` | seconds | Read `runs/main/report.md` |

Estimates are [Guessing]-grade until the smoke run gives real throughput numbers.

**If you run out of memory:** lower `generation.batch_size` (inference), or lower
`train.per_device_batch_size` to 2 and raise `grad_accum` to 8 (same effective batch).
Training resumes from the last checkpoint if interrupted: rerun the same command.

## 3. The human-validation step (the 150 errors)
1. Hide the `classifier_label` column. Two teammates fill `annotator_1` / `annotator_2`
   independently, using the labels FORMAT_ERROR, ARITHMETIC_SLIP, DISTRACTOR_USE, PLAN_ERROR,
   UNVERIFIABLE.
2. Then compute agreement:
```python
import csv
from dreammachine.diagnosis import cohen_kappa
rows = list(csv.DictReader(open("runs/main/annotate_errors.csv")))
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
```bash
DREAMMACHINE_DB=runs/dreammachine.db uvicorn dreammachine.api.app:app --reload
# http://127.0.0.1:8000/docs
```
