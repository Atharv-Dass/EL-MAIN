# EL MAIN: DreamMachine

Backend and frontend of the project (MAIN EL, Semester 3).

**DreamMachine** diagnoses where a small language model (≤2B) fails at multi-step arithmetic
reasoning. It generates synthetic training data that is correct by construction and aimed at that
weakness, fine-tunes the model with QLoRA on a consumer GPU, and measures whether *targeting* helps
beyond simply training on *harder* data.

- Research spec (novelty, maths, experiment design): [`docs/RESEARCH.md`](docs/RESEARCH.md)
- Component registry (what's built and what's confirmed): [`docs/COMPONENTS.md`](docs/COMPONENTS.md)
- How to run it on your GPU: [`docs/RUNNING.md`](docs/RUNNING.md)

## Layout
```
dreammachine/
  generator/   symbolic DAG problem generator (exact arithmetic, Q-matrix features)
  diagnosis/   answer extraction, CoT parsing, error taxonomy, LLTM fit, targeting
  data/        GSM8K / GSM-Symbolic loaders, data mixing
  models/      inference runner + prompts            (GPU)
  train/       QLoRA fine-tuning                     (GPU)
  experiments/ end-to-end pipeline CLI + configs
  store/       SQLite results store
  api/         FastAPI backend for the future frontend
tests/
docs/
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # core + tests (CPU only)
pip install -e ".[dev,api]"        # + backend API
pip install -e ".[dev,api,ml]"     # + GPU stack (torch, transformers, peft, bitsandbytes)
pytest -q
```

## Try the generator
```bash
python -m dreammachine.generator.preview --n 5 --steps 4 --digits 2 --distractors 1
```
