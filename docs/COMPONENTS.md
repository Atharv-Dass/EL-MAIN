# Component Registry

The record of which parts of DreamMachine are locked in.

| Status | Meaning |
|---|---|
| PROPOSED | Designed, not built |
| IMPLEMENTED | Built, tests pass, pushed. **Awaiting the team's verification.** |
| CONFIRMED | Verified by the team (Attel Bhavani Prasad). Changes need explicit approval. |

Rules: only a team member moves a component to CONFIRMED. After that, Claude does not change it
without asking first. Every component names the command that verifies it.

| # | Component | Path | Status | Verify with | Notes |
|---|---|---|---|---|---|
| C1 | Computation DAG (exact rationals) | `dreammachine/generator/dag.py` | IMPLEMENTED | `pytest tests/test_generator.py -k dag` | |
| C2 | Q-matrix features (steps, magnitude, ops, carries, distractors, merge) | `dreammachine/generator/features.py` | IMPLEMENTED | `pytest tests/test_generator.py -k "carr or borrow"` | Carries checked against brute force |
| C3 | Surface families (8 train, 2 held out) | `dreammachine/generator/families.py` | IMPLEMENTED | Read 20 samples (see below) | Needs a human read for naturalness |
| C4 | Constrained sampler + GSM8K-style solutions | `dreammachine/generator/sampler.py` | IMPLEMENTED | `pytest tests/test_generator.py` | 10k-item property test |
| C5 | Answer extraction | `dreammachine/diagnosis/extract.py` | PROPOSED | | |
| C6 | CoT equation parser | `dreammachine/diagnosis/cot_parse.py` | PROPOSED | | |
| C7 | Reference traces + alignment | `dreammachine/diagnosis/align.py` | PROPOSED | | |
| C8 | Error taxonomy classifier | `dreammachine/diagnosis/taxonomy.py` | PROPOSED | | |
| C9 | LLTM fit + bootstrap | `dreammachine/diagnosis/lltm.py` | PROPOSED | | |
| C10 | Weakness targeting + difficulty matching | `dreammachine/diagnosis/targeting.py` | PROPOSED | | |
| C11 | Human-agreement tools (κ, stratified sample) | `dreammachine/diagnosis/agreement.py` | PROPOSED | | |
| C12 | Dataset loaders + data mixer | `dreammachine/data/` | PROPOSED | | |
| C13 | Model runner + prompts | `dreammachine/models/` | PROPOSED | | GPU |
| C14 | QLoRA trainer | `dreammachine/train/` | PROPOSED | | GPU |
| C15 | Experiment CLI + configs | `dreammachine/experiments/` | PROPOSED | | |
| C16 | Results store | `dreammachine/store/` | PROPOSED | | |
| C17 | FastAPI backend | `dreammachine/api/` | PROPOSED | | |

## How to verify C3 (the manual read)
```bash
python -m dreammachine.generator.preview --n 20 --steps 3 --digits 2 --distractors 1
python -m dreammachine.generator.preview --n 10 --split heldout --merge --steps 5
```
Check three things: every sentence has exactly one arithmetic meaning, the question asks for the
DAG root, and the held-out families read differently from the train families.
