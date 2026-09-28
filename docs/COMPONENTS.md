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
| C5 | Answer extraction (####, \\boxed, 'answer is', last number; strips <think>) | `dreammachine/diagnosis/extract.py` | IMPLEMENTED | `pytest tests/test_diagnosis.py -k extract` | |
| C6 | CoT equation parser + exact re-check (safe AST eval, chained eqs, rounding) | `dreammachine/diagnosis/cot_parse.py` | IMPLEMENTED | `pytest tests/test_diagnosis.py -k parse` | |
| C7 | Reference traces (DAG or GSM8K `<<>>` annotations) + alignment | `dreammachine/diagnosis/align.py` | IMPLEMENTED | `pytest tests/test_diagnosis.py -k gsm8k` | Word-numbers ('half') invisible on real GSM8K |
| C8 | Error taxonomy: FORMAT / SLIP / DISTRACTOR / PLAN / UNVERIFIABLE | `dreammachine/diagnosis/taxonomy.py` | IMPLEMENTED | `pytest tests/test_diagnosis.py` | Replaces the Phase-1 taxonomy; needs team sign-off |
| C9 | LLTM fit (Newton/IRLS, ridge, Wald + bootstrap CIs) | `dreammachine/diagnosis/lltm.py` | IMPLEMENTED | `pytest tests/test_lltm_targeting.py -k "recovery or coverage"` | 95% CI coverage verified by simulation |
| C10 | Weakness ranking, targeted arm, difficulty-matched control, untargeted arm | `dreammachine/diagnosis/targeting.py` | IMPLEMENTED | `pytest tests/test_lltm_targeting.py -k arms` | Core paper contribution |
| C11 | Human-agreement tools (κ, confusion, stratified sample) | `dreammachine/diagnosis/agreement.py` | IMPLEMENTED | `pytest tests/test_diagnosis.py -k "kappa or stratified"` | |
| C4b | Factorial probe sets, natural prior, selection pools | `dreammachine/generator/probes.py` | IMPLEMENTED | `pytest tests/test_lltm_targeting.py` | Natural prior weights are a guess at GSM8K; calibrate on real data |
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
