# WORK.md — what has been done so far, and what comes next

Written 2026-10-07, in plain words. Details live in `PLAN.md`, `docs/RUNNING.md` (all measured numbers) and
`docs/COMPONENTS.md` (status of each part). Results marked **GPU** were run on the RTX 4060 laptop; **CPU** means
only the automatic tests were run. **Explore** runs are practice runs and are never used as paper evidence.

---

## 1. The project in one paragraph

DreamMachine takes a small language model (≤ 2B parameters), finds out **which kind of maths word problem it fails
at** (its "weakness"), writes new training problems **by code** aimed exactly at that weakness, fine-tunes the
model on them, and tests it again. The new idea: we compare that **targeted** training with a **matched control**
— training problems that are *just as hard* but hard for a *different* reason. If targeted wins, it is the
targeting that helps, not just "harder data".

---

## 2. What was built (milestones B0–B8, all done)

| Milestone | In simple words |
|---|---|
| **B0** | Made everything work on native Windows (file encodings, no WSL, clear error messages) and wrote `CLAUDE.md`, the project memory. |
| **B1** | First smoke test on the GPU: the model loads, answers questions, and 4-bit training (QLoRA) works on 8 GB. |
| **B2** | Prompt versions: `dm_v1`, your friend's `v2_700` (stored byte for byte), and later Qwen's own `qwen_boxed`. Answer extractors for each. |
| **B3** | A results database (SQLite) and a job table, so long runs survive crashes and can be resumed. |
| **B4** | The pipeline: preflight → baseline → diagnose → choose target → build data → train → evaluate → compare, one step at a time. |
| **B5** | A worker that runs queued jobs in the background, one step per process; cancel, resume, crash recovery, GPU temperature logging. |
| **B6** | The results maker: before/after per arm, arm vs arm, confidence intervals, regressions, `report.md`, "paper-eligible" check. |
| **B7** | The web API (`http://127.0.0.1:8000/api/v1`, 39 endpoints) exactly as `docs/API_CONTRACT.md` (version 0.2.1). The frontend (built later by Manus) talks to this. |
| **B8** | The first real end-to-end run on the GPU through the API (Qwen3-0.6B, explore). It worked; it also found bugs, which were fixed (see §4). |

About 200 automatic tests check all of this (**CPU**). Claude never pushes: you push yourself (the branch is pushed and in sync as of 2026-10-18).

---

## 3. Decisions made along the way

| Topic | Decision | Why |
|---|---|---|
| Models | **Qwen3-0.6B and Qwen3-1.7B** | Same family, two sizes: we can see how model size changes the result. |
| Prompt | **`qwen_boxed`** (the model card's maths prompt; answer in `\boxed{}`) | Best accuracy on 200 dev problems: 0.705 (0.6B) / 0.860 (1.7B), vs `dm_v1` 0.565 / 0.790 and `v2_700` 0.360 / 0.775 (**GPU**). |
| Answer length | 512 new tokens, batch 32, thinking mode off | 256 tokens cut off 17.5% of answers; batch 32 is 1.8x faster than 16; thinking mode was slower without a clear gain (**GPU**). |
| Training solutions | **Written by our code** (not by a language model) | See §4.3 — the alternatives cost the targeted gain. |
| Headline result | **targeted − matched control** on GSM8K test and GSM-Symbolic | Confirmed by you (D8). |
| Match quality | judged by the **KS statistic** | With thousands of problems the p-value is almost always small. |
| GSM-Symbolic error types | **not reported** in the paper for now | The error labeller is unreliable there (35% "distractor" labels vs 0% on GSM8K). Accuracy is fine and is reported. |

---

## 4. Experiments so far (all Qwen3-0.6B, all **explore**, all **GPU**)

### 4.1 Settings and prompts (Oct 4–5)
Tried prompts, answer lengths, thinking mode, both models — only on 200 problems taken from GSM8K's **training**
set (the "dev set"). We never tune on the test set, otherwise the final test numbers would be too optimistic.

### 4.2 First full pipeline, B8 (Oct 5)
Whole pipeline in ~2.6 h. Bugs found and fixed: the matched control ran out of problems (pool now grows
automatically); GSM-Symbolic's first 50 problems were all the same template (now spread across templates);
Windows silently moves GPU memory into normal RAM and becomes very slow (memory is now capped at 92%).

### 4.3 The GSM8K problem (Oct 5–6)
After training, the model got **worse on GSM8K** (about −0.20 to −0.24), even though it got **much better on the
targeted weakness** (+0.19 to +0.26 on held-out probes; the weak slice went from 0.10 to ~0.45). We tested why:

| What we tried | GSM8K dev change | Targeted gain (probes) | Verdict |
|---|---|---|---|
| Learning rate, epochs, mix ratio (5 runs) | −0.20 to −0.24 | +0.19 to +0.25 | Not a settings problem |
| Train on real GSM8K only | −0.225 | −0.14 | Not caused by our synthetic data |
| **Model writes its own solutions**, 2 tries (self-distillation) | −0.115 | +0.10 | Half the loss, half the gain |
| **Code writes longer, step-by-step solutions** | −0.28 | +0.19 | Worse: the model started repeating itself |
| Model writes its own solutions, up to 8 tries | −0.08 | +0.08 (not significant) | Least loss, but the gain is gone |

**Conclusion:** the model copies the short style of the training solutions. Making it keep its own style also
removes what it learns about the weakness. **Decision:** keep the code-written solutions for the paper, state the
GSM8K drop openly as a cost, and report self-distillation as an extra experiment. The drop is the same for every
arm, so the targeted-vs-control comparison stays fair.

### 4.4 Faster testing (Oct 7)
Merging the trained add-on (LoRA adapter) into the model before testing: **1.84x faster, same accuracy**.

---

## 5. The paper runs (done)

### 5.1 What each paper run does (`configs/main.yaml`, locked)
- **5 training setups × 3 random seeds = 15 trainings per model**, each with 600,000 tokens of training text:
  - `real_only` — real GSM8K problems only (the plain baseline)
  - `untargeted` — our generated problems, not aimed at anything (3 parts generated : 1 part real)
  - `matched_control` — just as hard as targeted, but for a different reason (3 : 1)
  - `targeted` — aimed at the weakness (3 : 1 and 9 : 1). The 1 : 1 version was removed to save time.
- **Before training** (baseline): the untrained model answers GSM8K test, GSM-Symbolic and probe problems.
- **After each training**: 770 problems — **500 real test problems** (GSM8K 300 + GSM-Symbolic 200) and
  **270 generated probes** (to see if the weakness itself improved). All are also in the baseline, so every
  before/after comparison uses the same problems.
- **Diagnosis**: 864 generated problems × 3 answers each → which property makes the model fail (0.6B: problems
  with more than 4 steps). This decides the training data, so it is needed once per model.

### 5.2 Time saved today
- Evaluation shrank from 3,683 to 770 problems (about 2.5 h → about 30 min each, on 0.6B).
- The 1 : 1 ratio is gone (18 → 15 trainings per model).
- **0.6B reuses its earlier baseline and diagnosis** (they already cover every problem needed), so it went
  straight to training. 1.7B has no earlier baseline on the test sets, so it runs its own once.
- Total GPU time for both models went from about 8.8 days to **about 4 days**.

### 5.3 Status (updated 2026-10-10 04:30) — both paper runs are done
| Job | Model | State | Finish |
|---|---|---|---|
| `80b37628b193` | Qwen3-0.6B | **done**, paper-eligible | finished 2026-10-09 01:19 |
| `409305f3e2f2` | Qwen3-1.7B | **done**, paper-eligible | finished 2026-10-10 04:04 |

The 0.6B run was stopped once (2026-10-08 ~02:30) because the laptop ran low on memory; it was resumed from its
last checkpoint, nothing finished was lost, ~9 hours were lost. Since then the worker runs in its own minimized
window ("DreamMachine paper worker"), outside Claude Code. Results: `runs/pipelines/<job>/results.json` and `report.md`. Worker log: `runs/paper_worker.log`.
Doc edits during a run are committed at once (a run checks for a clean repo when it starts).

### 5.3a First paper result: Qwen3-0.6B (in plain words; numbers in `docs/RUNNING.md` §14)
- **Headline (targeted vs matched control on real test problems): no difference.** GSM8K −1.2 points
  (CI −3.9 to +1.4), GSM-Symbolic −0.2 points (CI −3.7 to +3.2).
- **Every trained model got worse on real test problems** (about −16 to −24 points) — the expected style cost.
- **On generated problems, all generated training data helped a lot** (+25 to +58 points), but **targeted helped
  less than matched control and untargeted** — even on the long problems it was aimed at.
- **Targeting did shrink the "many steps" weakness the most**, but the other data fixed multiplication and division
  more, which long problems also need. Details in `docs/weakness.md`.
- What this means for the paper is a team decision (PLAN.md open question 10).

### 5.3b Second paper result: Qwen3-1.7B (numbers in `docs/RUNNING.md` §15)
- **Same headline: targeted vs matched control on real test problems — no difference** (GSM8K +1.1 points,
  CI −1.7 to +3.9; GSM-Symbolic +1.5, CI −2.0 to +5.0).
- Real-test cost a bit smaller than for 0.6B (about −14 to −19 points).
- The diagnosis found the same main weakness (many steps), but milder; training on any generated data made long
  problems easy (41% → 86-89%), so the arms are almost equal on the probes.

### 5.4 How the diagnosis (and everything else) is saved on this laptop
Yes — when a model is weak at, for example, multi-step problems, that is written down, in three places:

1. **The job's own folder** `runs/pipelines/<job id>/` (one folder per run):

   | File / folder | What is in it |
   |---|---|
   | `diagnosis.json` | Every weakness found: its name (e.g. `steps`), how much it hurts (`eta`, `effect`), the 95% confidence interval, whether it is significant, and the ids of the 864 problems used |
   | `target.json` | The one weakness chosen for training, e.g. `{"feature": "steps", "threshold": 4.0}` = "problems with more than 4 steps" |
   | `annotate_errors.csv` | A sample of wrong answers for a human to label (checks the automatic error labels) |
   | `data/` | The training data built from the target, one file per arm and seed |
   | `adapters/` | The trained models |
   | `evals/` | Accuracy after each training (and the baseline) |
   | `logs/` | Step logs and GPU temperatures |

2. **The results database** `runs/dreammachine.db`: every single answer of every run (question, model answer,
   right/wrong, error type, the problem's properties). The diagnosis is a run of kind `diagnose`.

3. **`docs/weakness.md`**: the same weaknesses written for humans (accuracy tables, plain words), one section per
   model. Updated by Claude after each model's diagnosis and again after training.

Where to look now: Qwen3-0.6B → `runs/pipelines/04f319d1cca7/` (diagnosis made there, reused by
`80b37628b193`); Qwen3-1.7B → `runs/pipelines/409305f3e2f2/`.
**`runs/` is not in git** (too big; it stays only on this laptop) — copy it somewhere safe after the runs.

---

## 6. Next steps

**Claude:**
- Done: both paper runs finished and written up — 0.6B on 2026-10-09 (`docs/RUNNING.md` §14), 1.7B on
  2026-10-10 (§15), both in `docs/weakness.md` (diagnosis and "did training fix it?"). Nothing is left for Claude
  until the team decides how to read the result (PLAN.md question 10). A changed targeting rule would be a new
  protocol and needs new runs.

**You / the team:**
1. Back up `runs/` (not in git; it exists only on this laptop).
2. Decide PLAN.md open question 10 (paper framing; whether to target several weaknesses).
3. Done: the branch `claude/wonderful-hypatia-pqs6t9` is pushed and in sync with origin.
4. Label the human error sample `runs/pipelines/5a584e16390b/annotate_errors.csv` (checks the automatic error labels).
5. Check the citation "O'Grady & Ramlan (2026)" in `docs/RESEARCH.md` — it could not be found online.
6. Decide the KS-statistic threshold for a "bad match" (today it only warns).
7. Before going public: LICENSE, `CITATION.cff`, automatic tests on GitHub (CI).
8. Frontend: Manus builds it from `docs/API_CONTRACT.md` and `docs/openapi.json`.
9. Optional: add Llama-3.2-1B as a third model (needs the licence on Hugging Face and `huggingface-cli login`).

---

## 7. Words used here

| Word | Meaning |
|---|---|
| Arm | One kind of training data (real_only, untargeted, matched_control, targeted). |
| Ratio 3 : 1 | 3 parts generated problems to 1 part real GSM8K problems, counted in tokens. |
| Seed | The random starting point of a training run; 3 seeds show how much results vary by chance. |
| Baseline | The untrained model's answers — the "before". |
| Probes | Problems generated by our code with known properties, used to measure the weakness. |
| Diagnosis / LLTM | The statistics that find which property (steps, digits, carries, …) makes the model fail. |
| Dev set | 200 GSM8K *training* problems kept aside for choosing settings, so the test set stays untouched. |
| QLoRA / adapter | Cheap fine-tuning: a small add-on is trained instead of the whole model. |
| Explore vs paper | Explore = practice, never evidence. Paper = the fixed protocol, the real evidence. |
