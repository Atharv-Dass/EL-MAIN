# weakness.md — what each model is weak at (in plain words)

One section per model, filled in after that model's **diagnosis** (paper settings: 864 generated problems × 3
answers each). Numbers come from the files named in each section; how they are saved is explained in
`docs/WORK.md` §5.4. All numbers here are **GPU** results.

How to read the tables:
- **Accuracy** = how often the untrained model got that kind of problem right.
- **Effect** = how much that property hurts, after separating it from the others (the LLTM statistics).
  Bigger = hurts more. "Significant" = the 95% confidence interval does not include zero, so it is not chance.
- Raw accuracies mix properties together (a problem with 3 divisions also has many steps); the effect column is
  the cleaned-up comparison and is what decides the target.

---

## Qwen3-0.6B — diagnosed 2026-10-07 (paper settings)

Files: `runs/pipelines/04f319d1cca7/diagnosis.json` (reused by the running paper job `80b37628b193`),
`target.json`; every answer in `runs/dreammachine.db`, run `32bcbd16e4`.

**Overall:** right on 36% of the 864 diagnosis problems. Most wrong answers were a wrong plan (65% —
it set the problem up wrongly), then arithmetic slips (18%).

### Main weakness: long problems (many steps) → this is the training target
The more calculation steps a problem needs, the worse the model does:

| Steps in the problem | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|
| Accuracy | 73% | 53% | 39% | 21% | 14% | 14% |

Every extra step roughly **halves the odds** of a right answer. The model handles short problems well but falls
apart after about 4 steps. **Target: problems with more than 4 steps** — the targeted arm trains on these.

### Other weaknesses (all significant, smaller in total)

| Rank | Weakness (plain words) | Effect | What the model does |
|---|---|---|---|
| 1 | **Many steps** | 2.99 | 73% right at 2 steps → 14% at 6–7 steps (the target) |
| 2 | **Division** | 1.20 | One division cuts the odds of a right answer to about a third (49% with no division, 20% with one) |
| 3 | **Big numbers** | 0.76 | 45% right when all numbers are below 100, 29% when some reach 1,000+ |
| 4 | **Multiplication** | 0.72 | Each multiplication lowers the odds by about 30% (49% with none, 27% with one) |
| 5 | **Distracting information** (numbers in the text that are not needed) | 0.34 | 43% with no distractor, 32% with one or two |

### Not a weakness
- **Carrying / borrowing digits** (e.g. 47 + 38): no effect after the others are separated (effect −0.23, not
  significant). Raw accuracy only drops for problems with very many carries, which are also the long problems.
- **Combining two separate calculations** ("merge"): could not be measured — the diagnosis problems did not vary it.

### Did training fix it? (paper run `80b37628b193`, finished 2026-10-09, mean of 3 seeds)
Accuracy on **long problems (more than 4 steps)** in probe stories the model never trained on, before → after:

| Training data | Long problems (> 4 steps) | Short problems (≤ 4 steps) |
|---|---|---|
| Untrained model | 18% | 52% |
| Real GSM8K problems only | 9% (worse) | 41% |
| **Aimed at long problems (targeted, 3:1)** | **54%** | 66% |
| Aimed at long problems, more of it (targeted, 9:1) | 66% | 75% |
| Just as hard, other reason (matched control) | 65% | 87% |
| Generated problems, not aimed (untargeted) | 71% | 82% |

In plain words:
- **Long problems got much easier for every kind of generated training data** (18% → 54-71%).
- **Aiming at "many steps" did not beat the other generated data**, even on long problems. It did make the
  *number of steps* matter less than any other training did (the steps effect shrank by about 1.0 vs ~0 for the
  others), but it taught less about **multiplication and division**, which long problems also contain. The other
  arms improved those a lot, and that counted for more.
- **On real test problems (GSM8K, GSM-Symbolic) all trained models got worse by about 16-24 points** — the
  known cost of the short training-solution style (`docs/RUNNING.md` §10-13) — and targeted was no different from
  the matched control there.

Full numbers: `docs/RUNNING.md` §14.

---

## Qwen3-1.7B — diagnosed 2026-10-09 (paper settings)

Files: `runs/pipelines/409305f3e2f2/diagnosis.json`, `target.json`; every answer in `runs/dreammachine.db`, run
`81fe9ad019b0`.

**Overall:** right on 64% of the 864 diagnosis problems (0.6B: 36%). Most wrong answers were a wrong plan (76%),
then arithmetic slips (11%).

### Main weakness: long problems (many steps) → the training target, as for 0.6B

| Steps in the problem | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|
| Accuracy (1.7B) | 89% | 78% | 72% | 63% | 44% | 37% |
| Accuracy (0.6B, for comparison) | 73% | 53% | 39% | 21% | 14% | 14% |

Each extra step lowers the odds of a right answer by about 30% (0.6B: about half). The bigger model holds up much
longer but still breaks down on 6-7 step problems. **Target: problems with more than 4 steps.**

### Other weaknesses

| Rank | Weakness (plain words) | Effect (0.6B) | What the model does |
|---|---|---|---|
| 1 | **Many steps** | 1.87 (2.99) | 89% right at 2 steps → 37% at 7 steps (the target) |
| 2 | **Division** | 1.10 (1.20) | One division cuts the odds to about a third (79% with none, 53% with one) — as bad as for 0.6B |
| 3 | **Multiplication** | 0.53 (0.72) | 75% with none, 62% with one, 46% with two |
| 4 | **Distracting information** | 0.33 (0.34) | 70% with no distractor, 58% with two — same as 0.6B |

**Not a weakness (for 1.7B):** big numbers (no longer significant; it was for 0.6B), carrying/borrowing digits.
"Merge" could not be measured (the problems did not vary it).

### Did training fix it? (paper run `409305f3e2f2`, finished 2026-10-10, mean of 3 seeds)
Accuracy on **long problems (more than 4 steps)** in probe stories the model never trained on:

| Training data | Long problems (> 4 steps) | Short problems (≤ 4 steps) |
|---|---|---|
| Untrained model | 41% | 73% |
| Real GSM8K problems only | 34% (worse) | 66% |
| **Aimed at long problems (targeted, 3:1)** | **86%** | 88% |
| Aimed at long problems, more of it (targeted, 9:1) | 87% | 92% |
| Just as hard, other reason (matched control) | 87% | 97% |
| Generated problems, not aimed (untargeted) | 89% | 97% |

In plain words:
- **Long problems became easy for every kind of generated training data** (41% → 86-89%), so the arms can hardly
  be told apart here — the 1.7B model is near the top of what these problems can show.
- **Aiming at "many steps" again did not beat the other generated data.**
- **On real test problems all trained models got worse by about 14-19 points** (0.6B: 16-24), and targeted was no
  different from the matched control there.

Full numbers: `docs/RUNNING.md` §15.
