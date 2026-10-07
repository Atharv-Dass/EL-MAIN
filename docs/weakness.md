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

### Did training fix it?
To be filled in when the paper run finishes (around Oct 8 evening): accuracy on long problems (> 4 steps) before
vs after training, for the targeted arm and the matched control.

---

## Qwen3-1.7B — not diagnosed yet

Its paper run (job `409305f3e2f2`) starts after the 0.6B run, around **Oct 8 evening**; the diagnosis takes
about 3 hours. This section will then be filled in the same way. It may be weak at something different from
0.6B — that comparison is one of the things the paper looks at.
