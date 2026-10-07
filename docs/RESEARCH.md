# DreamMachine — Research Specification

> Working title: **Is it the targeting or the difficulty? Construct-valid weakness diagnosis for
> synthetic fine-tuning of small language models**

## 1. Positioning against prior work

| Work | What it does | Gap we exploit |
|---|---|---|
| GSM8K (Cobbe+ 2021) | Benchmark with calculator-annotated solutions (`<<a*b=c>>`) | The annotations give a free reference trace. We use them for step-level alignment. |
| GSM-Symbolic (Mirzadeh+ 2024) | Templated GSM8K variants that vary numbers and clauses | An evaluation tool only. It is never used to *diagnose* or *train*. |
| CDS (Zhao+ 2025) | Cognitive-diagnosis-guided synthesis | Knowledge points are tagged by an LLM. |
| CRAFT (Gupta+ 2026, 2607.16122) | Rubric criteria → capability tree → targeted SFT | Needs rubric datasets and an LLM judge. Finance/legal domains. |
| KITE (Luo+ 2026, 2607.17043) | DINA diagnosis → noisy generation → KBU boundary filter (GSM8K, MATH) | The Q-matrix is **LLM-tagged, not validated by experts, with fixed slip/guess priors**. There is no difficulty-matched control. |
| O'Grady & Ramlan (2026, 2607.18266) | GPT-5-mini synthetic arithmetic data, LoRA on Qwen3-0.6B/1.7B on consumer hardware | Untargeted. It does not isolate whether targeting helps. |

**Our claim:** every targeted-synthesis paper above compares "targeted" data against random data. But
targeted data is *harder* than random data by construction, so the reported gains confound
**targeting** with **difficulty**. We remove that confound. Our generator knows each item's skill
profile exactly (the Q-matrix is known by construction), so we can build a
**difficulty-matched off-target control**.

## 2. Contributions
1. **Construct-valid diagnosis.** A symbolic generator whose items carry their computation DAG. The
   Q-matrix (steps, magnitude, op counts, carries, distractors, merge structure) is exact, not tagged.
2. **LLTM weakness attribution.** Factorial probing plus a Linear Logistic Test Model. This gives a
   per-factor difficulty with confidence intervals, and the factorial design makes it a causal
   attribution.
3. **Annotation-free step-level error classification** (arithmetic slip / plan error / distractor use /
   format), validated against human labels with Cohen's κ.
4. **A 4-arm controlled study** at equal token budget: real-only, untargeted synthetic,
   difficulty-matched off-target synthetic, targeted synthetic. It separates the effect of targeting
   from the effect of difficulty on ≤2B models trained with QLoRA on an 8 GB consumer GPU.

A null result for (4), meaning targeted = difficulty-matched > untargeted, is a finding worth
publishing: it would mean prior gains were a difficulty effect.

## 3. Mathematics

### 3.1 Items and the Q-matrix
Item *i* is a DAG *G_i* with leaves (numbers stated in the question) and op nodes
(+, −, ×, ÷), evaluated in exact rationals. Its feature vector *q_i ∈ ℝ^K* is computed from *G_i*:

| k | feature | definition |
|---|---|---|
| 1 | steps | number of op nodes |
| 2 | log10_max | log₁₀ max over all node values |
| 3 | n_mul | count of × nodes |
| 4 | n_div | count of ÷ nodes |
| 5 | n_carry | Σ column carries (+) and borrows (−) |
| 6 | n_distractors | numeric sentences not in *G_i* |
| 7 | has_merge | 1 if the DAG is a tree of two chains |

### 3.2 LLTM (Fischer, 1973)
Model *m* answers item *i* correctly *s_i* times out of *n_i* samples:

  logit P(correct_i) = θ − Σ_k q_ik η_k

θ is the model's ability, and η_k is the log-odds cost of one unit of factor *k*.
We estimate (θ, η) by penalised maximum likelihood (a ridge λ on η only, for stability):

  ℓ(θ, η) = Σ_i [ s_i z_i − n_i log(1 + e^{z_i}) ] − (λ/2)‖η‖²,  z_i = θ − q_iᵀη

Standard errors come from the observed Fisher information *Xᵀ W X* (with W = diag(n_i p_i(1−p_i))
and X = [1, −q]). Intervals come from a bootstrap over items.

### 3.3 Weakness selection
The raw η_k values are not comparable, because factors have different ranges. We rank factors by
**range-normalised effect**

  E_k = η_k · (Q_k^{90} − Q_k^{10})

where Q_k^p is the p-th percentile of factor *k* on the natural (GSM8K-like) item distribution.
The target factor is k* = argmax E_k, taken over factors whose lower CI bound on η_k is > 0.

### 3.4 Information-optimal difficulty
The Fisher information a Bernoulli response carries about ability is p(1 − p). It peaks at
p = 0.5, which is the zone of proximal development. Targeted items are selected from a large
generated pool with (i) q_{ik*} above the natural median and (ii) predicted p̂_i ∈ [0.3, 0.7].
They are weighted by p̂_i(1 − p̂_i).

### 3.5 Difficulty-matched off-target control
From the same pool, take the items with q_{ik*} ≤ natural median. For each targeted item, greedily
match (without replacement) the unused off-target item with the nearest predicted logit ẑ.
Matching quality is reported as the mean |Δẑ| and a two-sample KS statistic on p̂. Both arms then
have the same predicted-difficulty distribution. They differ only in *which* factor makes items
hard.

### 3.6 Step-level error classification
Each response is classified in this order:
1. No extractable answer → FORMAT_ERROR.
2. Answer = gold → CORRECT.
3. Parse every `a op b = c` in the response. The first equation whose left side ≠ right side
   → ARITHMETIC_SLIP.
4. The last equation equals gold but the extracted answer differs → FORMAT_ERROR.
5. No equations → UNVERIFIABLE.
6. An operand equals a distractor number (and no relevant value) → DISTRACTOR_USE.
7. Otherwise → PLAN_ERROR. The first reference step whose value never appears is the divergence point.

The reference trace comes from the DAG (synthetic items) or from calculator annotations (GSM8K).

### 3.7 Primary outcome and test
The outcome is the change in accuracy on the target-factor slice of held-out probes, *and* the
change in η_{k*} (a re-fitted LLTM after training). Arms are compared over 3 seeds with a paired
bootstrap over items. We also report the GSM8K-test and GSM-Symbolic accuracy change to check
for regression.

## 4. Experimental design
- **Models (8 GB VRAM):** screen Qwen3-0.6B, Qwen3-1.7B (non-thinking), Llama-3.2-1B-Instruct and
  Gemma-3-1B-it. Choose the model with the largest usable headroom.
- **Probe set:** full factorial over steps {2..7} × digits {1..4} × distractors {0,1,2} × op
  profile {add/sub, mixed, mul/div-heavy}, several items per cell. Only train-split families are used.
- **Arms (equal token budget, 3 seeds):** A real-only (GSM8K train) · B untargeted synthetic ·
  C difficulty-matched off-target · D targeted. Ratios synthetic:real ∈ {1:1, 3:1, 9:1} are run
  on arm D.
- **Eval:** GSM8K test (1319), GSM-Symbolic (+P1/P2), a probe set from *held-out families*, and a
  target-factor slice.
- **QLoRA:** NF4, r=16, α=32, dropout 0.05, seq 512, gradient checkpointing.

## 5. Threats to validity
- **Template overfitting.** Mitigated with held-out families that use different phrasing. We report the
  train-family vs held-out gap.
- **GSM-Symbolic overlap.** Our templates are written independently. We will n-gram-check the
  overlap and report it.
- **Contamination of GSM8K in pre-training.** This affects all arms equally. The conclusions rest on
  the *differences between arms*.
- **Naturalness of the synthetic text.** It is lower than human-written text. Arm A plus the mixing
  ratios measure the cost.
