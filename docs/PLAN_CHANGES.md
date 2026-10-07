# PLAN_CHANGES.md — every change to PLAN.md, in plain words

One line per change, newest first. **Rule (asked by the user, 2026-10-07):** whoever changes `PLAN.md` adds a line
here in the same commit. The full text of any old version: `git show <commit>:PLAN.md`; all changes:
`git log -p PLAN.md`.

| Date | Commit | What changed in PLAN.md | Why |
|---|---|---|---|
| 2026-10-07 | `6960704` | D8 decided (headline = targeted − matched control on GSM8K + GSM-Symbolic); D14 decided (self-distillation not in the paper); new **D15** (smaller evaluations: 1,392 before / 770 after training, GSM-Symbolic p1/p2 dropped, no 1:1 ratio → 15 trainings per model, merged adapters for evaluation); new **D16** (a paper run reuses an earlier baseline + diagnosis of the same model); open questions 1, 7, 9 resolved, 2 and 8 updated | The user's decisions of 2026-10-07, to cut GPU time from ~8.8 to ~4 days and settle the GSM8K question |
| 2026-10-06 | `0f5d63f` | Question 7: result of self-distillation with up to 8 tries (−0.08 on GSM8K, targeted gain gone) and the recommendation | Dev-set test (option a2) finished |
| 2026-10-06 | `d25e54a` | Question 7: result of longer code-written solutions (−0.28, the model loops) | Dev-set test (option b) finished |
| 2026-10-06 | `4b36fae` | Question 7: result of self-distillation with 2 tries (loss halved, gain halved) | Dev-set test (option a) finished |
| 2026-10-06 | `9336744` | New **D14**: self-distillation of the training solutions, approved for a dev-set test | The user chose option (a) to fix the GSM8K drop |
| 2026-10-06 | `c3633f5` | Question 7: dev tuning done — learning rate, epochs and ratio do not fix the GSM8K drop; the cause is the short solution style | Overnight tuning results |
| 2026-10-05 | `8c67121` | New open questions 7 (GSM8K drop after training), 8 (GSM-Symbolic error labels), 9 (KS statistic or p-value) | Found in the first real pipeline run (B8) |
| 2026-10-05 | `4537764` | **D10** decided: `qwen_boxed` is the project prompt | Best accuracy in the prompt comparison; chosen by the user |
| 2026-10-05 | `2c875ef` | New **D13** (proposed): Llama-3.2-1B as an optional third model; Qwen2.5 stays out of the paper | The user delegated the model choice |
| 2026-10-05 | `adafde3` | New **D12**: two research models, Qwen3-0.6B and Qwen3-1.7B; question 3 resolved | Decided by the user (effect of model size) |
| 2026-10-04 | `0b67689` | Question 4 resolved: the exact `v2_700` prompt text was received (stored in `CLAUDE.md`) | The friend's prompt arrived |
| 2026-10-04 | `d83bccb` | Plan version 1.1 after an outside review (cooperative cancel on Windows, mock mode for the frontend, disk checks, timing, provenance) | Review feedback |
| 2026-10-04 | `33fe299` | First version of the Phase 2 plan | Start of Phase 2 |
