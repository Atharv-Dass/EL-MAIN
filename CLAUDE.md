# CLAUDE.md — project memory for DreamMachine (EL-MAIN)

Read this first in every session, then `PLAN.md` (what to build, in what order) and
`docs/API_CONTRACT.md` (what the frontend sees). Branch: `claude/wonderful-hypatia-pqs6t9`.

## What this project is
DreamMachine finds where a small LM (≤ 2B) fails at multi-step arithmetic word problems, generates
training problems **by code** aimed at that weakness, fine-tunes with QLoRA on one RTX 4060 laptop (8 GB),
and re-tests. **Novelty: targeting vs difficulty** — the `matched_control` arm is just as hard as `targeted`
but hard for a different reason. Science: `docs/RESEARCH.md`. Status of parts: `docs/COMPONENTS.md`.

## Rules (from PLAN.md §0 — the full list is there)
- Order of truth: code on the branch > `PLAN.md` > `docs/API_CONTRACT.md` (law for the frontend) > `docs/RESEARCH.md`.
  If plan and code disagree: **stop and ask**.
- Do not invent function names, paths, dataset fields, model names, prompt texts or numbers. Open the file.
- **Backend only.** No frontend code (Manus AI builds it later from the contract).
- Do not edit `docs/RESEARCH.md` or any `CONFIRMED` component without the user's explicit OK.
  Claude may set `IMPLEMENTED`; only the team sets `CONFIRMED`.
- Label every result **CPU-verified** or **GPU-verified**. Never claim a GPU result that was not run.
- **Explore ≠ paper.** Explore runs are never paper evidence.
- Training data is made by the code generator. Do not reintroduce LLM-written training data.
- Ask before: any new dependency; downloads > ~1 GB; writing > ~5 GB. (GPU use was approved by the user
  on 2026-10-04 for smoke testing.) Never `git push` (the user pushes). No tokens/secrets in configs or commits.
- Stop at every milestone (PLAN.md §10): `pytest -q`, update `docs/COMPONENTS.md`, commit, print the §10.1
  report with the CONTRACT DIGEST.

## Windows conventions (native Windows, no WSL, is the main target; Linux must keep working)
- PowerShell commands in docs; `python -m ...` entry points; `pathlib`.
- `encoding="utf-8"` on **every** text `open` / `read_text` / `write_text` (enforced by `tests/test_windows.py`).
- No POSIX signals. Subprocesses via `sys.executable` with list arguments (the repo path has spaces).
- `Popen.terminate()` is a hard kill on Windows → cooperative cancel first (PLAN.md §7.3).
- `dataloader_num_workers=0`; no `torch.compile`, no Triton.
- 4-bit needs bitsandbytes with CUDA; `check_4bit_support` raises a clear error, never a silent fallback.
- The API binds to `127.0.0.1` only.

## Research models (PLAN.md D12)
Both `Qwen/Qwen3-0.6B` and `Qwen/Qwen3-1.7B` (decided 2026-10-05): model size is a factor. One paper run per model.
Measured settings and dev-slice accuracy: `docs/RUNNING.md` §6–8 (512 new tokens, batch 32; thinking mode off;
Windows spills VRAM into system RAM instead of OOM — keep batches inside 8 GB).

## Prompt versions (PLAN.md §8.5, D10)
- `dm_v1` (default) = `INSTRUCTION` in `dreammachine/models/prompts.py`, placed in the user turn.
- `v2_700` (friend's prompt; text received 2026-10-04; registered in B2 as `PROMPTS["v2_700"]`,
  byte-for-byte, used as the SYSTEM message; pinned by sha256 in `tests/test_prompts.py`). Exact text between the markers, no added spaces or newline:

  `>>>`Solve the math problem step by step. At the end, write a separate final line containing only the numeric answer, with no units, dollar sign, commas, bold formatting, or extra text. Example final line: 18.`<<<`

- **`qwen_boxed` = THE PROJECT PROMPT (D10, chosen by the user 2026-10-05)**: the Qwen3 model card's math prompt,
  after the question. Default in code and all configs. Training answers end with `\boxed{n}` (`format_completion`).
- One `prompt_version` per project, set at the top level of a config. Benchmarks and extractors: `docs/BENCHMARKS.md`.

## API contract
- **Contract version: `0.2.0`** — IMPLEMENTED (B7). `dreammachine/api/app.py` (`API_VERSION`), response models in
  `dreammachine/api/schemas.py`, `docs/openapi.json` from `python -m dreammachine.api.export_openapi`,
  guarded by `tests/test_api_contract.py`. The old unprefixed routes are gone.
- Base URL: `http://127.0.0.1:8000/api/v1`. Every route/shape change: update `docs/API_CONTRACT.md`, bump the
  version, add a changelog line, regenerate `docs/openapi.json`, update this list — same commit.
- Endpoints (contract 0.2.0):
  ```
  GET    /api/v1/health
  GET    /api/v1/system
  GET    /api/v1/meta
  GET    /api/v1/models
  GET    /api/v1/benchmarks
  GET    /api/v1/presets
  GET    /api/v1/components
  GET    /api/v1/queue
  POST   /api/v1/pipelines
  GET    /api/v1/pipelines
  GET    /api/v1/pipelines/{pipeline_id}
  POST   /api/v1/pipelines/{pipeline_id}/cancel
  POST   /api/v1/pipelines/{pipeline_id}/resume
  DELETE /api/v1/pipelines/{pipeline_id}
  GET    /api/v1/pipelines/{pipeline_id}/baseline
  GET    /api/v1/pipelines/{pipeline_id}/diagnosis
  GET    /api/v1/pipelines/{pipeline_id}/data
  GET    /api/v1/pipelines/{pipeline_id}/data/{arm_key}/samples
  GET    /api/v1/pipelines/{pipeline_id}/data/{arm_key}/download
  GET    /api/v1/pipelines/{pipeline_id}/training
  GET    /api/v1/pipelines/{pipeline_id}/results
  GET    /api/v1/pipelines/{pipeline_id}/report
  GET    /api/v1/pipelines/{pipeline_id}/log
  POST   /api/v1/benchmark-runs
  GET    /api/v1/benchmark-runs
  GET    /api/v1/benchmark-runs/{job_id}
  POST   /api/v1/benchmark-runs/{job_id}/cancel
  DELETE /api/v1/benchmark-runs/{job_id}
  GET    /api/v1/benchmark-runs/{job_id}/baseline
  GET    /api/v1/benchmark-runs/{job_id}/diagnosis
  GET    /api/v1/benchmark-runs/{job_id}/log
  GET    /api/v1/runs
  GET    /api/v1/runs/{run_id}
  GET    /api/v1/runs/{run_id}/responses
  GET    /api/v1/runs/{run_id}/artifacts/{key}
  GET    /api/v1/compare
  POST   /api/v1/tools/generate
  POST   /api/v1/tools/classify
  POST   /api/v1/tools/lltm-fit
  ```

## Pointers
`PLAN.md` · `docs/API_CONTRACT.md` · `docs/RESEARCH.md` (protected) · `docs/COMPONENTS.md` · `docs/RUNNING.md`
