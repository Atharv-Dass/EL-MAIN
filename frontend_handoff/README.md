# Frontend handoff — everything a frontend AI (Manus) needs

The frontend is built **only from the files in this folder**. Do not read or edit the Python code.

## Read in this order
1. **`API_CONTRACT.md`** — the law. Base URL, error envelope, enums, every endpoint, TypeScript types (§7),
   which screen uses which endpoint (§8), pipeline state machine (§9). Contract version **0.2.1**.
2. **`openapi.json`** — the same API in machine-readable form (generated from the backend). Use it to
   generate a typed client. If it and the contract disagree, the contract wins; tell the team.
3. **`weakness.md`** — plain-words description of what the diagnosis finds for each model and how to read the
   results. Use it for wording on the Diagnosis and Results screens.
4. **`BENCHMARKS.md`** — the named test sets (`gsm8k`, `gsm_symbolic:*`, probes) shown in dropdowns.

## What the product is
DreamMachine finds where a small language model (≤ 2B parameters) fails at multi-step arithmetic word problems,
generates training problems by code aimed at that weakness, fine-tunes the model, and re-tests. The research
question: does *targeting* the weakness help beyond just training on *harder* data? The `matched_control`
arm is as hard as `targeted` but hard for a different reason.

## Rules the frontend must follow
- **Mock mode is required** (`VITE_USE_MOCK=true` or the framework's equivalent). The cloud builder cannot reach
  the backend on `127.0.0.1`, so build and demo against fixtures made from the examples in the contract (§6),
  matching the §7 types exactly. Show a clear **"MOCK DATA"** badge on every screen while mock mode is on.
- Label every run's **mode**: explore runs are shown as "Explore — not paper evidence".
- Never invent endpoints, fields or enum values. Use the strings in contract §4 exactly.
- Show the `worker_not_running` warning when a create call returns 202 with it.
- Real data only appears when the frontend runs on the same laptop as the backend
  (`http://127.0.0.1:8000/api/v1`; start with the command in contract §2).

## Keeping this folder current
These files are **copies** of the sources in `docs/`. After any change to the API contract or
`docs/openapi.json`, run `powershell -File frontend_handoff\sync.ps1` and commit the folder in the same commit.
