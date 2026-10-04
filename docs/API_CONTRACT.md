# DreamMachine — API Contract (backend ⇄ frontend)

**Contract version: `0.1.0`** · Status: **DRAFT** (designed 2026-10-04, not implemented yet; becomes
"implemented" at backend milestone B7 in `PLAN.md`).

This file is the **only** thing the frontend may rely on. The frontend (to be built by Manus AI) talks to the
backend **only over HTTP** using the endpoints below. It must not import Python code, read the SQLite file,
or read files under `runs/`.

A machine-readable copy is generated from the running app: `docs/openapi.json`
(`python -m dreammachine.api.export_openapi`). If this file and `openapi.json` ever disagree, that is a bug —
report it; do not guess.

---

## 1. Repository layout (where things live)

```
EL-MAIN/
  dreammachine/          <- backend Python package (do NOT edit from the frontend)
    api/                 <- FastAPI app (this contract)
    jobs/                <- queue, worker, pipeline steps
    benchmarks/          <- benchmark registry
    experiments/ diagnosis/ generator/ data/ models/ train/ store/
  configs/               <- YAML presets (explore.yaml, main.yaml = paper, models.yaml)
  docs/
    API_CONTRACT.md      <- this file
    openapi.json         <- generated from the app
  runs/                  <- results + SQLite DB (backend only, git-ignored)
  tests/
  frontend/              <- THE FRONTEND GOES HERE (separate project, its own package.json)
```

Frontend rules:
- Put all frontend code in `frontend/`. Do not create or change files outside it.
- Read the API base URL from an environment variable: `VITE_API_BASE_URL`
  (or the framework's equivalent), default `http://127.0.0.1:8000/api/v1`.
- Dev server on port `5173` or `3000` (both are allowed by CORS by default).

## 2. Running the backend (Windows PowerShell)

```powershell
cd C:\dev\EL-MAIN
.\.venv\Scripts\Activate.ps1
python -m dreammachine.serve          # API on http://127.0.0.1:8000 + the GPU worker
# or separately:
uvicorn dreammachine.api.app:app --host 127.0.0.1 --port 8000
python -m dreammachine.jobs.worker
```
Interactive docs (when running): `http://127.0.0.1:8000/docs`.

---

## 3. Conventions

| Topic | Rule |
|---|---|
| Base URL | `http://127.0.0.1:8000/api/v1` |
| Format | JSON, UTF-8. Request bodies `Content-Type: application/json`. One exception: `GET .../report` returns `text/markdown`. |
| Field names | `snake_case` |
| IDs | strings, 12 lowercase hex characters (e.g. `"a1b2c3d4e5f6"`). Job ids and run ids are different things. |
| Times | ISO 8601 UTC strings with `Z`, e.g. `"2026-10-05T10:15:00Z"`, or `null`. |
| Accuracy / shares | floats from 0 to 1. Differences (`mean_diff`, CIs) are also in 0–1 units (0.03 = 3 points). |
| Missing values | `null` (never `NaN`). |
| Auth | None. The API listens on `127.0.0.1` only. |
| CORS | Allowed origins from env `DREAMMACHINE_CORS_ORIGINS` (comma-separated). Default: `http://localhost:5173, http://127.0.0.1:5173, http://localhost:3000, http://127.0.0.1:3000`. |
| Pagination | `?limit=&offset=` → `Page<T>` (see §7). Default `limit=50`, max `limit=500`. |
| Live updates | **Polling.** Poll `GET /pipelines/{id}` or `GET /benchmark-runs/{id}` every **2 s** while `status` is `queued` or `running`; stop when it is `done`, `failed` or `cancelled`. No WebSockets in v0.1. |
| Long text logs | `GET .../log?offset=<bytes>` returns the next chunk and `next_offset`. |

### 3.1 Error envelope (every non-2xx response)
```json
{ "error": { "code": "validation_error", "message": "ratio must be one of 1, 3, 9", "details": { "fields": [ { "loc": ["body", "ratio"], "msg": "..." } ] } } }
```
| HTTP | `code` values |
|---|---|
| 404 | `not_found` |
| 409 | `conflict` (e.g. cancel a finished job, resume a running job), `not_ready` (results asked for before any evaluation finished) |
| 422 | `validation_error`, `paper_mode_locked`, `reuse_mismatch`, `invalid_feature`, `unknown_model`, `unknown_benchmark` |
| 500 | `internal_error` |
| 503 | `worker_unavailable` (only on create endpoints when the worker heartbeat is older than 30 s; the job is still created and queued — see `details.job_id`) |

Step errors (inside a job, not HTTP errors) use: `no_significant_weakness`, `gpu_unavailable`,
`out_of_memory`, `model_access_denied`, `dataset_unavailable`, `disk_low`, `interrupted`, `cancelled`,
`internal_error`.

---

## 4. Enums (exact strings)

```ts
type Mode         = "explore" | "paper";
type JobKind      = "pipeline" | "benchmark_run";
type JobStatus    = "queued" | "running" | "done" | "failed" | "cancelled";
type StepStatus   = "pending" | "running" | "done" | "failed" | "skipped" | "cancelled";
type Stage        = "preflight" | "baseline" | "diagnose" | "choose_target" | "build_data" | "train" | "evaluate" | "compare";
type Arm          = "real_only" | "untargeted" | "matched_control" | "targeted";      // "base" = untrained model, results only
type Feature      = "steps" | "log10_max" | "n_mul" | "n_div" | "n_carry" | "n_distractors" | "has_merge";
type ErrorType    = "CORRECT" | "FORMAT_ERROR" | "ARITHMETIC_SLIP" | "DISTRACTOR_USE" | "PLAN_ERROR" | "UNVERIFIABLE";
type TargetStrategy = "auto" | "feature" | "untargeted";
type BenchmarkKind  = "external" | "synthetic";
type ProgressUnit   = "examples" | "steps" | "items";
```
Benchmark names: `gsm8k`, `gsm_symbolic:main`, `gsm_symbolic:p1`, `gsm_symbolic:p2`,
`probes_train_families`, `probes_heldout_families` (+ `jsonl:<path>` for local files, CLI only).
Step keys: `preflight`, `baseline`, `diagnose`, `choose_target`, `build_data`, `train:<arm key>`,
`evaluate:<arm key>`, `compare`. Arm key = `<arm>_r<ratio>_s<seed>`, e.g. `targeted_r3_s0`, `real_only_r0_s0`.

---

## 5. Endpoint list

All paths are relative to the server root and include the `/api/v1` prefix.

<!-- ENDPOINTS:START -->
| Method | Path | Purpose |
|---|---|---|
| GET | /api/v1/health | Is the API up; versions |
| GET | /api/v1/system | GPU, worker status, versions |
| GET | /api/v1/meta | Features, error types, arms, stages, families (with plain-English labels) |
| GET | /api/v1/models | Models the user may pick |
| GET | /api/v1/benchmarks | Benchmarks the user may pick |
| GET | /api/v1/presets | Defaults and allowed values for the explore and paper forms |
| GET | /api/v1/components | Component registry (built / confirmed) |
| GET | /api/v1/queue | What the worker is doing and what is waiting |
| POST | /api/v1/pipelines | Start a full pipeline (benchmark → … → comparison) |
| GET | /api/v1/pipelines | List pipelines |
| GET | /api/v1/pipelines/{pipeline_id} | Pipeline status with all steps (poll this) |
| POST | /api/v1/pipelines/{pipeline_id}/cancel | Cancel a queued or running pipeline |
| POST | /api/v1/pipelines/{pipeline_id}/resume | Resume a failed or cancelled pipeline |
| GET | /api/v1/pipelines/{pipeline_id}/baseline | Base model results per benchmark |
| GET | /api/v1/pipelines/{pipeline_id}/diagnosis | Weakness ranking and the chosen target |
| GET | /api/v1/pipelines/{pipeline_id}/data | Training sets per arm, sizes, match quality |
| GET | /api/v1/pipelines/{pipeline_id}/data/{arm_key}/samples | A few example training problems from one arm |
| GET | /api/v1/pipelines/{pipeline_id}/training | Training progress, loss curves |
| GET | /api/v1/pipelines/{pipeline_id}/results | Final (or partial) comparison |
| GET | /api/v1/pipelines/{pipeline_id}/report | report.md as text/markdown |
| GET | /api/v1/pipelines/{pipeline_id}/log | Job log, chunked |
| POST | /api/v1/benchmark-runs | Benchmark (+ diagnose) a model without training |
| GET | /api/v1/benchmark-runs | List benchmark runs |
| GET | /api/v1/benchmark-runs/{job_id} | Benchmark run status (poll this) |
| POST | /api/v1/benchmark-runs/{job_id}/cancel | Cancel a benchmark run |
| GET | /api/v1/benchmark-runs/{job_id}/baseline | Results per benchmark |
| GET | /api/v1/benchmark-runs/{job_id}/diagnosis | Weakness ranking |
| GET | /api/v1/benchmark-runs/{job_id}/log | Job log, chunked |
| GET | /api/v1/runs | Low-level stored runs |
| GET | /api/v1/runs/{run_id} | One stored run + artifact keys |
| GET | /api/v1/runs/{run_id}/responses | Individual model answers, filterable |
| GET | /api/v1/runs/{run_id}/artifacts/{key} | One stored JSON artifact |
| GET | /api/v1/compare | Compare any two evaluation runs |
| POST | /api/v1/tools/generate | Generate example problems (playground) |
| POST | /api/v1/tools/classify | Classify one model answer (playground) |
| POST | /api/v1/tools/lltm-fit | Fit an LLTM on given data (playground) |
<!-- ENDPOINTS:END -->

---

## 6. Endpoint details

Example values below are **made up** to show the shape. They are not real results.

### 6.1 System and catalog

**`GET /health`** → `{ "status": "ok", "version": "0.1.0", "api_version": "0.1.0" }`

**`GET /system`** →
```json
{
  "gpu": { "available": true, "name": "NVIDIA GeForce RTX 4060 Laptop GPU", "vram_total_mb": 8188, "vram_free_mb": 7400 },
  "worker": { "alive": true, "pid": 1234, "heartbeat_at": "2026-10-05T10:15:02Z", "current_job_id": "a1b2c3d4e5f6", "current_step": "train:targeted_r3_s0" },
  "versions": { "python": "3.11.9", "torch": "2.x", "transformers": "4.x", "peft": "0.x", "bitsandbytes": "0.x" },
  "platform": "Windows-11"
}
```
If torch is not installed, `gpu.available` is `false` and the other gpu fields are `null`.

**`GET /meta`** →
```json
{
  "features": [ { "name": "n_carry", "label": "Carries and borrows", "description": "How many column carries (addition) and borrows (subtraction) the problem needs." } ],
  "error_types": [ { "name": "ARITHMETIC_SLIP", "label": "Arithmetic slip", "description": "A written equation is false." } ],
  "arms": [ { "name": "matched_control", "label": "Matched control", "description": "Equally hard problems, hard for a different reason." } ],
  "stages": [ { "name": "diagnose", "label": "Find weaknesses" } ],
  "families": { "marble_jar": { "split": "train", "unit": "marbles" } }
}
```
`features` has all 7, `error_types` all 6, `arms` all 4 plus `base`, `stages` all 8.

**`GET /models`** → `Model[]`
```json
[ { "id": "Qwen/Qwen3-0.6B", "display_name": "Qwen3 0.6B", "params_b": 0.6, "gated": false, "notes": null } ]
```
Source: `configs/models.yaml`. `params_b` is approximate. Gated models need `huggingface-cli login` on the laptop.

**`GET /benchmarks`** → `BenchmarkInfo[]`
```json
[ { "name": "gsm8k", "kind": "external", "description": "GSM8K test set (1319 grade-school word problems).", "size": 1319 },
  { "name": "probes_heldout_families", "kind": "synthetic", "description": "Generated probe problems in story styles never used for training.", "size": null } ]
```

**`GET /presets`** → defaults and allowed values for the two forms
```json
{
  "explore": {
    "defaults": { "benchmarks": ["gsm8k", "gsm_symbolic:main", "probes_train_families", "probes_heldout_families"], "benchmark_limit": 300, "target": { "strategy": "auto" }, "arms": ["targeted", "matched_control"], "ratio": 3, "seeds": [0] },
    "allowed": { "ratio": [1, 3, 9], "seeds": [0, 1, 2], "max_seeds": 3, "min_benchmark_limit": 10 }
  },
  "paper": {
    "description": "Fixed protocol: 4 arms incl. matched control, 3 seeds, ratios 1/3/9 for targeted, equal token budget.",
    "fixed": { "arms": ["real_only", "untargeted", "matched_control", "targeted"], "seeds": [0, 1, 2], "main_ratio": 3, "targeted_ratios": [1, 3, 9] },
    "settable": ["name", "model"],
    "n_steps": 42
  }
}
```
(`n_steps` = 5 setup steps + 18 train + 18 evaluate + compare = 42 with the current `main.yaml`. Numbers in
`explore.defaults` come from `configs/explore.yaml` and may change.)

**`GET /components`** → `[{ "id": "C10", "component": "...", "path": "...", "status": "IMPLEMENTED", "verify": "...", "notes": "..." }]`

**`GET /queue`** →
```json
{ "current": { "job_id": "a1b2c3d4e5f6", "kind": "pipeline", "name": "qwen06-try1", "current_step": "evaluate:targeted_r3_s0" },
  "queued": [ { "job_id": "0f9e8d7c6b5a", "kind": "benchmark_run", "name": "llama-check", "position": 1 } ] }
```

### 6.2 Pipelines (the main flow)

**`POST /pipelines`** — start one pipeline. Returns **202** with the `Pipeline` object (status `queued`).

Explore request:
```json
{
  "mode": "explore",
  "name": "qwen06-carry-try1",
  "model": "Qwen/Qwen3-0.6B",
  "benchmarks": ["gsm8k", "gsm_symbolic:main", "probes_train_families", "probes_heldout_families"],
  "benchmark_limit": 300,
  "target": { "strategy": "auto" },
  "arms": ["targeted", "matched_control"],
  "ratio": 3,
  "seeds": [0],
  "reuse_from": null
}
```
Paper request (nothing else allowed):
```json
{ "mode": "paper", "name": "paper-qwen17", "model": "Qwen/Qwen3-1.7B" }
```
Validation rules:
- `mode` required. `model` required and must be in `GET /models` (else `unknown_model`).
- Paper mode: any field other than `mode`, `name`, `model` → `422 paper_mode_locked`.
- `name` optional, 1–80 chars; default `"<model short name>-<yyyymmdd-hhmm>"`.
- `benchmarks`: non-empty, each in `GET /benchmarks` (else `unknown_benchmark`).
- `benchmark_limit`: integer ≥ 10, or `null` for the full sets.
- `target.strategy = "feature"` needs `target.feature` from the 7 features (else `invalid_feature`).
- `target.strategy = "untargeted"`: `arms` may only contain `untargeted` and `real_only`.
- `matched_control` in `arms` requires `target.strategy` `auto` or `feature`.
- `arms`: 1–4 unique arms. `ratio` ∈ {1, 3, 9}. `seeds`: 1–3 unique values from {0, 1, 2}.
- `reuse_from`: id of an explore pipeline or a benchmark run with the same model, benchmarks,
  benchmark_limit, prompt version and generation settings; else `422 reuse_mismatch`.

**`GET /pipelines?mode=&status=&limit=&offset=`** → `Page<PipelineSummary>` (newest first).

**`GET /pipelines/{pipeline_id}`** → `Pipeline` (poll every 2 s while active).
```json
{
  "id": "a1b2c3d4e5f6",
  "kind": "pipeline",
  "name": "qwen06-carry-try1",
  "mode": "explore",
  "status": "running",
  "created_at": "2026-10-05T10:15:00Z",
  "started_at": "2026-10-05T10:15:03Z",
  "finished_at": null,
  "queue_position": null,
  "cancel_requested": false,
  "request": { "...": "exactly what was sent" },
  "resolved": { "model": "Qwen/Qwen3-0.6B", "benchmarks": ["gsm8k", "gsm_symbolic:main", "probes_train_families", "probes_heldout_families"], "benchmark_limit": 300, "arms": ["targeted", "matched_control"], "ratios": [3], "seeds": [0], "prompt_version": "dm_v1", "config_hash": "9c1f…" },
  "target": { "strategy": "auto", "feature": "n_carry", "threshold": 1.0 },
  "progress": { "done_steps": 5, "total_steps": 10, "fraction": 0.5 },
  "current_step": "train:targeted_r3_s0",
  "steps": [
    { "index": 0, "key": "preflight", "stage": "preflight", "label": "Check setup", "status": "done", "progress": { "done": 1, "total": 1, "unit": "items", "fraction": 1.0 }, "started_at": "2026-10-05T10:15:03Z", "finished_at": "2026-10-05T10:15:09Z", "duration_s": 6.0, "run_ids": [], "error": null },
    { "index": 5, "key": "train:targeted_r3_s0", "stage": "train", "label": "Train: targeted, ratio 3, seed 0", "status": "running", "progress": { "done": 120, "total": 400, "unit": "steps", "fraction": 0.3 }, "started_at": "2026-10-05T10:41:00Z", "finished_at": null, "duration_s": null, "run_ids": [], "error": null }
  ],
  "paper_eligible": false,
  "reuse_from": null,
  "warnings": [ "seed 0: only 1830/2100 targeted items; increase data.pool_size" ],
  "error": null
}
```
`target` is `null` until `choose_target` is done. `error` (job level) is the first failing step's error:
`{ "code": "out_of_memory", "message": "…", "step_key": "train:targeted_r3_s0" }`.

**`POST /pipelines/{id}/cancel`** → 200 `Pipeline` (status becomes `cancelled` once the step stops; it may
show `running` with `cancel_requested: true` for a few seconds). On a finished job → `409 conflict`.

**`POST /pipelines/{id}/resume`** → 200 `Pipeline` (back to `queued`; done/skipped steps are kept).
Only for `failed` or `cancelled` → otherwise `409 conflict`.

**`GET /pipelines/{id}/baseline`** → `BaselineResult` (`409 not_ready` until the baseline step is done or skipped)
```json
{
  "model": "Qwen/Qwen3-0.6B",
  "reused_from": null,
  "run_ids": ["5e6f7a8b9c0d"],
  "benchmarks": [
    { "name": "gsm8k", "kind": "external", "n": 300, "accuracy": 0.41,
      "error_distribution": { "ARITHMETIC_SLIP": 0.31, "PLAN_ERROR": 0.44, "FORMAT_ERROR": 0.05, "DISTRACTOR_USE": 0.08, "UNVERIFIABLE": 0.12 } }
  ]
}
```
`error_distribution` = share of each error type **among wrong answers** (sums to 1, or `{}` if no errors).

**`GET /pipelines/{id}/diagnosis`** → `DiagnosisResult` (409 `not_ready` until diagnose is done/skipped)
```json
{
  "run_id": "7a8b9c0d1e2f",
  "probe_accuracy": 0.52,
  "theta": 1.8,
  "fit": { "converged": true, "mcfadden_r2": 0.21, "n_items": 864, "n_obs": 2592 },
  "weaknesses": [
    { "rank": 1, "feature": "n_carry", "label": "Carries and borrows", "eta": 0.42, "ci_low": 0.30, "ci_high": 0.55, "range": 3.0, "effect": 1.26, "significant": true }
  ],
  "target": { "strategy": "auto", "feature": "n_carry", "threshold": 1.0, "significant": true },
  "error_distribution": { "ARITHMETIC_SLIP": 0.5, "PLAN_ERROR": 0.3, "DISTRACTOR_USE": 0.1, "FORMAT_ERROR": 0.05, "UNVERIFIABLE": 0.05 }
}
```
Plain meaning for the UI: **η (eta)** = how much one more unit of this feature lowers the model's odds of being
right; **effect** = η × the feature's normal range; weaknesses are sorted by significant first, then effect.
`target` is `null` before `choose_target` finishes, or if no significant weakness was found.

**`GET /pipelines/{id}/data`** → `DataResult` (409 `not_ready` until build_data is done)
```json
{
  "target": "n_carry", "threshold": 1.0, "budget_tokens": 200000,
  "arms": [ { "key": "targeted_r3_s0", "arm": "targeted", "ratio": 3, "seed": 0, "n": 2400, "tokens": 199500, "n_synthetic": 1900, "n_real": 500 } ],
  "matching": [ { "seed": 0, "n_requested": 1830, "n_matched": 1830, "mean_abs_logit_diff": 0.03, "max_abs_logit_diff": 0.21, "ks_statistic": 0.02, "ks_pvalue": 0.9 } ],
  "warnings": []
}
```

**`GET /pipelines/{id}/data/{arm_key}/samples?limit=5&source=synthetic`** →
`[{ "prompt": "Omar owns 75 marbles. …", "completion": "75 - 71 = 4 …\n#### 118", "source": "synthetic:train", "features": { "steps": 3, "n_carry": 2, "...": 0 } }]`
`limit` 1–50 (default 5). `source` = `synthetic` | `real` | omitted (both). `features` is `null` for real problems.

**`GET /pipelines/{id}/training`** → `TrainingResult[]`
```json
[ { "key": "targeted_r3_s0", "arm": "targeted", "ratio": 3, "seed": 0, "status": "running",
    "step": 120, "total_steps": 400, "final_loss": null, "duration_s": null,
    "loss_curve": [ { "step": 10, "loss": 1.42 }, { "step": 20, "loss": 1.10 } ] } ]
```

**`GET /pipelines/{id}/results`** → `PipelineResults`. Works once at least one `evaluate:*` step is done
(`complete: false` until `compare` is done); before that `409 not_ready`.
```json
{
  "pipeline_id": "a1b2c3d4e5f6",
  "complete": true,
  "mode": "explore",
  "paper_eligible": false,
  "model": "Qwen/Qwen3-0.6B",
  "target": { "strategy": "auto", "feature": "n_carry", "threshold": 1.0 },
  "benchmarks": [ { "name": "gsm8k", "kind": "external" }, { "name": "probes_heldout_families", "kind": "synthetic" } ],
  "baseline": { "gsm8k": { "accuracy": 0.41, "n": 300 } },
  "arms": [
    {
      "arm": "targeted", "ratio": 3, "seeds": [0],
      "benchmarks": {
        "gsm8k": { "accuracy_mean": 0.44, "accuracy_sd": 0.0, "n_seeds": 1,
                   "vs_base": { "mean_diff": 0.03, "ci_low": 0.004, "ci_high": 0.057, "p_value": 0.02, "n": 300 },
                   "regression": false }
      },
      "target_slice": { "probes_heldout_families": { "above": { "before": 0.35, "after": 0.47, "n": 120 }, "at_or_below": { "before": 0.66, "after": 0.67, "n": 260 } } },
      "eta_change": { "n_carry": -0.21, "steps": 0.02, "log10_max": 0.0, "n_mul": 0.01, "n_div": -0.01, "n_distractors": 0.0, "has_merge": null },
      "error_type_shift": { "ARITHMETIC_SLIP": -0.08, "PLAN_ERROR": 0.05, "FORMAT_ERROR": 0.01, "DISTRACTOR_USE": 0.0, "UNVERIFIABLE": 0.02 }
    }
  ],
  "comparisons": [
    { "a": "targeted", "b": "matched_control", "ratio": 3, "benchmark": "gsm8k", "mean_diff": 0.01, "ci_low": -0.02, "ci_high": 0.04, "p_value": 0.48, "n": 300, "significant": false }
  ],
  "primary": {
    "status": "proposed",
    "question": "Does aiming at the weakness help beyond difficulty?",
    "a": "targeted", "b": "matched_control",
    "benchmarks": ["gsm8k", "gsm_symbolic:main"],
    "results": [ { "benchmark": "gsm8k", "mean_diff": 0.01, "ci_low": -0.02, "ci_high": 0.04, "p_value": 0.48, "n": 300, "significant": false } ]
  },
  "regressions": [ { "arm": "matched_control", "ratio": 3, "benchmark": "gsm_symbolic:main", "mean_diff": -0.04, "ci_low": -0.07, "ci_high": -0.01 } ],
  "warnings": []
}
```
Definitions the UI should show in tooltips:
- `vs_base` = paired comparison with the untrained model on the same problems (95% CI from a bootstrap).
- `significant` = the 95% CI does not include 0.
- `regression` = the model got worse than the untrained model: the whole CI is below 0.
- `eta_change` negative = that feature hurts **less** after training (good). `null` = could not be estimated.
- `primary.status` stays `"proposed"` until the team confirms the headline result.
- `primary` is `null` if the pipeline has no `targeted` + `matched_control` pair.

**`GET /pipelines/{id}/report`** → `text/markdown` (the `report.md` file). 409 `not_ready` before `compare`.

**`GET /pipelines/{id}/log?offset=0`** →
`{ "text": "…next chunk…", "next_offset": 20480, "eof": false }` (max 64 KB per call; call again with
`next_offset` until `eof` is true and the job is finished).

### 6.3 Benchmark runs (test a model without training)

**`POST /benchmark-runs`** → 202 `BenchmarkRun`
```json
{ "name": "qwen06-check", "model": "Qwen/Qwen3-0.6B", "benchmarks": ["gsm8k", "gsm_symbolic:main"], "benchmark_limit": 300, "include_diagnosis": true }
```
Steps: `preflight`, `baseline`, and `diagnose` if `include_diagnosis` (default `true`).
Its id can be passed as `reuse_from` when creating an explore pipeline.

**`GET /benchmark-runs`** → `Page<BenchmarkRunSummary>`.
**`GET /benchmark-runs/{job_id}`** → `BenchmarkRun` (same fields as `Pipeline` except: `kind: "benchmark_run"`,
no `target`, no `paper_eligible`, mode is always `"explore"`).
**`POST /benchmark-runs/{job_id}/cancel`**, **`GET /benchmark-runs/{job_id}/baseline`**,
**`GET /benchmark-runs/{job_id}/diagnosis`**, **`GET /benchmark-runs/{job_id}/log`** — same shapes as the
pipeline versions.

### 6.4 Runs, answers and generic compare (drill-down)

**`GET /runs?kind=&job_id=&limit=&offset=`** → `Page<Run>`; `Run` =
`{ "id", "name", "kind": "screen|diagnose|build-data|train|eval", "status", "mode", "job_id", "config": {…}, "metrics": {…}|null, "created_at", "finished_at" }`.

**`GET /runs/{run_id}`** → `Run` + `"artifacts": ["lltm", "summary", …]`.

**`GET /runs/{run_id}/responses?correct=false&error_type=ARITHMETIC_SLIP&source=gsm8k&limit=50&offset=0`** →
`Page<ResponseRecord>`:
```json
{ "items": [ { "example_id": "gsm8k-test-12", "source": "gsm8k", "sample": 0, "text": "…model answer…", "correct": false,
               "error_type": "ARITHMETIC_SLIP",
               "diagnosis": { "error_type": "ARITHMETIC_SLIP", "predicted": "118", "gold": "128", "n_equations": 3, "n_slips": 1, "slip_index": 1, "slip_ops": ["+"], "slip_operand_digits": 2, "divergence_step": null, "progress": 0.66 },
               "features": {} } ],
  "total": 177, "limit": 50, "offset": 0 }
```
`features` is `{}` for real (GSM8K) problems.

**`GET /runs/{run_id}/artifacts/{key}`** → the stored JSON value.

**`GET /compare?before=<eval run id>&after=<eval run id>`** →
```json
{ "before": "5e6f7a8b9c0d", "after": "6f7a8b9c0d1e",
  "benchmarks": [ { "name": "gsm8k", "before": 0.41, "after": 0.44, "mean_diff": 0.03, "ci_low": 0.004, "ci_high": 0.057, "p_value": 0.02, "n": 300, "regression": false } ],
  "eta_change": { "n_carry": -0.21 }, "error_type_shift": { "ARITHMETIC_SLIP": -0.08 } }
```
Both ids must be `eval` runs (else `422 validation_error`).

### 6.5 Tools (playground; fast, CPU only)

**`POST /tools/generate`** — body: `{ "n": 5, "seed": 0, "steps": 3, "digits": 2, "op_weights": [0.35, 0.35, 0.15, 0.15], "n_distractors": 0, "merge": false, "split": "train", "family": null }`
(limits: n 1–500, steps 1–12, digits 1–6, n_distractors 0–5) → list of generated items
(`id, question, solution, answer, features, family, split, …` — shape of `Item.to_dict()`).

**`POST /tools/classify`** — body: `{ "response": "…", "item": {…generated item…} }` **or**
`{ "response": "…", "question": "…", "reference_solution": "GSM8K solution with <<a*b=c>>" }` → `Diagnosis`
(same shape as `diagnosis` in `ResponseRecord`).

**`POST /tools/lltm-fit`** — body: `{ "Q": [[…]], "successes": […], "trials": […]|null, "feature_names": […]|null, "l2": 0.01 }`
→ LLTM fit (`feature_names, theta, se_theta, eta, se_eta, ci_low, ci_high, ci_method, n_items, n_obs, loglik, loglik_null, mcfadden_r2, converged, iterations, dropped`).

---

## 7. Shared types (TypeScript notation, for the frontend)

```ts
interface Page<T> { items: T[]; total: number; limit: number; offset: number; }

interface Progress { done: number; total: number; unit: ProgressUnit; fraction: number; }   // fraction 0..1

interface StepError { code: string; message: string; details?: Record<string, unknown> | null; }

interface Step {
  index: number; key: string; stage: Stage; label: string; status: StepStatus;
  progress: Progress; started_at: string | null; finished_at: string | null; duration_s: number | null;
  run_ids: string[]; error: StepError | null;
}

interface JobBase {
  id: string; kind: JobKind; name: string; mode: Mode; status: JobStatus;
  created_at: string; started_at: string | null; finished_at: string | null;
  queue_position: number | null; cancel_requested: boolean;
  request: Record<string, unknown>; resolved: Record<string, unknown>;
  progress: { done_steps: number; total_steps: number; fraction: number };
  current_step: string | null; steps: Step[]; reuse_from: string | null;
  warnings: string[]; error: (StepError & { step_key: string }) | null;
}

interface Pipeline extends JobBase {
  kind: "pipeline";
  target: { strategy: TargetStrategy; feature: Feature | null; threshold: number | null } | null;
  paper_eligible: boolean;
}

interface BenchmarkRun extends JobBase { kind: "benchmark_run"; }

type PipelineSummary = Pick<Pipeline, "id" | "name" | "mode" | "status" | "created_at" | "finished_at" | "progress" | "target" | "paper_eligible"> & { model: string };
type BenchmarkRunSummary = Pick<BenchmarkRun, "id" | "name" | "status" | "created_at" | "finished_at" | "progress"> & { model: string };

interface Bootstrap { mean_diff: number; ci_low: number; ci_high: number; p_value: number; n: number; }
```
All other objects are defined by the examples in §6; every field shown there is always present
(value may be `null` where noted).

---

## 8. Screens → endpoints (guidance for the frontend)

| Screen | Uses |
|---|---|
| Home / status bar | `GET /health`, `GET /system` (GPU + worker alive), `GET /queue` |
| New run form | `GET /presets`, `GET /models`, `GET /benchmarks`, `GET /meta` → `POST /pipelines` or `POST /benchmark-runs` |
| Run progress | poll `GET /pipelines/{id}` every 2 s; step list with progress bars; `GET .../log`; cancel / resume buttons |
| Baseline & weaknesses | `GET .../baseline`, `GET .../diagnosis` (bar chart of `effect`, mark `significant`, highlight `target`) |
| "Try another weakness" | from a finished benchmark run or explore pipeline → open the new-run form with `reuse_from` set and `target.strategy = "feature"` |
| Training data | `GET .../data`, `GET .../data/{arm_key}/samples` |
| Training | `GET .../training` (loss curves) |
| Results | `GET .../results`: before/after per arm, comparisons table, primary result card, regressions in red; `GET .../report` for download |
| Answer explorer | `GET /runs/{run_id}/responses` with filters (run ids come from `steps[].run_ids`) |
| History | `GET /pipelines`, `GET /benchmark-runs` |
| Playground (optional) | `/tools/*` |

Show a clear badge for **mode**: explore runs are labelled "Explore — not paper evidence". Show
`paper_eligible` only for paper runs.

---

## 9. State machine

```
job:   queued -> running -> done
                       \-> failed    --resume--> queued
                       \-> cancelled --resume--> queued
       queued -> cancelled
step:  pending -> running -> done | failed | cancelled
       pending -> skipped            (reuse)
```

---

## 10. Versioning and changelog

- Semantic versioning. While `< 1.0.0`, a breaking change bumps the **minor** version; additions bump the
  **patch** version.
- Every change to a route or a response shape: update this file, bump the version, add a line below, and
  regenerate `docs/openapi.json` in the same commit. `GET /health.api_version` always equals this version.

| Version | Date | Change |
|---|---|---|
| 0.1.0 | 2026-10-04 | First draft: pipelines, benchmark runs, catalog, results, runs drill-down, tools. Not implemented yet. |
