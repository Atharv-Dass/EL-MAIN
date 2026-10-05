"""Response and request models: docs/API_CONTRACT.md §7, one class per TypeScript type.

Every listed field is always present in a response (`| null` -> Optional with no default). Fields outside the
contract are dropped on output, so the published schema (docs/openapi.json) is exactly the contract.
"""

from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")

Mode = Literal["explore", "paper"]
JobKind = Literal["pipeline", "benchmark_run"]
JobStatus = Literal["queued", "running", "done", "failed", "cancelled"]
StepStatus = Literal["pending", "running", "done", "failed", "skipped", "cancelled"]
Stage = Literal["preflight", "baseline", "diagnose", "choose_target", "build_data", "train", "evaluate", "compare"]
Arm = Literal["real_only", "untargeted", "matched_control", "targeted"]
Feature = Literal["steps", "log10_max", "n_mul", "n_div", "n_carry", "n_distractors", "has_merge"]
ErrorTypeName = Literal["CORRECT", "FORMAT_ERROR", "ARITHMETIC_SLIP", "DISTRACTOR_USE", "PLAN_ERROR", "UNVERIFIABLE"]
TargetStrategy = Literal["auto", "feature", "untargeted"]
BenchmarkKind = Literal["external", "synthetic"]
ProgressUnit = Literal["examples", "steps", "items"]
RunKind = Literal["screen", "diagnose", "build-data", "train", "eval"]
ErrorShares = dict[str, float]


class Model_(BaseModel):
    model_config = ConfigDict(extra="ignore")


# ---------------------------------------------------------------- shared
class Page(Model_, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int


class Progress(Model_):
    done: int
    total: int
    unit: ProgressUnit
    fraction: float


class ErrorBody(Model_):
    code: str
    message: str
    details: dict[str, Any] | None


class ApiError(Model_):
    error: ErrorBody


class StepError(Model_):
    code: str
    message: str
    details: dict[str, Any] | None


class JobError_(StepError):
    step_key: str | None


class Bootstrap(Model_):
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int


class LogChunk(Model_):
    text: str
    next_offset: int
    eof: bool


# --------------------------------------------------------------- requests
class TargetAuto(Model_):
    strategy: Literal["auto"]


class TargetFeature(Model_):
    strategy: Literal["feature"]
    feature: Feature


class TargetUntargeted(Model_):
    strategy: Literal["untargeted"]


class CreateExplorePipeline(Model_):
    mode: Literal["explore"]
    model: str
    name: str | None = None
    benchmarks: list[str] | None = None
    benchmark_limit: int | None = None
    target: TargetAuto | TargetFeature | TargetUntargeted | None = None
    arms: list[Arm] | None = None
    ratio: Literal[1, 3, 9] | None = None
    seeds: list[int] | None = None
    reuse_from: str | None = None


class CreatePaperPipeline(Model_):
    mode: Literal["paper"]
    model: str
    name: str | None = None


class CreateBenchmarkRun(Model_):
    model: str
    name: str | None = None
    benchmarks: list[str] | None = None
    benchmark_limit: int | None = None
    include_diagnosis: bool | None = None


# ------------------------------------------------------------------ jobs
class ResolvedConfig(Model_):
    model: str
    benchmarks: list[str]
    benchmark_limit: int | None
    arms: list[Arm]
    ratios: list[float]
    seeds: list[int]
    prompt_version: str
    config_hash: str
    baseline_hash: str
    diagnosis_hash: str | None


class GpuSummary(Model_):
    max_temp_c: float | None
    min_sm_clock_mhz: float | None
    max_memory_used_mb: float | None


class StepMetrics(Model_):
    load_s: float | None
    run_s: float | None
    gpu: GpuSummary | None


class Step(Model_):
    index: int
    key: str
    stage: Stage
    label: str
    status: StepStatus
    progress: Progress
    started_at: str | None
    finished_at: str | None
    duration_s: float | None
    run_ids: list[str]
    metrics: StepMetrics | None
    error: StepError | None


class JobProgress(Model_):
    done_steps: int
    total_steps: int
    fraction: float


class Target(Model_):
    strategy: TargetStrategy
    feature: Feature | None
    threshold: float | None


class JobBase(Model_):
    id: str
    kind: JobKind
    name: str
    mode: Mode
    status: JobStatus
    created_at: str
    started_at: str | None
    finished_at: str | None
    queue_position: int | None
    cancel_requested: bool
    request: dict[str, Any]
    resolved: ResolvedConfig
    progress: JobProgress
    eta_s: float | None
    output_dir: str
    output_bytes: int
    current_step: str | None
    steps: list[Step]
    reuse_from: str | None
    warnings: list[str]
    error: JobError_ | None


class Pipeline(JobBase):
    kind: Literal["pipeline"]
    target: Target | None
    paper_eligible: bool


class BenchmarkRun(JobBase):
    kind: Literal["benchmark_run"]
    mode: Literal["explore"]


class PipelineSummary(Model_):
    id: str
    name: str
    mode: Mode
    status: JobStatus
    created_at: str
    finished_at: str | None
    progress: JobProgress
    target: Target | None
    paper_eligible: bool
    model: str


class BenchmarkRunSummary(Model_):
    id: str
    name: str
    status: JobStatus
    created_at: str
    finished_at: str | None
    progress: JobProgress
    model: str


class DeleteResult(Model_):
    deleted: Literal[True]
    freed_bytes: int


# --------------------------------------------------------- stage results
class BaselineBenchmark(Model_):
    name: str
    kind: BenchmarkKind
    n: int
    accuracy: float
    error_distribution: ErrorShares


class BaselineResult(Model_):
    model: str
    reused_from: str | None
    run_ids: list[str]
    benchmarks: list[BaselineBenchmark]


class Weakness(Model_):
    rank: int
    feature: Feature
    label: str
    eta: float
    ci_low: float | None
    ci_high: float | None
    range: float
    effect: float
    significant: bool


class Fit(Model_):
    converged: bool
    mcfadden_r2: float | None
    n_items: int
    n_obs: int


class DiagnosisTarget(Target):
    significant: bool


class DiagnosisResult(Model_):
    run_id: str
    probe_accuracy: float
    theta: float
    fit: Fit
    weaknesses: list[Weakness]
    target: DiagnosisTarget | None
    error_distribution: ErrorShares


class DataArm(Model_):
    key: str
    arm: Arm
    ratio: float
    seed: int
    n: int
    tokens: int
    n_synthetic: int
    n_real: int


class Matching(Model_):
    seed: int
    n_requested: int
    n_matched: int
    mean_abs_logit_diff: float | None
    max_abs_logit_diff: float | None
    ks_statistic: float | None
    ks_pvalue: float | None


class DataResult(Model_):
    target: Feature | None
    threshold: float | None
    budget_tokens: int
    arms: list[DataArm]
    matching: list[Matching]
    warnings: list[str]


class TrainingSample(Model_):
    prompt: str
    completion: str
    source: str
    features: dict[str, float] | None


class LossPoint(Model_):
    step: int
    loss: float


class TrainingResult(Model_):
    key: str
    arm: Arm
    ratio: float
    seed: int
    status: StepStatus
    step: int
    total_steps: int
    final_loss: float | None
    duration_s: float | None
    loss_curve: list[LossPoint]


# ------------------------------------------------------- final comparison
class ArmBenchmarkResult(Model_):
    accuracy_mean: float
    accuracy_sd: float
    n_seeds: int
    vs_base: Bootstrap
    regression: bool


class SliceResult(Model_):
    before: float | None
    after: float | None
    n: int


class TargetSlice(Model_):
    above: SliceResult
    at_or_below: SliceResult


class ArmResult(Model_):
    arm: Arm
    ratio: float
    seeds: list[int]
    benchmarks: dict[str, ArmBenchmarkResult]
    target_slice: dict[str, TargetSlice]
    eta_change: dict[str, float | None]
    error_type_shift: ErrorShares


class Comparison(Bootstrap):
    a: Arm
    b: Arm
    ratio: float
    benchmark: str
    significant: bool


class Regression(Model_):
    arm: Arm
    ratio: float
    benchmark: str
    mean_diff: float
    ci_low: float
    ci_high: float


class PrimaryRow(Bootstrap):
    benchmark: str
    significant: bool


class Primary(Model_):
    status: Literal["proposed", "confirmed"]
    question: str
    a: Literal["targeted"]
    b: Literal["matched_control"]
    benchmarks: list[str]
    results: list[PrimaryRow]


class BenchmarkRef(Model_):
    name: str
    kind: BenchmarkKind


class BaselineCell(Model_):
    accuracy: float
    n: int


class PipelineResults(Model_):
    pipeline_id: str
    complete: bool
    mode: Mode
    paper_eligible: bool
    model: str
    target: Target | None
    benchmarks: list[BenchmarkRef]
    baseline: dict[str, BaselineCell]
    arms: list[ArmResult]
    comparisons: list[Comparison]
    primary: Primary | None
    regressions: list[Regression]
    warnings: list[str]


class CompareBenchmark(Model_):
    name: str
    before: float
    after: float
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int
    regression: bool


class CompareResult(Model_):
    before: str
    after: str
    benchmarks: list[CompareBenchmark]
    eta_change: dict[str, float | None]
    error_type_shift: ErrorShares
    warnings: list[str]


# ------------------------------------------------------- catalog / runs
class Health(Model_):
    status: Literal["ok"]
    version: str
    api_version: str


class GpuInfo(Model_):
    available: bool
    name: str | None
    vram_total_mb: float | None
    vram_free_mb: float | None


class WorkerInfo(Model_):
    alive: bool
    pid: int | None
    heartbeat_at: str | None
    current_job_id: str | None
    current_step: str | None


class DiskInfo(Model_):
    free_gb: float
    runs_bytes: int
    min_free_gb: float


class SystemInfo(Model_):
    gpu: GpuInfo
    worker: WorkerInfo
    versions: dict[str, str | None]
    platform: str
    disk: DiskInfo


class NamedLabel(Model_):
    name: str
    label: str
    description: str


class StageLabel(Model_):
    name: Stage
    label: str


class FamilyInfo(Model_):
    split: str
    unit: str


class Meta(Model_):
    features: list[NamedLabel]
    error_types: list[NamedLabel]
    arms: list[NamedLabel]
    stages: list[StageLabel]
    families: dict[str, FamilyInfo]


class ModelInfo(Model_):
    id: str
    display_name: str
    params_b: float | None
    gated: bool
    notes: str | None


class BenchmarkInfo(Model_):
    name: str
    kind: BenchmarkKind
    description: str
    size: int | None


class ExploreDefaults(Model_):
    benchmarks: list[str]
    benchmark_limit: int | None
    target: TargetAuto
    arms: list[Arm]
    ratio: int
    seeds: list[int]


class ExploreAllowed(Model_):
    ratio: list[int]
    seeds: list[int]
    max_seeds: int
    min_benchmark_limit: int


class ExplorePreset(Model_):
    defaults: ExploreDefaults
    allowed: ExploreAllowed


class PaperFixed(Model_):
    arms: list[Arm]
    seeds: list[int]
    main_ratio: float
    targeted_ratios: list[float]


class PaperPreset(Model_):
    description: str
    fixed: PaperFixed
    settable: list[str]
    n_steps: int


class Presets(Model_):
    explore: ExplorePreset
    paper: PaperPreset
    prompt_versions: list[str]       # 0.2.1
    default_prompt_version: str      # 0.2.1: the project prompt (PLAN.md D10)


class Component(Model_):
    id: str
    component: str
    path: str
    status: str
    verify: str
    notes: str


class QueueCurrent(Model_):
    job_id: str
    kind: JobKind
    name: str
    current_step: str | None


class QueueItem(Model_):
    job_id: str
    kind: JobKind
    name: str
    position: int


class Queue(Model_):
    current: QueueCurrent | None
    queued: list[QueueItem]


class Run(Model_):
    id: str
    name: str
    kind: Literal["screen", "diagnose", "build-data", "train", "eval"]
    status: str
    mode: Mode | None
    job_id: str | None
    config: dict[str, Any]
    metrics: dict[str, Any] | None
    created_at: str
    finished_at: str | None


class RunWithArtifacts(Run):
    artifacts: list[str]


class ResponseDiagnosis(Model_):
    error_type: ErrorTypeName
    predicted: str | None
    gold: str | None
    n_equations: int
    n_slips: int
    slip_index: int | None
    slip_ops: list[str]
    slip_operand_digits: int | None
    divergence_step: int | None
    progress: float | None
    lastline: str | None             # 0.2.1: last number on the last line (tolerant), null if none / not recorded
    lastline_v0: str | None          # 0.2.1: first number on the last line (the friend's extractor)
    format_ok: bool | None           # 0.2.1: last line is strictly a bare number
    truncated: bool | None           # 0.2.1: the answer hit max_new_tokens (null if the runner did not report it)


class ResponseRecord(Model_):
    example_id: str
    source: str
    sample: int
    text: str
    correct: bool
    error_type: ErrorTypeName
    diagnosis: ResponseDiagnosis
    features: dict[str, float]


# ------------------------------------------------------------------ tools
class GenerateRequest(Model_):
    n: int = Field(5, ge=1, le=500)
    seed: int = 0
    steps: int = Field(3, ge=1, le=12)
    digits: int = Field(2, ge=1, le=6)
    op_weights: tuple[float, float, float, float] = (0.35, 0.35, 0.15, 0.15)
    n_distractors: int = Field(0, ge=0, le=5)
    merge: bool = False
    split: str = "train"
    family: str | None = None


class ClassifyRequest(Model_):
    response: str
    question: str | None = None
    reference_solution: str | None = Field(None, description="GSM8K-format solution with <<a*b=c>> annotations")
    item: dict | None = Field(None, description="a generated item (from /tools/generate); takes precedence")


class LLTMRequest(Model_):
    Q: list[list[float]]
    successes: list[float]
    trials: list[float] | None = None
    feature_names: list[str] | None = None
    l2: float = 1e-2
