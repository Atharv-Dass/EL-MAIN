"""Error codes of the job system (docs/API_CONTRACT.md §3.1). Request errors become HTTP 4xx (B7);
step errors are stored on the step as {code, message, details} and copied to the job."""

from __future__ import annotations

from ..errors import Cancelled, NoSignificantWeakness

REQUEST_CODES = ("validation_error", "paper_mode_locked", "reuse_mismatch", "invalid_feature", "unknown_model",
                 "unknown_benchmark", "not_found", "conflict", "not_ready", "paper_protected")
STEP_CODES = ("no_significant_weakness", "gpu_unavailable", "out_of_memory", "model_access_denied",
              "dataset_unavailable", "disk_low", "interrupted", "cancelled", "internal_error")

OOM_HINT = ("CUDA ran out of GPU memory. Lower `generation.batch_size` (inference), `train.per_device_batch_size` "
            "(and raise `train.grad_accum` to keep the effective batch), or `train.max_seq_len`.")


class JobError(Exception):
    """A failure with a contract error code (a bad request, or a step failure such as disk_low)."""

    def __init__(self, code: str, message: str, details: dict | None = None) -> None:
        if code not in REQUEST_CODES + STEP_CODES:
            raise ValueError(f"unknown error code {code!r}")
        super().__init__(message)
        self.code, self.message, self.details = code, message, details

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "details": self.details}


def _status_code(e: BaseException) -> int | None:
    resp = getattr(e, "response", None)
    return getattr(resp, "status_code", None)


def error_from_exception(e: BaseException, step_key: str = "") -> dict:
    """Map an exception raised inside a step to {code, message, details}. The traceback goes to the job log,
    never into the message."""
    if isinstance(e, JobError):
        return e.to_dict()
    if isinstance(e, Cancelled):
        return {"code": "cancelled", "message": str(e) or "cancelled by the user", "details": None}
    if isinstance(e, NoSignificantWeakness):
        return {"code": "no_significant_weakness", "message": str(e), "details": None}
    name, text = type(e).__name__, str(e)
    if name == "OutOfMemoryError" or "out of memory" in text.lower():
        return {"code": "out_of_memory", "message": OOM_HINT, "details": {"step_key": step_key}}
    if name in ("GatedRepoError", "RepositoryNotFoundError") or _status_code(e) in (401, 403):
        return {"code": "model_access_denied",
                "message": "The model is gated or private. Accept its licence on Hugging Face and run "
                           "`huggingface-cli login` on this laptop.", "details": None}
    if "needs a CUDA GPU" in text or "cannot run 4-bit" in text:
        return {"code": "gpu_unavailable", "message": text, "details": None}
    if type(e).__module__.split(".")[0] == "datasets" or "DatasetNotFound" in name:
        return {"code": "dataset_unavailable", "message": f"A dataset could not be loaded ({name}).", "details": None}
    return {"code": "internal_error",
            "message": f"Unexpected error in step {step_key or '?'} ({name}); see the job log for the traceback.",
            "details": None}
