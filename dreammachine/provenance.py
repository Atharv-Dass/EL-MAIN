"""Where a result came from (PLAN.md §9.4): code version, package versions, machine.

Cheap to call: no torch import (the GPU name is read only if torch is already loaded).
"""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("torch", "transformers", "peft", "bitsandbytes", "datasets", "accelerate")


def git_state(root: Path = REPO_ROOT) -> dict:
    """{"commit": sha | "unknown", "dirty": bool | None}."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, encoding="utf-8",
                              timeout=10, check=True).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "dirty": None}


def package_versions() -> dict:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def gpu_name() -> str | None:
    torch = sys.modules.get("torch")
    try:
        return torch.cuda.get_device_name(0) if torch is not None and torch.cuda.is_available() else None
    except Exception:
        return None


def hf_revision(model_id: str) -> str | None:
    """Commit sha of the cached Hugging Face snapshot of `model_id` (offline; None for local folders)."""
    if Path(model_id).exists():
        return None
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model_id, local_files_only=True)).name
    except Exception:
        return None


def provenance() -> dict:
    return {"git": git_state(), "python": platform.python_version(), "packages": package_versions(),
            "platform": platform.platform(), "gpu": gpu_name()}
