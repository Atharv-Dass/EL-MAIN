"""The model catalog: configs/models.yaml (models the user may pick)."""

from __future__ import annotations

from pathlib import Path

MODELS_YAML = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"
_FIELDS = ("id", "display_name", "params_b", "gated", "notes")


def load_models(path: str | Path | None = None) -> list[dict]:
    import yaml

    raw = yaml.safe_load(Path(path or MODELS_YAML).read_text(encoding="utf-8")) or {}
    models = []
    for m in raw.get("models", []):
        missing = [f for f in ("id", "display_name") if not m.get(f)]
        if missing:
            raise ValueError(f"models.yaml entry {m!r} is missing {missing}")
        models.append({f: m.get(f) for f in _FIELDS} | {"gated": bool(m.get("gated", False))})
    ids = [m["id"] for m in models]
    if len(ids) != len(set(ids)):
        raise ValueError("models.yaml lists a model id twice")
    return models


def model_ids(path: str | Path | None = None) -> list[str]:
    return [m["id"] for m in load_models(path)]
