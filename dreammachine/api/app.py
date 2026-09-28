"""FastAPI backend for the (later) frontend.

    uvicorn dreammachine.api.app:app --reload

Long GPU jobs (screen/diagnose/train/evaluate) run from the CLI and write to the
same SQLite store. This API serves their results and exposes the CPU-cheap
building blocks (generation, single-response diagnosis, LLTM fitting).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .. import __version__
from ..diagnosis.align import ReferenceTrace
from ..diagnosis.lltm import fit_lltm
from ..diagnosis.taxonomy import classify
from ..generator.families import FAMILIES
from ..generator.features import FEATURES
from ..generator.sampler import GenerationError, GenSpec, generate_many
from ..store.db import Store

REPO_ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ schemas
class GenerateRequest(BaseModel):
    n: int = Field(5, ge=1, le=500)
    seed: int = 0
    steps: int = Field(3, ge=1, le=12)
    digits: int = Field(2, ge=1, le=6)
    op_weights: tuple[float, float, float, float] = (0.35, 0.35, 0.15, 0.15)
    n_distractors: int = Field(0, ge=0, le=5)
    merge: bool = False
    split: str = "train"
    family: str | None = None


class DiagnoseRequest(BaseModel):
    response: str
    question: str | None = None
    reference_solution: str | None = Field(None, description="GSM8K-format solution with <<a*b=c>> annotations")
    item: dict | None = Field(None, description="a generated item (from /generate); takes precedence")


class LLTMRequest(BaseModel):
    Q: list[list[float]]
    successes: list[float]
    trials: list[float] | None = None
    feature_names: list[str] | None = None
    l2: float = 1e-2


# ---------------------------------------------------------------- registry
_ROW = re.compile(r"^\|\s*(C\w+)\s*\|(.+)\|\s*$")


def parse_components(path: Path) -> list[dict[str, str]]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line)
        if not m:
            continue
        cells = [c.strip() for c in m.group(2).split("|")]
        cells += [""] * (5 - len(cells))
        out.append({"id": m.group(1), "component": cells[0], "path": cells[1].strip("`"),
                    "status": cells[2], "verify": cells[3].strip("`"), "notes": cells[4]})
    return out


def _nan_to_none(x: Any) -> Any:
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, list):
        return [_nan_to_none(v) for v in x]
    if isinstance(x, dict):
        return {k: _nan_to_none(v) for k, v in x.items()}
    return x


# --------------------------------------------------------------------- app
def create_app(store: Store | None = None, components_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="DreamMachine API", version=__version__)
    db = store or Store(os.environ.get("DREAMMACHINE_DB", "runs/dreammachine.db"))
    comp_path = components_path or REPO_ROOT / "docs" / "COMPONENTS.md"

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/meta")
    def meta() -> dict:
        return {"features": list(FEATURES),
                "families": {n: {"split": f.split, "unit": f.unit} for n, f in FAMILIES.items()}}

    @app.get("/components")
    def components() -> list[dict]:
        return parse_components(comp_path)

    @app.post("/generate")
    def generate(req: GenerateRequest) -> list[dict]:
        try:
            spec = GenSpec(steps=req.steps, digits=req.digits, op_weights=req.op_weights,
                           n_distractors=req.n_distractors, merge=req.merge, split=req.split, family=req.family)
            return [it.to_dict() for it in generate_many(spec, req.n, seed=req.seed)]
        except (ValueError, KeyError, GenerationError) as e:
            raise HTTPException(status_code=422, detail=str(e))

    @app.post("/diagnose")
    def diagnose(req: DiagnoseRequest) -> dict:
        if req.item is not None:
            try:
                from ..generator.sampler import Item
                trace = ReferenceTrace.from_item(Item.from_dict(req.item))
            except (TypeError, KeyError) as e:
                raise HTTPException(status_code=422, detail=f"invalid item: {e}")
        elif req.question and req.reference_solution:
            try:
                trace = ReferenceTrace.from_gsm8k(req.question, req.reference_solution)
            except (ValueError, ZeroDivisionError) as e:
                raise HTTPException(status_code=422, detail=f"invalid reference solution: {e}")
        else:
            raise HTTPException(status_code=422, detail="give either `item` or `question` + `reference_solution`")
        return classify(req.response, trace).to_dict()

    @app.post("/lltm/fit")
    def lltm_fit(req: LLTMRequest) -> dict:
        try:
            r = fit_lltm(np.array(req.Q), np.array(req.successes),
                         None if req.trials is None else np.array(req.trials), req.feature_names, l2=req.l2)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return _nan_to_none(r.to_dict())

    @app.get("/runs")
    def runs(kind: str | None = None) -> list[dict]:
        return db.list_runs(kind)

    @app.get("/runs/{run_id}")
    def run(run_id: str) -> dict:
        r = db.get_run(run_id)
        if r is None:
            raise HTTPException(status_code=404, detail="run not found")
        return {**r, "artifacts": db.list_artifacts(run_id)}

    @app.get("/runs/{run_id}/responses")
    def responses(run_id: str, limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)) -> list[dict]:
        if db.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="run not found")
        return db.get_responses(run_id, limit=limit, offset=offset)

    @app.get("/runs/{run_id}/artifacts/{key}")
    def artifact(run_id: str, key: str) -> Any:
        v = db.get_artifact(run_id, key)
        if v is None:
            raise HTTPException(status_code=404, detail="artifact not found")
        return _nan_to_none(v)

    return app


def __getattr__(name: str):  # lazily build `app` so importing this module has no side effects
    if name == "app":
        return create_app()
    raise AttributeError(name)
