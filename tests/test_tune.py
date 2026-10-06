"""Dev-set tuning command (dreammachine.experiments.tune): CPU plumbing with the tiny model + simulated runner."""

import json
from dataclasses import asdict

import yaml

from dreammachine.data import Example, from_items, save_jsonl
from dreammachine.experiments import pipeline as P
from dreammachine.experiments.tune import tune
from dreammachine.generator import GenSpec, generate_many
from test_build_data_refactor import _diagnosis
from test_pipeline import SimulatedRunner


def test_tune_keeps_dev_out_of_training_and_resumes(tmp_path, tiny_model_dir, monkeypatch):
    oracle = {}
    real_from_items = P.from_items

    def recording(items):
        exs = real_from_items(items)
        oracle.update({e.question: e for e in exs})
        return exs

    from dreammachine.benchmarks import builtin
    monkeypatch.setattr(P, "from_items", recording)
    monkeypatch.setattr(builtin, "from_items", recording)
    exs = from_items(generate_many(GenSpec(steps=3, digits=2), 300, seed=9))
    real = [Example(id=f"r{i}", source="gsm8k", question=e.question, answer=e.answer, solution=e.solution)
            for i, e in enumerate(exs)]
    oracle.update({e.question: e for e in real})
    save_jsonl(real, tmp_path / "real.jsonl")
    small = {"steps": [2, 4], "digits": [1, 2], "distractors": [0], "op_profiles": ["addsub", "mixed"]}
    base = P.Config(name="b", output_dir=str(tmp_path / "unused"), model=str(tiny_model_dir),
                    real_data=str(tmp_path / "real.jsonl"), diagnose={"grid": small},
                    data={"pool_size": 2500, "natural_size": 1200, "token_budget": 4000, "ratios": [3],
                          "grid": {"steps": [2, 3, 4, 5], "digits": [1, 2, 3]}},
                    train={"load_in_4bit": False, "max_steps": 2, "per_device_batch_size": 2, "grad_accum": 1,
                           "target_modules": ["q_proj", "v_proj"], "save_steps": 1000, "logging_steps": 1},
                    eval={"probe_per_cell": 2, "grid": small}, gpu={"memory_fraction": 0})
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(asdict(base)), encoding="utf-8")
    src = tmp_path / "diag"
    src.mkdir()
    diag = _diagnosis(base)
    (src / "diagnosis.json").write_text(json.dumps(diag), encoding="utf-8")
    (src / "target.json").write_text(json.dumps({"strategy": "auto", "feature": "n_carry",
                                                 "threshold": diag["threshold"]}), encoding="utf-8")
    spec = {"base": str(tmp_path / "base.yaml"), "model": str(tiny_model_dir), "dev_holdout": 20,
            "diagnosis_from": str(src), "output_dir": str(tmp_path / "tune"), "eval": {"probes_heldout_families": 10},
            "variants": [{"name": "t", "arm": "targeted", "ratio": 3, "train": {"learning_rate": 5e-5}},
                         {"name": "real", "arm": "real_only", "data": {"solution_style": "steps"}},
                         {"name": "d", "arm": "targeted", "ratio": 3, "data": {"token_budget": 3000},
                          "distill": {"samples": 2, "chunk": 8}}]}
    (tmp_path / "tune.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
    factory = lambda model, adapter, gen: SimulatedRunner(oracle, seed=len(adapter or ""), n_samples=gen.n_samples)

    res = tune(tmp_path / "tune.yaml", factory=factory, log=lambda s: None)
    assert set(res) == {"base", "t", "real", "d"}
    dev_ids = {f"r{i}" for i in range(280, 300)}
    assert set(res["base"]["eval"]["dev"]["items"]) == dev_ids
    for name, arm, ratio in (("t", "targeted", 3), ("real", "real_only", 0), ("d", "targeted", 3)):
        rows = [json.loads(l) for l in (tmp_path / "tune" / name / "data" / f"{arm}_r{ratio}_s0.jsonl")
                .read_text(encoding="utf-8").splitlines()]
        assert not dev_ids & {r["id"] for r in rows}, name          # no dev problem in the training data
        assert res[name]["vs_base"]["dev"]["n"] == 20
        assert all(r["completion"].startswith("Let's solve") for r in rows) == (name == "real")   # solution_style
    assert res["t"]["train"]["learning_rate"] == 5e-5
    assert "targeted" in (tmp_path / "tune" / "summary.md").read_text(encoding="utf-8")
    d = res["d"]["distill"]                                            # distilled variant (PLAN.md D14)
    assert d["n_distilled"] > 0 and d["n_items"] == d["n_distilled"] + d["n_kept_original"]
    assert res["d"]["data"]["targeted_r3_s0"]["tokens"] <= 3000
    data = tmp_path / "tune" / "d" / "data"
    assert (data / "targeted_r3_s0.original.jsonl").exists() and (data / "targeted_r3_s0.distill.json").exists()
    calls = []
    tune(tmp_path / "tune.yaml", factory=lambda *a: calls.append(a) or factory(*a), log=lambda s: None)
    assert calls == []                                                 # everything finished: nothing re-run
