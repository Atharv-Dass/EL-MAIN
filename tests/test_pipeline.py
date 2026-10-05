"""Integration test of the experiment pipeline with a simulated model.

The simulated model answers correctly with probability sigmoid(theta - q.eta)
for a planted eta whose dominant weakness is carrying. The pipeline must
recover that weakness, build all arms with equal budgets and a matched control,
train (tiny model, CPU), evaluate, and report.
"""

import csv
import json
import math
import random

import numpy as np
import pytest

from dreammachine.data import Example, from_items, save_jsonl
from dreammachine.experiments import pipeline as P
from dreammachine.generator import FEATURES, GenSpec, generate_many
from dreammachine.store import Store

PLANTED = dict(steps=0.2, log10_max=0.2, n_mul=0.1, n_div=0.1, n_carry=0.6, n_distractors=0.1, has_merge=0.0)


class SimulatedRunner:
    def __init__(self, oracle: dict[str, Example], seed: int = 0, n_samples: int = 1):
        self.oracle, self.rng, self.n = oracle, random.Random(seed), n_samples

    def _answer(self, ex: Example) -> str:
        if ex.features:
            z = 3.0 - sum(PLANTED[k] * ex.features[k] for k in FEATURES)
        else:
            z = 0.0
        if self.rng.random() < 1 / (1 + math.exp(-z)):
            return ex.solution
        step = ex.trace().steps[0]
        a_b = step.expr
        for sym in "+-*/":
            a_b = a_b.replace(sym, f" {sym} ")
        wrong = step.value + 1
        return f"{a_b} = {wrong}\n#### {wrong}"

    def generate(self, questions):
        return [[self._answer(self.oracle[q]) for _ in range(self.n)] for q in questions]


@pytest.fixture
def env(tmp_path, monkeypatch):
    oracle: dict[str, Example] = {}
    real_from_items = P.from_items

    def recording_from_items(items):
        exs = real_from_items(items)
        oracle.update({e.question: e for e in exs})
        return exs

    monkeypatch.setattr(P, "from_items", recording_from_items)
    from dreammachine.benchmarks import builtin  # eval probe sets are built here since B2
    monkeypatch.setattr(builtin, "from_items", recording_from_items)

    def fake_gsm(n, seed):
        exs = from_items(generate_many(GenSpec(steps=3, digits=2), n, seed=seed))
        out = [Example(id=f"g{seed}-{i}", source="gsm8k", question=e.question, answer=e.answer,
                       solution=e.solution) for i, e in enumerate(exs)]
        oracle.update({e.question: e for e in out})
        return out

    save_jsonl(fake_gsm(400, 1), tmp_path / "gsm_train.jsonl")
    save_jsonl(fake_gsm(40, 2), tmp_path / "gsm_test.jsonl")
    small = {"steps": [2, 4, 6], "digits": [1, 2, 3], "distractors": [0, 1], "op_profiles": ["addsub", "mixed"]}
    cfg = P.Config(
        name="t", output_dir=str(tmp_path / "out"), model="", candidates=["sim-a"],
        db=":memory:", seeds=[0], real_data=str(tmp_path / "gsm_train.jsonl"),
        eval_data={"gsm8k_test": str(tmp_path / "gsm_test.jsonl")},
        screen={"gsm8k_limit": 40, "probe_per_cell": 1, "grid": small},
        diagnose={"probe_per_cell": 6, "n_samples": 3, "bootstrap": 40, "natural_pool": 1500, "annotate": 20,
                  "grid": small},
        data={"pool_size": 6000, "natural_size": 2000, "token_budget": 15000, "ratios": [3], "main_ratio": 3,
              "grid": {"steps": [2, 3, 4, 5, 6], "digits": [1, 2, 3, 4]}},
        eval={"gsm8k_limit": 40, "gsm_symbolic": [], "probe_per_cell": 2, "grid": small},
    )
    store = Store(":memory:")
    factory = lambda model, adapter, gen: SimulatedRunner(oracle, seed=hash(adapter or "") % 1000,
                                                          n_samples=gen.n_samples)
    return cfg, store, factory


def test_pipeline_end_to_end(env, tiny_model_dir):
    cfg, store, factory = env

    scr = P.screen(cfg, store, factory)
    assert scr["candidates"][0]["gsm8k_accuracy"] > 0.25 and scr["recommended"] == "sim-a"

    diag = P.diagnose(cfg, store, factory)
    assert diag["target"]["feature"] == "n_carry"
    with (P.Path(cfg.output_dir) / "annotate_errors.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 20 and all(r["annotator_1"] == "" for r in rows)

    manifest = P.build_data(cfg, store)
    arms = manifest["arms"]
    for arm in ("real_only_r0_s0", "untargeted_r3_s0", "matched_control_r3_s0", "targeted_r3_s0"):
        assert arms[arm]["tokens"] <= 15000 and arms[arm]["tokens"] > 0.9 * 15000, (arm, arms[arm])
    assert arms["real_only_r0_s0"]["n_synthetic"] == 0
    assert arms["targeted_r3_s0"]["n_synthetic"] > arms["targeted_r3_s0"]["n_real"]
    assert arms["match_s0"]["mean_abs_logit_diff"] < 0.2

    # Training plumbing on the tiny CPU model (the base model is swapped in for the test).
    cfg.model = str(tiny_model_dir)
    cfg.train = {"load_in_4bit": False, "max_steps": 2, "per_device_batch_size": 2, "grad_accum": 1,
                 "target_modules": ["q_proj", "v_proj"], "save_steps": 1000}
    adapter = P.train_arm(cfg, store, "targeted", 3.0, 0)
    assert (adapter / "adapter_config.json").exists()

    P.evaluate_model(cfg, store, factory, "base")
    for arm in ("real_only", "untargeted", "matched_control", "targeted"):
        m = P.evaluate_model(cfg, store, factory, arm, 0.0 if arm == "real_only" else 3.0, 0)
        assert "probes_heldout_families" in m and "eta" in m
    rep = P.report(cfg, store)
    assert {"base", "real_only", "untargeted", "matched_control", "targeted"} <= set(rep["arms"])
    assert any(k.startswith("targeted - matched_control") for k in rep["comparisons"])
    assert (P.Path(cfg.output_dir) / "report.md").read_text(encoding="utf-8").startswith("# Results")


def test_paired_bootstrap():
    rng = np.random.default_rng(0)
    a = rng.binomial(1, 0.7, 500).astype(float)
    b = rng.binomial(1, 0.5, 500).astype(float)
    r = P.paired_bootstrap(a, b, n_boot=2000)
    assert r["ci_low"] < r["mean_diff"] < r["ci_high"] and r["ci_low"] > 0 and r["p_value"] < 0.01
    same = P.paired_bootstrap(a, a, n_boot=500)
    assert same["mean_diff"] == 0 and same["p_value"] == 1.0


def test_gsm_row_ids_are_unique():
    """GSM-Symbolic's `id` is a template id shared by all instances; ids must pair it with `instance`
    or responses (keyed by example_id) overwrite each other. GSM8K rows have no id: use the index."""
    from dreammachine.data.loaders import _from_gsm_rows

    sym = [{"id": t, "instance": k, "question": "q", "answer": "1 + 1 = 2\n#### 2", "original_id": 9}
           for t in range(3) for k in range(4)]
    ids = [e.id for e in _from_gsm_rows(sym, "gsm_symbolic:main", "gsmsym-main")]
    assert len(set(ids)) == 12 and ids[5] == "gsmsym-main-1-1"
    gsm = [{"question": "q", "answer": "#### 2"}] * 3
    assert [e.id for e in _from_gsm_rows(gsm, "gsm8k", "gsm8k-test")] == ["gsm8k-test-0", "gsm8k-test-1",
                                                                          "gsm8k-test-2"]


def test_dev_holdout_is_disjoint_from_training(tmp_path):
    exs = [Example(id=f"r{i}", source="gsm8k", question=f"q{i}", answer="1") for i in range(10)]
    save_jsonl(exs, tmp_path / "real.jsonl")
    cfg = P.Config(name="t", output_dir=str(tmp_path / "out"), real_data=str(tmp_path / "real.jsonl"))
    assert P.dev_set(cfg) == [] and len(P._real_train(cfg)) == 10   # off by default: unchanged behaviour
    cfg.data = {"dev_holdout": 3}
    dev, train = {e.id for e in P.dev_set(cfg)}, {e.id for e in P._real_train(cfg)}
    assert dev == {"r7", "r8", "r9"} and len(train) == 7 and not dev & train


def test_config_loads_yaml():
    for path in ("configs/main.yaml", "configs/smoke.yaml"):
        cfg = P.Config.load(path)
        assert cfg.model and cfg.seeds
        json.dumps(cfg.__dict__)
