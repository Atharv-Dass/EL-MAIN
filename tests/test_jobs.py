"""Pipeline orchestrator (PLAN.md milestone B4). CPU plumbing only: a simulated model with a planted weakness
answers every benchmark; the tiny random Llama is trained for 2 steps. No network."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from dreammachine.data import Example, from_items, save_jsonl
from dreammachine.experiments import pipeline as P
from dreammachine.generator import GenSpec, generate_many
from dreammachine.jobs import JobError, create_job_from_request, execute_step, finish_job, run_job
from dreammachine.jobs.errors import error_from_exception
from dreammachine.jobs.requests import load_presets
from dreammachine.store import Store
from test_pipeline import SimulatedRunner

BENCH = ["gsm8k", "probes_train_families", "probes_heldout_families"]
SMALL = {"steps": [2, 4, 6], "digits": [1, 2, 3], "distractors": [0, 1], "op_profiles": ["addsub", "mixed"]}


@pytest.fixture
def env(tmp_path, monkeypatch, tiny_model_dir):
    """Presets sized for CPU, local data files, and a factory for the simulated model."""
    oracle: dict[str, Example] = {}
    real_from_items = P.from_items

    def recording(items):
        exs = real_from_items(items)
        oracle.update({e.question: e for e in exs})
        return exs

    from dreammachine.benchmarks import builtin
    monkeypatch.setattr(P, "from_items", recording)
    monkeypatch.setattr(builtin, "from_items", recording)

    def fake_gsm(n, seed):
        exs = from_items(generate_many(GenSpec(steps=3, digits=2), n, seed=seed))
        out = [Example(id=f"g{seed}-{i}", source="gsm8k", question=e.question, answer=e.answer,
                       solution=e.solution) for i, e in enumerate(exs)]
        oracle.update({e.question: e for e in out})
        return out

    save_jsonl(fake_gsm(400, 1), tmp_path / "gsm_train.jsonl")
    save_jsonl(fake_gsm(40, 2), tmp_path / "gsm_test.jsonl")

    def preset(**over):
        c = P.Config(
            name="test", output_dir=str(tmp_path / "unused"), model=str(tiny_model_dir), seeds=[0],
            real_data=str(tmp_path / "gsm_train.jsonl"), eval_data={"gsm8k_test": str(tmp_path / "gsm_test.jsonl")},
            diagnose={"probe_per_cell": 6, "n_samples": 3, "bootstrap": 40, "natural_pool": 1500, "annotate": 10,
                      "grid": SMALL},
            data={"pool_size": 6000, "natural_size": 2000, "token_budget": 15000, "ratios": [3], "main_ratio": 3,
                  "grid": {"steps": [2, 3, 4, 5, 6], "digits": [1, 2, 3, 4]}},
            train={"load_in_4bit": False, "max_steps": 2, "per_device_batch_size": 2, "grad_accum": 1,
                   "target_modules": ["q_proj", "v_proj"], "save_steps": 1000, "logging_steps": 1},
            eval={"benchmark_limit": 20, "gsm_symbolic": [], "probe_per_cell": 1, "grid": SMALL,
                  "load_in_4bit": False},
            preflight={"min_free_gb": 0.01}, gpu={"required": False, "memory_fraction": 0})
        for k, v in over.items():
            setattr(c, k, v)
        return c

    presets = {"explore": preset(), "paper": preset(seeds=[0, 1, 2], data={**preset().data, "ratios": [1, 3, 9]})}
    store = Store(tmp_path / "dm.db")
    factory = lambda model, adapter, gen: SimulatedRunner(oracle, seed=len(adapter or ""), n_samples=gen.n_samples)

    def create(kind="pipeline", **req):
        base = {"mode": "explore", "model": str(tiny_model_dir), "benchmarks": BENCH} if kind == "pipeline" \
            else {"model": str(tiny_model_dir), "benchmarks": BENCH}
        return create_job_from_request(store, kind, {**base, **req}, presets=presets, known_models=[],
                                       runs_root=tmp_path / "pipelines", allow_local_models=True)

    return store, factory, create, presets, tmp_path


# ----------------------------------------------------------------- step lists
def test_paper_step_list_is_18_train_18_eval():
    store = Store(":memory:")
    jid = create_job_from_request(store, "pipeline", {"mode": "paper", "model": "Qwen/Qwen3-0.6B"},
                                  presets=load_presets())
    steps = store.get_steps(jid)
    stages = [s["stage"] for s in steps]
    assert len(steps) == 42 and stages.count("train") == 18 and stages.count("evaluate") == 18
    assert stages[:5] == ["preflight", "baseline", "diagnose", "choose_target", "build_data"] and stages[-1] == "compare"
    keys = [s["key"] for s in steps if s["stage"] == "train"][:6]
    assert keys == ["train:real_only_r0_s0", "train:untargeted_r3_s0", "train:matched_control_r3_s0",
                    "train:targeted_r1_s0", "train:targeted_r3_s0", "train:targeted_r9_s0"]
    for a, b in zip(steps[5:-1:2], steps[6:-1:2]):          # train -> evaluate, interleaved per arm key
        assert a["key"].split(":")[1] == b["key"].split(":")[1] and (a["stage"], b["stage"]) == ("train", "evaluate")
    job = store.get_job(jid)
    assert job["resolved"]["benchmark_limit"] is None and job["mode"] == "paper"
    assert job["resolved"]["config"]["name"] == f"pl_{jid}" and job["output_dir"].endswith(jid)


def test_explore_and_benchmark_run_step_lists(env):
    store, _, create, _, _ = env
    jid = create(arms=["targeted", "matched_control"], ratio=3, seeds=[0, 2])
    assert [s["key"] for s in store.get_steps(jid)][5:9] == [
        "train:matched_control_r3_s0", "evaluate:matched_control_r3_s0", "train:targeted_r3_s0",
        "evaluate:targeted_r3_s0"]
    assert len(store.get_steps(jid)) == 5 + 2 * 2 * 2 + 1
    assert [s["key"] for s in store.get_steps(create("benchmark_run"))] == ["preflight", "baseline", "diagnose"]
    assert [s["key"] for s in store.get_steps(create("benchmark_run", include_diagnosis=False))] == [
        "preflight", "baseline"]


# ----------------------------------------------------------------- validation
@pytest.mark.parametrize("req, code", [
    ({"mode": "paper", "model": "Qwen/Qwen3-0.6B", "seeds": [0]}, "paper_mode_locked"),
    ({"mode": "explore", "model": "nope/model"}, "unknown_model"),
    ({"mode": "explore", "model": "Qwen/Qwen3-0.6B", "benchmarks": ["gsm9k"]}, "unknown_benchmark"),
    ({"mode": "explore", "model": "Qwen/Qwen3-0.6B", "target": {"strategy": "feature", "feature": "x"}},
     "invalid_feature"),
    ({"mode": "explore", "model": "Qwen/Qwen3-0.6B", "target": {"strategy": "untargeted"}}, "validation_error"),
    ({"mode": "explore", "model": "Qwen/Qwen3-0.6B", "ratio": 2}, "validation_error"),
    ({"mode": "explore", "model": "Qwen/Qwen3-0.6B", "seeds": [0, 0]}, "validation_error"),
    ({"mode": "explore", "model": "Qwen/Qwen3-0.6B", "benchmark_limit": 5}, "validation_error"),
    ({"mode": "explore", "model": "Qwen/Qwen3-0.6B", "colour": "red"}, "validation_error"),
    ({"mode": "train", "model": "Qwen/Qwen3-0.6B"}, "validation_error"),
])
def test_request_validation(req, code):
    with pytest.raises(JobError) as e:
        create_job_from_request(Store(":memory:"), "pipeline", req, presets=load_presets())
    assert e.value.code == code


def test_untargeted_strategy_allows_only_untargeted_and_real_only():
    ok = create_job_from_request(Store(":memory:"), "pipeline",
                                 {"mode": "explore", "model": "Qwen/Qwen3-0.6B", "target": {"strategy": "untargeted"},
                                  "arms": ["untargeted", "real_only"]}, presets=load_presets())
    assert ok
    with pytest.raises(JobError, match="matched control needs a target"):
        create_job_from_request(Store(":memory:"), "pipeline",
                                {"mode": "explore", "model": "Qwen/Qwen3-0.6B", "target": {"strategy": "untargeted"},
                                 "arms": ["untargeted", "matched_control"]}, presets=load_presets())


def test_hashes():
    s = Store(":memory:")
    mk = lambda **kw: s.get_job(create_job_from_request(
        s, "pipeline", {"mode": "explore", "model": "Qwen/Qwen3-0.6B", **kw}, presets=load_presets()))
    a, b, c = mk(arms=["targeted"]), mk(arms=["matched_control"], seeds=[1]), mk(benchmark_limit=50)
    assert a["baseline_hash"] == b["baseline_hash"] and a["diagnosis_hash"] == b["diagnosis_hash"]
    assert a["config_hash"] != b["config_hash"]
    assert c["baseline_hash"] != a["baseline_hash"] and c["diagnosis_hash"] == a["diagnosis_hash"]


# ----------------------------------------------------------------- end to end
def test_explore_pipeline_end_to_end_and_reuse(env):
    store, factory, create, _, tmp = env
    a = create(arms=["targeted", "matched_control"])
    assert run_job(store, a, factory, log=lambda s: None) == "done"
    job, steps = store.get_job(a), store.get_steps(a)
    assert all(s["status"] == "done" for s in steps), [(s["key"], s["status"], s["error"]) for s in steps]
    out = Path(job["output_dir"])
    tgt = json.loads((out / "target.json").read_text(encoding="utf-8"))
    assert tgt["strategy"] == "auto" and tgt["feature"] == "n_carry"            # the planted weakness
    assert sorted(p.name for p in (out / "data").glob("*.jsonl")) == ["matched_control_r3_s0.jsonl",
                                                                      "targeted_r3_s0.jsonl"]
    by_key = {s["key"]: s for s in steps}
    base = by_key["baseline"]
    assert base["progress_done"] == base["progress_total"] > 0 and base["progress_unit"] == "examples"
    tr = by_key["train:targeted_r3_s0"]
    assert (tr["progress_done"], tr["progress_total"], tr["progress_unit"]) == (2, 2, "steps")
    assert [p["step"] for p in tr["metrics"]["loss_curve"]] == [1, 2] and tr["metrics"]["load_s"] > 0
    assert not list((out / "adapters" / "targeted_r3_s0").glob("checkpoint-*"))
    ev = by_key["evaluate:targeted_r3_s0"]
    assert ev["run_ids"] and store.get_run(ev["run_ids"][0])["job_id"] == a
    assert "probes_heldout_families:target_slice" in store.get_run(ev["run_ids"][0])["metrics"]
    assert store.count_runs(job_id=a) == 7          # baseline, diagnose, build-data, 2 train, 2 eval
    assert all(r["mode"] == "explore" for r in store.list_runs(job_id=a))
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert res["complete"] is True and res["mode"] == "explore" and res["paper_eligible"] is False
    assert set(res["baseline"]) == set(BENCH) and {a["arm"] for a in res["arms"]} == {"targeted", "matched_control"}
    assert any(c["a"] == "targeted" and c["b"] == "matched_control" for c in res["comparisons"])
    assert res["primary"]["status"] == "proposed" and [r["benchmark"] for r in res["primary"]["results"]] == ["gsm8k"]
    assert res["target"] == {"strategy": "auto", "feature": "n_carry", "threshold": tgt["threshold"]}
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "EXPLORE — not paper evidence" in md and "Primary result" in md and not (out / "report.json").exists()
    assert (out / "provenance.json").exists() and (out / "config.resolved.yaml").exists()

    # reuse: same model/benchmarks/limit/prompt/generation -> baseline and diagnosis are skipped
    b = create(arms=["targeted"], reuse_from=a)
    sb = {s["key"]: s for s in store.get_steps(b)}
    assert sb["baseline"]["status"] == "skipped" and sb["baseline"]["run_ids"] == base["run_ids"]
    assert sb["diagnose"]["status"] == "skipped" and sb["diagnose"]["run_ids"] == by_key["diagnose"]["run_ids"]
    assert run_job(store, b, factory, log=lambda s: None) == "done"
    res_b = json.loads((Path(store.get_job(b)["output_dir"]) / "results.json").read_text(encoding="utf-8"))
    assert res_b["baseline"] == res["baseline"]      # the reused baseline is part of the comparison
    assert store.count_runs(job_id=b) == 3          # build-data, train, eval; baseline + diagnosis are A's runs

    # a different benchmark_limit cannot reuse anything
    with pytest.raises(JobError) as e:
        create(arms=["targeted"], reuse_from=a, benchmark_limit=30)
    assert e.value.code == "reuse_mismatch"


def test_reuse_benchmark_run_without_diagnosis(env):
    store, factory, create, _, _ = env
    src = create("benchmark_run", include_diagnosis=False)
    assert run_job(store, src, factory, log=lambda s: None) == "done"
    new = create(arms=["targeted"], reuse_from=src)
    st = {s["key"]: s["status"] for s in store.get_steps(new)}
    assert st["baseline"] == "skipped" and st["diagnose"] == "pending"   # no diagnosis to reuse: run it fresh


# --------------------------------------------------------- cancel and errors
def test_cooperative_cancel_during_baseline(env):
    store, factory, create, _, _ = env
    jid = create(arms=["targeted"])
    assert execute_step(store, jid, 0, factory, log=lambda s: None) == "done"
    store.update_job(jid, cancel_requested=True)
    assert execute_step(store, jid, 1, factory, log=lambda s: None) == "cancelled"
    assert finish_job(store, jid) == "cancelled"
    job = store.get_job(jid)
    assert job["error"]["code"] == "cancelled" and job["error"]["step_key"] == "baseline"
    assert store.get_step(jid, 2)["status"] == "pending"     # later steps wait for a resume


def test_no_significant_weakness_fails_choose_target(env):
    store, factory, create, _, _ = env
    jid = create(arms=["targeted"])
    out = Path(store.get_job(jid)["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnosis.json").write_text(json.dumps({"target": None, "threshold": None, "weaknesses": []}),
                                        encoding="utf-8")
    assert execute_step(store, jid, 3, factory, log=lambda s: None) == "failed"
    err = store.get_step(jid, 3)["error"]
    assert err["code"] == "no_significant_weakness" and "untargeted" in err["message"]


def test_preflight_disk_low(env):
    store, factory, create, presets, _ = env
    presets["explore"].preflight = {"min_free_gb": 1e9}
    jid = create(arms=["targeted"])
    assert execute_step(store, jid, 0, factory, log=lambda s: None) == "failed"
    assert store.get_step(jid, 0)["error"]["code"] == "disk_low"


def test_error_mapping():
    class OutOfMemoryError(RuntimeError):
        pass

    class GatedRepoError(Exception):
        pass

    assert error_from_exception(OutOfMemoryError("CUDA out of memory"))["code"] == "out_of_memory"
    assert "batch_size" in error_from_exception(RuntimeError("CUDA out of memory. Tried"))["message"]
    assert error_from_exception(GatedRepoError("401"))["code"] == "model_access_denied"
    assert error_from_exception(RuntimeError("load_in_4bit needs a CUDA GPU"))["code"] == "gpu_unavailable"
    e = error_from_exception(KeyError("x"), "train:targeted_r3_s0")
    assert e["code"] == "internal_error" and "train:targeted_r3_s0" in e["message"] and "Traceback" not in e["message"]


def test_execute_entry_point_runs_one_step_in_a_subprocess(env):
    store, _, create, _, tmp = env
    jid = create("benchmark_run", include_diagnosis=False)
    res = subprocess.run([sys.executable, "-m", "dreammachine.jobs.execute", "--job", jid, "--step", "0",
                          "--db", str(tmp / "dm.db")], capture_output=True, text=True, encoding="utf-8",
                         cwd=Path(__file__).resolve().parents[1], timeout=300)
    assert res.returncode == 0, res.stdout + res.stderr
    assert Store(tmp / "dm.db").get_step(jid, 0)["status"] == "done"
