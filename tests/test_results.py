"""Comparison + results (PLAN.md milestone B6) on stored fake eval runs with known answers."""

import json

import pytest

from dreammachine.experiments.evaluate import ResponseRecord
from dreammachine.generator import FEATURES
from dreammachine.jobs import JobError
from dreammachine.jobs.results import build_results, compare_runs, results_markdown, write_results
from dreammachine.store import Store

N_GSM, N_PROBE = 40, 21
ETA_BASE = {f: 0.5 for f in FEATURES} | {"n_carry": 0.6}


def _records(gsm_correct, probe_correct, wrong_type):
    recs = [ResponseRecord(f"g{i:02d}", "gsm8k", 0, "t", gsm_correct(i), "CORRECT" if gsm_correct(i) else wrong_type,
                           {}, {}) for i in range(N_GSM)]
    recs += [ResponseRecord(f"p{i:02d}", "probes_heldout_families", 0, "t", probe_correct(i),
                            "CORRECT" if probe_correct(i) else wrong_type, {},
                            {f: 0.0 for f in FEATURES} | {"n_carry": float(i % 3)}) for i in range(N_PROBE)]
    return recs


def _eval_run(store, job_id, arm, ratio, seed, recs, eta, precision=False):
    run = store.create_run(f"x:eval:{arm}", "eval", {"arm": arm, "ratio": ratio, "seed": seed, "model": "m",
                                                     "prompt_version": "qwen_boxed", "eval_load_in_4bit": precision},
                           job_id=job_id, mode="explore")
    store.add_responses(run, recs)
    store.finish_run(run, metrics={"eta": eta})
    return run


@pytest.fixture
def job(tmp_path):
    """A pipeline with a baseline and 4 arms. Base: half of GSM8K right, probes right only below the threshold.
    targeted r3: worse on GSM8K (regression), all probes right, n_carry eta down, slips instead of plan errors.
    matched_control r3: same answers as base. real_only: everything right. targeted r1: same as base."""
    store = Store(":memory:")
    out = tmp_path / "job"
    out.mkdir()
    (out / "target.json").write_text(json.dumps({"strategy": "auto", "feature": "n_carry", "threshold": 1.0}),
                                     encoding="utf-8")
    jid = store.create_job("pipeline", "fake", "explore", {"mode": "explore", "model": "m"}, output_dir=str(out),
                           resolved={"model": "m", "benchmarks": ["gsm8k", "probes_heldout_families"],
                                     "main_ratio": 3.0, "seeds": [0, 1], "prompt_version": "qwen_boxed"})
    base_ok = (lambda i: i % 2 == 0, lambda i: i % 3 < 2)
    base = _eval_run(store, jid, "base", None, None, _records(*base_ok, "PLAN_ERROR"), ETA_BASE)
    steps = [{"key": "baseline", "stage": "baseline"}]
    runs = {}
    plan = [("targeted", 3.0, (lambda i: i % 4 == 0, lambda i: True), "ARITHMETIC_SLIP", [0.4, 0.5]),
            ("matched_control", 3.0, base_ok, "PLAN_ERROR", [0.6, 0.6]),
            ("real_only", 0.0, (lambda i: True, lambda i: True), "PLAN_ERROR", [0.5, 0.5]),
            ("targeted", 1.0, base_ok, "PLAN_ERROR", [0.6, 0.6])]
    for arm, ratio, ok, wrong, etas in plan:
        for seed in (0, 1):
            steps.append({"key": f"evaluate:{arm}_r{ratio:g}_s{seed}", "stage": "evaluate"})
            runs[(arm, ratio, seed)] = _eval_run(store, jid, arm, ratio, seed, _records(*ok, wrong),
                                                 ETA_BASE | {"n_carry": etas[seed]})
    steps.append({"key": "compare", "stage": "compare"})
    store.set_steps(jid, steps)
    store.update_step(jid, 0, status="done", run_ids=[base])
    for i, s in enumerate(steps[1:-1], start=1):
        arm_r_s = s["key"].split(":")[1]
        arm, r, sd = arm_r_s.rsplit("_", 2)
        store.update_step(jid, i, status="done", run_ids=[runs[(arm, float(r[1:]), int(sd[1:]))]])
    return store, jid, base, runs


def _arm(res, arm, ratio):
    return next(a for a in res["arms"] if a["arm"] == arm and a["ratio"] == ratio)


def test_before_vs_after_and_regressions(job):
    store, jid, _, _ = job
    res = build_results(store, jid)
    assert res["complete"] is False and res["paper_eligible"] is False      # compare step not done; explore
    assert res["baseline"] == {"gsm8k": {"accuracy": 0.5, "n": N_GSM},
                               "probes_heldout_families": {"accuracy": 14 / 21, "n": N_PROBE}}
    t = _arm(res, "targeted", 3)
    g = t["benchmarks"]["gsm8k"]
    assert g["accuracy_mean"] == 0.25 and g["accuracy_sd"] == 0.0 and g["n_seeds"] == 2 and t["seeds"] == [0, 1]
    assert g["vs_base"]["mean_diff"] == -0.25 and g["vs_base"]["ci_high"] < 0 and g["regression"] is True
    assert _arm(res, "matched_control", 3)["benchmarks"]["gsm8k"]["regression"] is False
    assert _arm(res, "real_only", 0)["benchmarks"]["gsm8k"]["vs_base"]["mean_diff"] == 0.5
    assert res["regressions"] == [{"arm": "targeted", "ratio": 3, "benchmark": "gsm8k", "mean_diff": -0.25,
                                   "ci_low": g["vs_base"]["ci_low"], "ci_high": g["vs_base"]["ci_high"]}]


def test_arm_pairs_only_at_the_same_ratio(job):
    store, jid, _, _ = job
    pairs = {(c["a"], c["b"], c["ratio"], c["benchmark"]) for c in build_results(store, jid)["comparisons"]}
    assert ("targeted", "matched_control", 3, "gsm8k") in pairs
    assert ("targeted", "matched_control", 1, "gsm8k") not in pairs           # no matched_control at ratio 1
    assert {("targeted", "real_only", 1, "gsm8k"), ("targeted", "real_only", 3, "gsm8k")} <= pairs  # real_only: any
    c = next(c for c in build_results(store, jid)["comparisons"]
             if (c["a"], c["b"], c["ratio"], c["benchmark"]) == ("targeted", "matched_control", 3, "gsm8k"))
    assert c["mean_diff"] == -0.25 and c["significant"] is True and c["n"] == N_GSM


def test_eta_change_sign_error_shift_and_target_slice(job):
    store, jid, _, _ = job
    t = _arm(build_results(store, jid), "targeted", 3)
    assert t["eta_change"]["n_carry"] == pytest.approx(-0.15)     # mean(0.4, 0.5) - 0.6: hurts LESS after training
    assert t["eta_change"]["steps"] == 0.0 and set(t["eta_change"]) == set(FEATURES)
    assert t["error_type_shift"] == {"ARITHMETIC_SLIP": 1.0, "PLAN_ERROR": -1.0}
    s = t["target_slice"]["probes_heldout_families"]
    assert s["above"] == {"before": 0.0, "after": 1.0, "n": 7}              # n_carry = 2 > threshold 1
    assert s["at_or_below"] == {"before": 1.0, "after": 1.0, "n": 14}


def test_primary_block(job):
    store, jid, _, _ = job
    p = build_results(store, jid)["primary"]
    assert p["status"] == "proposed" and (p["a"], p["b"]) == ("targeted", "matched_control")
    assert p["benchmarks"] == ["gsm8k"] and [r["benchmark"] for r in p["results"]] == ["gsm8k"]
    assert p["results"][0]["mean_diff"] == -0.25 and p["results"][0]["significant"] is True


def test_primary_is_null_without_matched_control(job):
    store, jid, _, runs = job
    for s in store.get_steps(jid):
        if "matched_control" in s["key"]:
            store.update_step(jid, s["idx"], status="pending", run_ids=[])
    assert build_results(store, jid)["primary"] is None


def test_paper_eligibility(job):
    store, jid, _, _ = job
    store.update_step(jid, len(store.get_steps(jid)) - 1, status="done")
    assert build_results(store, jid)["complete"] is True
    assert build_results(store, jid)["paper_eligible"] is False                 # explore: never paper evidence
    store._conn.execute("UPDATE jobs SET mode='paper', request=? WHERE id=?",
                        (json.dumps({"mode": "paper", "model": "m"}), jid))
    store.update_job(jid, provenance={"git": {"commit": "abc", "dirty": False}})
    assert build_results(store, jid)["paper_eligible"] is True
    store.update_job(jid, provenance={"git": {"commit": "abc", "dirty": True}})
    res = build_results(store, jid)
    assert res["paper_eligible"] is False and any("dirty" in w for w in res["warnings"])


def test_not_ready_and_not_found(job):
    store, jid, _, _ = job
    other = store.create_job("pipeline", "empty", "explore", {}, resolved={"benchmarks": ["gsm8k"]})
    store.set_steps(other, [{"key": "baseline", "stage": "baseline"}])
    with pytest.raises(JobError) as e:
        build_results(store, other)
    assert e.value.code == "not_ready"
    with pytest.raises(JobError) as e:
        build_results(store, "nope")
    assert e.value.code == "not_found"


def test_compare_runs(job):
    store, jid, base, runs = job
    c = compare_runs(store, base, runs[("targeted", 3.0, 0)])
    g = next(b for b in c["benchmarks"] if b["name"] == "gsm8k")
    assert (g["before"], g["after"], g["mean_diff"], g["regression"]) == (0.5, 0.25, -0.25, True)
    assert c["eta_change"]["n_carry"] == pytest.approx(-0.2) and c["warnings"] == []
    four_bit = _eval_run(store, jid, "targeted", 3.0, 0, [], ETA_BASE, precision=True)
    assert any("precision" in w for w in compare_runs(store, base, four_bit)["warnings"])
    diag = store.create_run("x:diagnose", "diagnose", {})
    with pytest.raises(JobError) as e:
        compare_runs(store, base, diag)
    assert e.value.code == "validation_error"


def test_write_results_and_markdown(job):
    store, jid, _, _ = job
    res = write_results(store, jid)
    out = store.get_job(jid)["output_dir"]
    assert json.loads(open(f"{out}/results.json", encoding="utf-8").read()) == res and res["complete"] is True
    md = open(f"{out}/report.md", encoding="utf-8").read()
    assert "EXPLORE — not paper evidence" in md and "Regressions" in md and "−0.150" not in md
    assert "| targeted | 3 | " in md and "-0.150" in md      # eta change row
    assert results_markdown(res, store.get_job(jid)).startswith("# DreamMachine results — fake")
