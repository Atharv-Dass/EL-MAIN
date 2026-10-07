"""B4 refactor guard: build_data must write byte-identical training files to the pre-split version (same seeds).
The pre-split function is frozen below; it calls only helpers whose behaviour the split does not change."""

import json
import math

import numpy as np
import pytest

from dreammachine.data import Example, from_items, save_jsonl
from dreammachine.diagnosis.lltm import LLTMResult, fit_lltm
from dreammachine.diagnosis.targeting import (
    baseline_threshold, design_matrix, select_matched_control, select_targeted, select_untargeted,
)
from dreammachine.experiments import pipeline as P
from dreammachine.generator import FEATURES, GenSpec, generate_many
from dreammachine.generator.probes import build_pool, factorial_probe_set
from dreammachine.data.mixer import mix, to_train, whitespace_tokens
from dreammachine.store import Store

PLANTED = dict(steps=0.3, log10_max=0.2, n_mul=0.1, n_div=0.1, n_carry=0.6, n_distractors=0.1, has_merge=0.0)


def _legacy_build_data(cfg, store):
    """build_data as it was before the B4 split (commit ded1727), with one deliberate later change: the number of
    items to select comes from P.synthetic_need (B8). Pool growth (data.pool_max) is not copied: the test config is
    sized so the first pool is enough, and test_pool_grows_when_band_is_short covers growth."""
    dz = cfg.data
    diag = json.loads((cfg.out / "diagnosis.json").read_text(encoding="utf-8"))
    if not diag["target"]:
        raise RuntimeError("diagnosis found no significant weakness; nothing to target")
    lltm = LLTMResult.from_dict(diag["lltm"])
    target, threshold = diag["target"]["feature"], diag["threshold"]
    budget = int(dz.get("token_budget", 600_000))
    ratios = [float(r) for r in dz.get("ratios", [3])]
    p_band = tuple(dz.get("p_band", [0.3, 0.7]))
    exclude = set(diag["probe_ids"])
    real = [to_train(e) for e in P._real_train(cfg)]
    manifest = {"target": target, "threshold": threshold, "budget_tokens": budget, "arms": {}}
    for seed in cfg.seeds:
        pool = build_pool(dz.get("pool_size", 40_000), seed=100 + seed, grid=cfg.grid(dz), exclude_ids=exclude)
        natural = build_pool(dz.get("natural_size", 20_000), seed=200 + seed, exclude_ids=exclude)
        max_syn_tokens = budget * max(ratios) / (1 + max(ratios))
        # B8 (deliberate change, not part of the refactor): size the selection by the shorter side's mean length.
        need = P.synthetic_need(pool, max_syn_tokens, target, threshold)
        targeted = select_targeted(pool, lltm, target, threshold, n=need, p_band=p_band, seed=seed)
        control, match = select_matched_control(pool, lltm, target, threshold, targeted, seed=seed)
        untargeted = select_untargeted(natural, need, seed=seed)
        if len(targeted) < need:
            manifest.setdefault("warnings", []).append(
                f"seed {seed}: only {len(targeted)}/{need} targeted items; increase data.pool_size")
        syn = {"targeted": [to_train(e) for e in from_items(targeted)],
               "matched_control": [to_train(e) for e in from_items(control)],
               "untargeted": [to_train(e) for e in from_items(untargeted)]}
        manifest["arms"][f"match_s{seed}"] = match.to_dict()
        path = P._arm_path(cfg, "real_only", 0, seed)
        rows = mix(real, [], 0, budget, seed=seed)
        P._write_train(rows, path)
        manifest["arms"][path.stem] = P._arm_stats(rows)
        for arm, data in syn.items():
            for ratio in ratios:
                rows = mix(real, data, ratio, budget, seed=seed)
                path = P._arm_path(cfg, arm, ratio, seed)
                P._write_train(rows, path)
                manifest["arms"][path.stem] = P._arm_stats(rows)
    (cfg.out / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _diagnosis(cfg, probe_seed=0):
    """A diagnosis.json with an LLTM fitted to planted weaknesses (no model needed)."""
    probes = factorial_probe_set(cfg.grid(cfg.diagnose), per_cell=3, seed=probe_seed)
    Q = design_matrix(probes, FEATURES)
    p = 1 / (1 + np.exp(-(3.0 - Q @ np.array([PLANTED[f] for f in FEATURES]))))
    s = np.random.default_rng(0).binomial(3, p).astype(float)
    lltm = fit_lltm(Q, s, np.full(len(probes), 3.0), list(FEATURES))
    natural = build_pool(800, seed=1)
    thr = baseline_threshold(design_matrix(natural, FEATURES), list(FEATURES), "n_carry")
    return {"run_id": "x", "model": "m", "summary": {}, "lltm": lltm.to_dict(), "weaknesses": [],
            "target": {"feature": "n_carry", "eta": 0.6, "ci_low": 0.4, "ci_high": 0.8, "range": 2.0,
                       "effect": 1.2, "significant": True},
            "threshold": thr, "probe_ids": sorted(it.id for it in probes)}


def _cfg(root, name):
    exs = from_items(generate_many(GenSpec(steps=3, digits=2), 300, seed=9))
    real = root / "real.jsonl"
    if not real.exists():
        save_jsonl([Example(id=f"r{i}", source="gsm8k", question=e.question, answer=e.answer, solution=e.solution)
                    for i, e in enumerate(exs)], real)
    small = {"steps": [2, 4], "digits": [1, 2], "distractors": [0], "op_profiles": ["addsub", "mixed"]}
    return P.Config(name=name, output_dir=str(root / name), seeds=[0, 1], real_data=str(real),
                    diagnose={"grid": small},
                    data={"pool_size": 2500, "natural_size": 1200, "token_budget": 6000, "ratios": [1, 3],
                          "main_ratio": 3, "grid": {"steps": [2, 3, 4, 5], "digits": [1, 2, 3]}})


@pytest.fixture
def two_cfgs(tmp_path):
    old, new = _cfg(tmp_path, "old"), _cfg(tmp_path, "new")
    diag = _diagnosis(old)
    for c in (old, new):
        (c.out / "diagnosis.json").write_text(json.dumps(diag), encoding="utf-8")
    return old, new


def test_build_data_is_byte_identical_after_split(two_cfgs):
    old, new = two_cfgs
    m_old = _legacy_build_data(old, None)
    m_new = P.build_data(new, Store(":memory:"))
    files_old = sorted(p.name for p in (old.out / "data").glob("*.jsonl"))
    files_new = sorted(p.name for p in (new.out / "data").glob("*.jsonl"))
    assert files_old == files_new and len(files_old) == 2 * (1 + 3 * 2)   # per seed: real_only + 3 arms x 2 ratios
    for name in files_old:
        assert (old.out / "data" / name).read_bytes() == (new.out / "data" / name).read_bytes(), name
    assert (old.out / "data" / "manifest.json").read_bytes() == (new.out / "data" / "manifest.json").read_bytes()
    assert m_old == m_new


def test_build_arm_matches_build_data(two_cfgs):
    old, new = two_cfgs
    P.build_data(old, Store(":memory:"))
    diag = json.loads((new.out / "diagnosis.json").read_text(encoding="utf-8"))
    stats = P.build_arm(new, "matched_control", 3.0, 1, "n_carry", diag["threshold"],
                        LLTMResult.from_dict(diag["lltm"]))
    name = "matched_control_r3_s1.jsonl"
    assert (new.out / "data" / name).read_bytes() == (old.out / "data" / name).read_bytes()
    assert stats["n"] > 0 and stats["n_synthetic"] > stats["n_real"]


def test_explore_subset_and_untargeted(two_cfgs, tmp_path):
    old, new = two_cfgs
    P.build_data(old, Store(":memory:"))
    m = P.build_data(new, Store(":memory:"), arms=[("targeted", 3.0), ("matched_control", 3.0)])
    assert sorted(p.name for p in (new.out / "data").glob("*.jsonl")) == [
        "matched_control_r3_s0.jsonl", "matched_control_r3_s1.jsonl", "targeted_r3_s0.jsonl", "targeted_r3_s1.jsonl"]
    for name in ("targeted_r3_s0.jsonl", "matched_control_r3_s1.jsonl"):   # same bytes as the full build
        assert (new.out / "data" / name).read_bytes() == (old.out / "data" / name).read_bytes()
    assert "match_s0" in m["arms"]
    # untargeted strategy: no target, so only untargeted / real_only can be built
    u = _cfg(tmp_path, "untargeted")
    (u.out / "diagnosis.json").write_text((new.out / "diagnosis.json").read_text(encoding="utf-8"), encoding="utf-8")
    (u.out / "target.json").write_text(json.dumps({"strategy": "untargeted", "feature": None, "threshold": None}),
                                       encoding="utf-8")
    mu = P.build_data(u, Store(":memory:"), arms=[("untargeted", 3.0), ("real_only", 0.0)])
    assert mu["target"] is None and "match_s0" not in mu["arms"] and "untargeted_r3_s0" in mu["arms"]
    with pytest.raises(ValueError, match="needs a target"):
        P.build_data(u, Store(":memory:"), arms=[("matched_control", 3.0)])


def _steps_cfg(tmp_path, name, pool_size, **data):
    cfg = _cfg(tmp_path, name)
    cfg.seeds = [0]
    cfg.data = {"pool_size": pool_size, "natural_size": 1200, "token_budget": 6000, "ratios": [3], "main_ratio": 3,
                "grid": {"steps": [2, 3, 4, 5, 6], "digits": [1, 2, 3]}, **data}
    diag = _diagnosis(cfg)
    natural = build_pool(800, seed=1)
    diag["target"] = {"feature": "steps", "eta": 0.5, "ci_low": 0.3, "ci_high": 0.7, "range": 3.0, "effect": 1.5,
                      "significant": True}
    diag["threshold"] = baseline_threshold(design_matrix(natural, FEATURES), list(FEATURES), "steps")
    (cfg.out / "diagnosis.json").write_text(json.dumps(diag), encoding="utf-8")
    return cfg


def test_every_arm_fills_the_budget_when_control_items_are_shorter(tmp_path):
    # B8: target `steps` -> control items (few steps) are much shorter than targeted ones
    cfg = _steps_cfg(tmp_path, "steps", pool_size=4000)
    m = P.build_data(cfg, Store(":memory:"), arms=[("targeted", 3.0), ("matched_control", 3.0)])
    for key in ("targeted_r3_s0", "matched_control_r3_s0"):
        assert m["arms"][key]["tokens"] > 0.95 * 6000, (key, m["arms"][key])
    assert "warnings" not in m


def test_pool_grows_when_band_is_short(tmp_path):
    cfg = _steps_cfg(tmp_path, "grow", pool_size=300)             # far too small on its own
    m = P.build_data(cfg, Store(":memory:"), arms=[("targeted", 3.0), ("matched_control", 3.0)])
    assert m["arms"]["matched_control_r3_s0"]["tokens"] > 0.95 * 6000 and "warnings" not in m


def test_impossible_budget_fails_with_a_clear_message(tmp_path):
    from dreammachine.errors import InsufficientData
    from dreammachine.jobs.errors import error_from_exception

    cfg = _steps_cfg(tmp_path, "tiny", pool_size=100, pool_max=100)
    with pytest.raises(InsufficientData, match="pool_size") as e:
        P.build_data(cfg, Store(":memory:"), arms=[("targeted", 3.0), ("matched_control", 3.0)])
    err = error_from_exception(e.value, "build_data")
    assert err["code"] == "internal_error" and "not enough synthetic data" in err["message"]
