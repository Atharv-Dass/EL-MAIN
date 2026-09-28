import numpy as np
import pytest

from dreammachine.diagnosis import (
    LLTMResult, baseline_threshold, bootstrap_lltm, choose_target, design_matrix, fit_lltm,
    rank_weaknesses, select_matched_control, select_targeted, select_untargeted,
)
from dreammachine.generator import FEATURES
from dreammachine.generator.probes import ProbeGrid, build_pool, factorial_probe_set

TRUE_THETA = 4.0
TRUE_ETA = np.array([0.5, 0.9, 0.3])
NAMES = ["steps", "log10_max", "n_distractors"]


def _simulate(n_items: int, rng: np.random.Generator, trials: int = 1):
    Q = np.column_stack([
        rng.integers(1, 8, n_items),
        rng.uniform(0.5, 4.0, n_items),
        rng.integers(0, 3, n_items),
    ]).astype(float)
    p = 1 / (1 + np.exp(-(TRUE_THETA - Q @ TRUE_ETA)))
    return Q, rng.binomial(trials, p), np.full(n_items, trials)


def test_parameter_recovery():
    Q, s, n = _simulate(5000, np.random.default_rng(0))
    r = fit_lltm(Q, s, n, NAMES, l2=0.0)
    assert r.converged
    assert abs(r.theta - TRUE_THETA) < 4 * r.se_theta
    assert np.all(np.abs(r.eta - TRUE_ETA) < 4 * r.se_eta)
    assert np.all(r.se_eta < 0.1)
    assert 0 < r.mcfadden_r2 < 1


def test_wald_interval_coverage():
    rng = np.random.default_rng(1)
    covered = np.zeros(3)
    reps = 200
    for _ in range(reps):
        Q, s, n = _simulate(600, rng)
        r = fit_lltm(Q, s, n, NAMES, l2=0.0)
        covered += (r.ci_low <= TRUE_ETA) & (TRUE_ETA <= r.ci_high)
    coverage = covered / reps
    assert np.all(coverage > 0.89) and np.all(coverage < 0.995), coverage


def test_binomial_trials_equivalent_to_repeated_rows():
    rng = np.random.default_rng(2)
    Q, s, n = _simulate(300, rng, trials=4)
    agg = fit_lltm(Q, s, n, NAMES, l2=0.0)
    rows_Q = np.repeat(Q, 4, axis=0)
    rows_s = np.concatenate([[1] * k + [0] * (4 - k) for k in s])
    flat = fit_lltm(rows_Q, rows_s, None, NAMES, l2=0.0)
    np.testing.assert_allclose(agg.eta, flat.eta, atol=1e-6)


def test_separation_stays_finite_and_constant_columns_dropped():
    Q = np.column_stack([np.arange(20.0), np.ones(20)])
    s = (np.arange(20) < 10).astype(float)  # perfectly separated
    r = fit_lltm(Q, s, None, ["x", "const"], l2=1e-2)
    assert np.isfinite(r.eta[0]) and r.eta[0] > 0
    assert r.dropped == ["const"] and np.isnan(r.eta[1])
    assert np.isfinite(r.predict_proba(Q)).all()


def test_bootstrap_and_serialisation():
    Q, s, n = _simulate(800, np.random.default_rng(3))
    r = bootstrap_lltm(Q, s, n, NAMES, n_boot=60, seed=0)
    assert r.ci_method.startswith("bootstrap")
    assert np.all(r.ci_low <= r.eta) and np.all(r.eta <= r.ci_high)
    r2 = LLTMResult.from_dict(r.to_dict())
    np.testing.assert_allclose(r2.eta, r.eta)
    np.testing.assert_allclose(r2.predict_proba(Q[:5]), r.predict_proba(Q[:5]))


def test_input_validation():
    with pytest.raises(ValueError):
        fit_lltm(np.zeros((3, 1)), np.array([1, 0]), None)
    with pytest.raises(ValueError):
        fit_lltm(np.zeros((2, 1)), np.array([2, 0]), np.array([1, 1]))


# ----------------------------------------------- end-to-end diagnosis on generated items
@pytest.fixture(scope="module")
def simulated_diagnosis():
    """A simulated 'model' whose true weakness is carrying (n_carry)."""
    probes = factorial_probe_set(ProbeGrid(steps=(2, 3, 4, 5, 6), digits=(1, 2, 3, 4)), per_cell=4, seed=0)
    Q = design_matrix(probes, FEATURES)
    true_eta = dict(steps=0.25, log10_max=0.2, n_mul=0.1, n_div=0.1, n_carry=0.55, n_distractors=0.15,
                    has_merge=0.0)
    eta = np.array([true_eta[f] for f in FEATURES])
    rng = np.random.default_rng(0)
    p = 1 / (1 + np.exp(-(3.0 - Q @ eta)))
    s = rng.binomial(3, p)
    result = fit_lltm(Q, s, np.full(len(s), 3), list(FEATURES))
    natural = build_pool(1500, seed=1)
    pool = build_pool(3000, seed=2, grid=ProbeGrid(steps=(2, 3, 4, 5, 6, 7), digits=(1, 2, 3, 4)))
    return result, natural, pool


def test_rank_weaknesses_finds_true_factor(simulated_diagnosis):
    result, natural, _ = simulated_diagnosis
    assert "has_merge" in result.dropped  # never varied in the probe grid -> not identifiable
    ranked = rank_weaknesses(result, design_matrix(natural, result.feature_names))
    target = choose_target(ranked)
    assert target is not None and target.feature == "n_carry"
    assert all(w.feature != "has_merge" for w in ranked)


def test_targeted_and_matched_arms(simulated_diagnosis):
    result, natural, pool = simulated_diagnosis
    nat_Q = design_matrix(natural, result.feature_names)
    thr = baseline_threshold(nat_Q, result.feature_names, "n_carry")
    targeted = select_targeted(pool, result, "n_carry", thr, n=200, seed=0)
    assert len(targeted) == 200
    k = result.feature_names.index("n_carry")
    tq = design_matrix(targeted, result.feature_names)
    tp = result.predict_proba(tq)
    assert np.all(tq[:, k] > thr) and np.all((tp >= 0.3) & (tp <= 0.7))

    control, report = select_matched_control(pool, result, "n_carry", thr, targeted, seed=0)
    cq = design_matrix(control, result.feature_names)
    assert report.n_matched == 200
    assert np.all(cq[:, k] <= thr)
    assert not {c.id for c in control} & {t.id for t in targeted}
    # the point of the control: same predicted difficulty, different source of it
    assert report.mean_abs_logit_diff < 0.1 and report.ks_statistic < 0.15
    assert tq[:, k].mean() > cq[:, k].mean() + 1

    untargeted = select_untargeted(natural, 200, seed=0)
    assert len({u.id for u in untargeted}) == 200
