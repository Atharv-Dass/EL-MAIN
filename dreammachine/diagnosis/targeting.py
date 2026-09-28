"""Weakness selection and the three synthetic arms (docs/RESEARCH.md §3.3-3.5).

    targeted         target factor elevated, predicted p in [0.3, 0.7], weighted by p(1-p)
    matched control  target factor at baseline, same predicted-difficulty distribution
    untargeted       drawn from the natural distribution

Only the matched control separates "targeting helps" from "harder data helps".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import ks_2samp

from ..generator.features import FEATURES
from ..generator.sampler import Item
from .lltm import LLTMResult


def design_matrix(items: list[Item], names: tuple[str, ...] | list[str] = FEATURES) -> np.ndarray:
    return np.array([[float(it.features[n]) for n in names] for it in items], dtype=float)


@dataclass
class Weakness:
    feature: str
    eta: float
    ci_low: float
    ci_high: float
    range: float     # p90 - p10 of the feature on the natural distribution
    effect: float    # eta * range: log-odds lost across the feature's typical span
    significant: bool

    def to_dict(self) -> dict:
        return asdict(self)


def rank_weaknesses(result: LLTMResult, natural_Q: np.ndarray) -> list[Weakness]:
    natural_Q = np.asarray(natural_Q, dtype=float)
    out = []
    for k, name in enumerate(result.feature_names):
        eta = float(result.eta[k])
        if np.isnan(eta):
            continue
        lo, hi = np.percentile(natural_Q[:, k], [10, 90])
        rng = float(hi - lo) or float(np.ptp(natural_Q[:, k]))
        out.append(Weakness(
            feature=name, eta=eta, ci_low=float(result.ci_low[k]), ci_high=float(result.ci_high[k]),
            range=rng, effect=eta * rng, significant=bool(result.ci_low[k] > 0),
        ))
    return sorted(out, key=lambda w: (w.significant, w.effect), reverse=True)


def choose_target(weaknesses: list[Weakness]) -> Weakness | None:
    """The significant weakness with the largest range-normalised effect, or None."""
    for w in weaknesses:
        if w.significant and w.effect > 0:
            return w
    return None


def baseline_threshold(natural_Q: np.ndarray, feature_names: list[str], target: str) -> float:
    """Items above this value count as 'elevated' on the target factor."""
    k = list(feature_names).index(target)
    return float(np.median(np.asarray(natural_Q)[:, k]))


def _predicted(items: list[Item], result: LLTMResult) -> tuple[np.ndarray, np.ndarray]:
    z = result.logit(design_matrix(items, result.feature_names))
    return z, 1 / (1 + np.exp(-z))


def select_targeted(pool: list[Item], result: LLTMResult, target: str, threshold: float, n: int,
                    p_band: tuple[float, float] = (0.3, 0.7), seed: int = 0) -> list[Item]:
    k = result.feature_names.index(target)
    Q = design_matrix(pool, result.feature_names)
    _, p = _predicted(pool, result)
    mask = (Q[:, k] > threshold) & (p >= p_band[0]) & (p <= p_band[1])
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    info = p[idx] * (1 - p[idx])
    take = min(n, len(idx))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(idx, size=take, replace=False, p=info / info.sum())
    return [pool[i] for i in chosen]


@dataclass
class MatchReport:
    n_requested: int
    n_matched: int
    mean_abs_logit_diff: float
    max_abs_logit_diff: float
    ks_statistic: float
    ks_pvalue: float

    def to_dict(self) -> dict:
        return asdict(self)


def select_matched_control(pool: list[Item], result: LLTMResult, target: str, threshold: float,
                           targeted: list[Item], seed: int = 0) -> tuple[list[Item], MatchReport]:
    """Greedy nearest-logit matching without replacement from items with the target at baseline."""
    k = result.feature_names.index(target)
    taken = {it.id for it in targeted}
    cand = [it for it in pool if it.id not in taken]
    Qc = design_matrix(cand, result.feature_names)
    keep = Qc[:, k] <= threshold
    cand = [it for it, ok in zip(cand, keep) if ok]
    if not cand or not targeted:
        return [], MatchReport(len(targeted), 0, float("nan"), float("nan"), float("nan"), float("nan"))
    zc, _ = _predicted(cand, result)
    zt, pt = _predicted(targeted, result)
    available = np.ones(len(cand), dtype=bool)
    order = np.random.default_rng(seed).permutation(len(targeted))
    matched_idx, diffs = [], []
    for i in order:
        if not available.any():
            break
        d = np.where(available, np.abs(zc - zt[i]), np.inf)
        j = int(np.argmin(d))
        available[j] = False
        matched_idx.append(j)
        diffs.append(float(d[j]))
    matched = [cand[j] for j in matched_idx]
    pm = 1 / (1 + np.exp(-zc[matched_idx]))
    ks = ks_2samp(pt, pm)
    return matched, MatchReport(
        n_requested=len(targeted), n_matched=len(matched), mean_abs_logit_diff=float(np.mean(diffs)),
        max_abs_logit_diff=float(np.max(diffs)), ks_statistic=float(ks.statistic),
        ks_pvalue=float(ks.pvalue),
    )


def select_untargeted(natural_pool: list[Item], n: int, seed: int = 0) -> list[Item]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(natural_pool), size=min(n, len(natural_pool)), replace=False)
    return [natural_pool[i] for i in idx]
