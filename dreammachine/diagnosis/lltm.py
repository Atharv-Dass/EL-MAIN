"""Linear Logistic Test Model (Fischer, 1973) for per-factor weakness attribution.

    logit P(correct_i) = theta - sum_k q_ik * eta_k

theta is the model's ability. eta_k is the log-odds cost of one unit of factor k
(one more step, one more order of magnitude, one more carry, ...). The q_ik are
exact item features (generator/features.py), so eta is an attribution over
controlled factors rather than a correlation over benchmark tags.

The fit is penalised maximum likelihood by Newton-Raphson (IRLS). A small ridge
on eta only keeps estimates finite under separation. Uncertainty comes from the
observed Fisher information (Wald) or an item bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_Z95 = 1.959963984540054


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1 / (1 + np.exp(-np.clip(z, -500, 500))),
                    np.exp(np.clip(z, -500, 500)) / (1 + np.exp(np.clip(z, -500, 500))))


@dataclass
class LLTMResult:
    feature_names: list[str]
    theta: float
    eta: np.ndarray
    se_theta: float
    se_eta: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    ci_method: str
    n_items: int
    n_obs: int
    loglik: float
    loglik_null: float
    converged: bool
    iterations: int
    dropped: list[str] = field(default_factory=list)  # constant columns: not identifiable

    def logit(self, Q: np.ndarray) -> np.ndarray:
        Q = np.atleast_2d(np.asarray(Q, dtype=float))
        eta = np.nan_to_num(self.eta, nan=0.0)
        return self.theta - Q @ eta

    def predict_proba(self, Q: np.ndarray) -> np.ndarray:
        return _sigmoid(self.logit(Q))

    @property
    def mcfadden_r2(self) -> float:
        return 1.0 - self.loglik / self.loglik_null if self.loglik_null < 0 else float("nan")

    def to_dict(self) -> dict:
        return {
            "feature_names": self.feature_names,
            "theta": self.theta,
            "se_theta": self.se_theta,
            "eta": self.eta.tolist(),
            "se_eta": self.se_eta.tolist(),
            "ci_low": self.ci_low.tolist(),
            "ci_high": self.ci_high.tolist(),
            "ci_method": self.ci_method,
            "n_items": self.n_items,
            "n_obs": self.n_obs,
            "loglik": self.loglik,
            "loglik_null": self.loglik_null,
            "mcfadden_r2": self.mcfadden_r2,
            "converged": self.converged,
            "iterations": self.iterations,
            "dropped": self.dropped,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LLTMResult":
        return cls(
            feature_names=list(d["feature_names"]), theta=d["theta"], se_theta=d["se_theta"],
            eta=np.array(d["eta"], dtype=float), se_eta=np.array(d["se_eta"], dtype=float),
            ci_low=np.array(d["ci_low"], dtype=float), ci_high=np.array(d["ci_high"], dtype=float),
            ci_method=d["ci_method"], n_items=d["n_items"], n_obs=d["n_obs"], loglik=d["loglik"],
            loglik_null=d["loglik_null"], converged=d["converged"], iterations=d["iterations"],
            dropped=list(d.get("dropped", [])),
        )


def _loglik(z: np.ndarray, s: np.ndarray, n: np.ndarray) -> float:
    return float(np.sum(s * z - n * np.logaddexp(0.0, z)))


def _newton(X: np.ndarray, s: np.ndarray, n: np.ndarray, penalty: np.ndarray,
            max_iter: int, tol: float) -> tuple[np.ndarray, np.ndarray, bool, int]:
    w = np.zeros(X.shape[1])
    p0 = np.clip(s.sum() / n.sum(), 1e-6, 1 - 1e-6)
    w[0] = np.log(p0 / (1 - p0))  # start at the marginal log-odds

    def objective(w_: np.ndarray) -> float:
        return -_loglik(X @ w_, s, n) + 0.5 * float(np.sum(penalty * w_ * w_))

    obj = objective(w)
    for it in range(1, max_iter + 1):
        z = X @ w
        p = _sigmoid(z)
        grad = -X.T @ (s - n * p) + penalty * w
        H = (X * (n * p * (1 - p))[:, None]).T @ X + np.diag(penalty)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        t = 1.0
        while t > 1e-10:  # backtracking line search keeps Newton monotone
            w_new = w - t * step
            obj_new = objective(w_new)
            if obj_new <= obj + 1e-12:
                break
            t *= 0.5
        w, obj_prev, obj = w_new, obj, obj_new
        if np.max(np.abs(t * step)) < tol or abs(obj_prev - obj) < tol * (1 + abs(obj)):
            z = X @ w
            p = _sigmoid(z)
            H = (X * (n * p * (1 - p))[:, None]).T @ X + np.diag(penalty)
            return w, H, True, it
    return w, H, False, max_iter


def fit_lltm(Q: np.ndarray, successes: np.ndarray, trials: np.ndarray | None = None,
             feature_names: list[str] | None = None, l2: float = 1e-2,
             max_iter: int = 100, tol: float = 1e-9) -> LLTMResult:
    """Fit the LLTM. Q: (items, K) features; successes/trials: per-item counts."""
    Q = np.asarray(Q, dtype=float)
    s = np.asarray(successes, dtype=float)
    n = np.ones_like(s) if trials is None else np.asarray(trials, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != s.shape[0] or s.shape != n.shape:
        raise ValueError("shape mismatch between Q, successes and trials")
    if np.any(s < 0) or np.any(s > n) or np.any(n <= 0):
        raise ValueError("need 0 <= successes <= trials and trials > 0")
    names = list(feature_names) if feature_names else [f"f{k}" for k in range(Q.shape[1])]

    keep = np.ptp(Q, axis=0) > 0
    dropped = [nm for nm, k in zip(names, keep) if not k]
    Qk = Q[:, keep]
    X = np.hstack([np.ones((Q.shape[0], 1)), -Qk])
    penalty = np.r_[0.0, np.full(Qk.shape[1], l2)]
    w, H, converged, iters = _newton(X, s, n, penalty, max_iter, tol)

    cov = np.linalg.pinv(H)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    K = Q.shape[1]
    eta = np.full(K, np.nan)
    se_eta = np.full(K, np.nan)
    eta[keep] = w[1:]
    se_eta[keep] = se[1:]

    p_bar = np.clip(s.sum() / n.sum(), 1e-12, 1 - 1e-12)
    ll_null = float(np.sum(s * np.log(p_bar) + (n - s) * np.log(1 - p_bar)))
    return LLTMResult(
        feature_names=names, theta=float(w[0]), eta=eta, se_theta=float(se[0]), se_eta=se_eta,
        ci_low=eta - _Z95 * se_eta, ci_high=eta + _Z95 * se_eta, ci_method="wald",
        n_items=int(Q.shape[0]), n_obs=int(n.sum()), loglik=_loglik(X @ w, s, n),
        loglik_null=ll_null, converged=converged, iterations=iters, dropped=dropped,
    )


def bootstrap_lltm(Q: np.ndarray, successes: np.ndarray, trials: np.ndarray | None = None,
                   feature_names: list[str] | None = None, n_boot: int = 200, alpha: float = 0.05,
                   seed: int = 0, l2: float = 1e-2) -> LLTMResult:
    """Point estimate from the full data, with percentile CIs from resampling items."""
    Q = np.asarray(Q, dtype=float)
    s = np.asarray(successes, dtype=float)
    n = np.ones_like(s) if trials is None else np.asarray(trials, dtype=float)
    base = fit_lltm(Q, s, n, feature_names, l2=l2)
    rng = np.random.default_rng(seed)
    draws = np.full((n_boot, Q.shape[1]), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, Q.shape[0], Q.shape[0])
        r = fit_lltm(Q[idx], s[idx], n[idx], feature_names, l2=l2)
        draws[b] = r.eta
    base.ci_low = np.nanpercentile(draws, 100 * alpha / 2, axis=0)
    base.ci_high = np.nanpercentile(draws, 100 * (1 - alpha / 2), axis=0)
    base.ci_method = f"bootstrap(n={n_boot})"
    return base
