"""harness_audit: a standalone multiplicity audit for a grid of backtested configurations.

Drop this file next to your analysis and call `audit(R)` with `R` the (n_configs, T) matrix of
daily returns your grid already produced. It needs numpy and scipy and nothing else, and it makes
no assumption about what the configurations are.

    import numpy as np, harness_audit
    R = np.load("grid_returns.npy")          # (n_configs, T)
    print(harness_audit.report(R))

What it gives you:
    n_eff          effective number of independent trials in the grid (participation ratio)
    n_eff_li_ji    a second estimate, from the eigenvalue count of Li and Ji (2005)
    e_max          the expected best-of-grid Sharpe under the null, from equation (1) of
                   Bailey and Lopez de Prado (2014), evaluated at N = 1, n_eff and n_configs
    dsr            the deflated Sharpe ratio of the best configuration at each of those N
    selection      best minus median Sharpe: what the search bought you, before deflation

The honest caveat, stated here because it belongs next to the number: equation (1) is derived for
independent Gaussian trials. `n_eff` is a spectral heuristic for the dependence, not a proof that
the maximum of your correlated grid behaves like the maximum of `n_eff` independent draws. If you
can afford it, run a permutation null over your own grid and compare; `implied_n` below reports the
trial count that would make equation (1) reproduce a measured null maximum, so you can check.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

GAMMA = 0.5772156649015329
ANN = math.sqrt(252.0)


def _standardise(R: np.ndarray):
    Z = np.asarray(R, dtype=np.float64)
    Z = Z - Z.mean(axis=1, keepdims=True)
    sd = Z.std(axis=1, ddof=0)
    live = sd > 0
    if not live.any():
        raise ValueError("every configuration has zero variance")
    return Z[live] / sd[live][:, None]


def eigenvalues(R: np.ndarray) -> np.ndarray:
    """All nonzero eigenvalues of the configuration correlation matrix, scaled to sum to n."""
    Z = _standardise(R)
    n, T = Z.shape
    lam = np.clip(np.linalg.eigvalsh((Z.T @ Z) / T), 0.0, None)[::-1]
    return lam * (n / lam.sum())


def n_eff(R: np.ndarray) -> float:
    lam = eigenvalues(R)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def n_eff_li_ji(R: np.ndarray) -> float:
    lam = eigenvalues(R)
    return float(sum(1.0 for x in lam if x >= 1) + sum(x - math.floor(x) for x in lam if x < 1))


def expected_max_sr(var_sr: float, n: float) -> float:
    """Equation (1), per observation, under E[SR] = 0.

    Equation (1) is an asymptotic for N much greater than 1 and diverges to minus infinity as
    N approaches 1. At a single trial the expected maximum IS the mean, which is zero under the
    null, and the deflated Sharpe ratio collapses to the probabilistic Sharpe ratio. Returning
    zero there is the correct limit, not a patch.
    """
    if float(n) <= 1.0:
        return 0.0
    n = max(float(n), 1.0000001)
    return math.sqrt(max(var_sr, 0.0)) * (
        (1 - GAMMA) * stats.norm.ppf(1 - 1 / n) + GAMMA * stats.norm.ppf(1 - 1 / (n * math.e)))


def deflated_sharpe(sr: float, var_sr: float, n: float, T: int, skew: float, kurt: float) -> float:
    """Equation (2). `sr` per observation, `kurt` non-excess."""
    sr0 = expected_max_sr(var_sr, n)
    denom = math.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr, 1e-12))
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(T - 1) / denom))


def implied_n(measured_max_annual_sr: float, var_sr: float) -> float:
    """The N at which equation (1) reproduces a measured best-of-grid Sharpe."""
    lo, hi = 1.0000001, 1e12
    for _ in range(300):
        mid = math.sqrt(lo * hi)
        if expected_max_sr(var_sr, mid) * ANN < measured_max_annual_sr:
            lo = mid
        else:
            hi = mid
    return float(math.sqrt(lo * hi))


def audit(R: np.ndarray, measured_null_max: float | None = None) -> dict:
    R = np.asarray(R, dtype=np.float64)
    n_cfg, T = R.shape
    sd = R.std(axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sr_ann = np.nan_to_num(R.mean(axis=1) / np.where(sd > 0, sd, np.nan) * ANN)
    sr = sr_ann / ANN
    best = int(np.argmax(sr))
    var_sr = float(sr.var(ddof=1))
    ne, nl = n_eff(R), n_eff_li_ji(R)
    skew = float(stats.skew(R[best]))
    kurt = float(stats.kurtosis(R[best], fisher=False))
    out = {
        "n_configs": n_cfg, "n_sessions": T,
        "sharpe_best": float(sr_ann[best]), "sharpe_median": float(np.median(sr_ann)),
        "sharpe_sd": float(sr_ann.std(ddof=1)),
        "selection_premium": float(sr_ann[best] - np.median(sr_ann)),
        "n_eff": ne, "n_eff_li_ji": nl,
        "e_max": {k: expected_max_sr(var_sr, v) * ANN
                  for k, v in (("N=1", 1.0), ("N=n_eff", ne), ("N=n_configs", float(n_cfg)))},
        "dsr": {k: deflated_sharpe(sr[best], var_sr, v, T, skew, kurt)
                for k, v in (("N=1", 1.0), ("N=n_eff", ne), ("N=n_configs", float(n_cfg)))},
    }
    if measured_null_max is not None:
        out["implied_n"] = implied_n(measured_null_max, var_sr)
        out["measured_null_max"] = float(measured_null_max)
    return out


def report(R: np.ndarray, measured_null_max: float | None = None) -> str:
    a = audit(R, measured_null_max)
    lines = [f"grid: {a['n_configs']} configurations over {a['n_sessions']} sessions",
             f"  Sharpe best {a['sharpe_best']:.2f}, median {a['sharpe_median']:.2f}, "
             f"sd {a['sharpe_sd']:.2f}, selection premium {a['selection_premium']:.2f}",
             f"  effective trials: participation ratio {a['n_eff']:.1f}, "
             f"Li-Ji {a['n_eff_li_ji']:.1f}"]
    for k in a["e_max"]:
        lines.append(f"  {k:<14} expected best-of-grid Sharpe {a['e_max'][k]:.2f}   "
                     f"deflated Sharpe {a['dsr'][k]:.3f}")
    if "implied_n" in a:
        lines.append(f"  measured null max {a['measured_null_max']:.2f} implies N = "
                     f"{a['implied_n']:,.0f}")
    return "\n".join(lines)


if __name__ == "__main__":
    import os
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "grid_returns.npy")
    null = float(sys.argv[2]) if len(sys.argv) > 2 else None
    print(report(np.load(path), null))
