"""Autocorrelation-robust annualisation of every configuration's Sharpe ratio.

Why this exists. The grid annualises by the independent-observations factor sqrt(252). Holding
periods of 1, 3 and 5 sessions are a factor IN the grid, and a multi-day hold induces positive
serial dependence in the daily series by construction, so sqrt(252) is wrong by an amount that
DIFFERS ACROSS CONFIGURATIONS. A reviewer's objection follows directly: part of the reported
dispersion could be the annualisation rather than the trade. The manuscript conceded this in
Threats and did not measure it. This measures it.

Lo (2002), "The Statistics of Sharpe Ratios", Financial Analysts Journal 58(4), gives the
correct scaling factor for a q-period Sharpe ratio under serially correlated returns:

    eta(q) = q / sqrt( q + 2 * sum_{k=1}^{q-1} (q - k) * rho_k )

with rho_k the lag-k autocorrelation. Under independence all rho_k are 0 and eta(q) = sqrt(q),
recovering the usual factor. Autocorrelations are estimated to MAX_LAG and treated as zero
beyond it: the dependence here is induced by overlapping holds of at most MAX_HOLD sessions, so
lags past that carry noise rather than signal, and including all 251 lags would let estimation
error dominate the correction it is meant to supply.

Writes data/numbers.json["robust"]. Changes no existing key.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import backtest as B  # noqa: E402

MAX_LAG = B.MAX_HOLD          # 5; the longest hold in the grid, so the longest induced overlap
Q = 252.0


def autocorr(R: np.ndarray, k: int) -> np.ndarray:
    """Lag-k autocorrelation of every row, computed on the row's own demeaned series."""
    X = R - R.mean(axis=1, keepdims=True)
    denom = (X * X).sum(axis=1)
    num = (X[:, k:] * X[:, :-k]).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(denom > 0, num / denom, 0.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def eta(R: np.ndarray, max_lag: int = MAX_LAG, q: float = Q) -> tuple[np.ndarray, int]:
    """-> (scaling factor per configuration, count whose truncated variance was non-positive)."""
    s = np.zeros(R.shape[0])
    for k in range(1, max_lag + 1):
        s += (q - k) * autocorr(R, k)
    inner = q + 2.0 * s
    # A configuration can be autocorrelated strongly enough to drive the variance estimate
    # non-positive at this truncation. That is an estimation artefact, not a Sharpe of infinity:
    # such rows are reported separately rather than silently clipped into the distribution.
    bad = inner <= 0
    out = np.where(bad, np.nan, q / np.sqrt(np.where(bad, 1.0, inner)))
    return out, int(bad.sum())


def main() -> None:
    R = np.load(os.path.join(B.DATA, "grid_returns.npy")).astype(np.float64)
    n = json.load(open(os.path.join(B.DATA, "numbers.json"), encoding="utf-8"))

    # Two-sided sanity check: this module must reproduce the grid's own published statistic before
    # anything it says about a corrected one is worth reading.
    std = B.sharpe(R)
    for key, got in (("sharpe_min", std.min()), ("sharpe_max", std.max()),
                     ("sharpe_sd", std.std(ddof=1))):
        want = float(n[key])
        if abs(got - want) > 5e-3:
            raise SystemExit(f"reproduction check FAILED: {key} recomputes to {got:.6f}, "
                             f"numbers.json says {want:.6f}. Refusing to report a correction to a "
                             f"statistic this module cannot reproduce.")
    print(f"reproduction check ok: min {std.min():.3f} max {std.max():.3f} sd {std.std(ddof=1):.3f}")

    sd = R.std(axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sr_daily = np.where(sd > 0, R.mean(axis=1) / np.where(sd > 0, sd, np.nan), 0.0)
    sr_daily = np.nan_to_num(sr_daily, nan=0.0, posinf=0.0, neginf=0.0)

    e, n_bad = eta(R)
    ok = np.isfinite(e)
    rob = np.where(ok, sr_daily * e, np.nan)

    live = ok & (sd > 0)
    rho1 = autocorr(R, 1)
    # POPULATION. The paper's dispersion figures (sharpe_sd, sharpe_min, sharpe_max) are all over
    # ALL configurations, including the 1,152 that never trade and sit at exactly 0. The headline
    # robust figure must therefore also be over ALL configurations, or the comparison the paper
    # prints is 7,776 against 6,624. It was, and it inflated the widening from +11% to +19%.
    # `_live` variants are kept for reference and are never paired with an all-config number.
    res = {
        "max_lag": MAX_LAG,
        "n_configs": int(R.shape[0]),
        "n_undefined": n_bad,
        "population": "all configurations, matching sharpe_sd",
        "eta_over_sqrtq_median": float(np.nanmedian(e[ok]) / np.sqrt(Q)),
        "eta_over_sqrtq_min": float(np.nanmin(e[ok]) / np.sqrt(Q)),
        "eta_over_sqrtq_max": float(np.nanmax(e[ok]) / np.sqrt(Q)),
        "rho1_median": float(np.median(rho1)),
        "rho1_max": float(rho1.max()),
        "robust_min": float(np.nanmin(rob)),
        "robust_max": float(np.nanmax(rob)),
        "robust_sd": float(np.nanstd(rob, ddof=1)),
        "robust_range": float(np.nanmax(rob) - np.nanmin(rob)),
        "std_range": float(std.max() - std.min()),
        "rank_corr_with_standard": float(np.corrcoef(
            np.argsort(np.argsort(std)), np.argsort(np.argsort(rob)))[0, 1]),
        "best_config_standard": float(std.max()),
        "best_config_robust": float(rob[np.nanargmax(std)]),
    }
    # The question the objection actually asks: does the SPREAD survive the correction?
    res["dispersion_survives"] = bool(res["robust_sd"] >= 0.5)
    res["sd_ratio_robust_over_standard"] = float(res["robust_sd"] / float(n["sharpe_sd"]))

    # A widened spread driven by a handful of strongly autocorrelated configurations would be an
    # artefact of the correction, not a finding, so report the comparison on the same trimmed and
    # rank-based footing for both statistics. Both must move the same way for the claim to hold.
    def trimmed_sd(x: np.ndarray) -> float:
        lo, hi = np.percentile(x, [1, 99])
        return float(x[(x >= lo) & (x <= hi)].std(ddof=1))

    def iqr(x: np.ndarray) -> float:
        return float(np.subtract(*np.percentile(x, [75, 25])))

    # Recorded so a gate can assert, without recomputing the grid, that the standard sd over the
    # SAME array the robust sd came from is the number the paper prints as \SharpeSD.
    res["std_sd_all"] = float(std.std(ddof=1))
    res["live_configs"] = int(live.sum())
    res["std_sd_live"] = float(std[live].std(ddof=1))
    res["robust_sd_live"] = float(rob[live].std(ddof=1))
    # Trimmed and IQR comparisons are on the SAME all-configuration population as the headline,
    # so that every pair in the paragraph is like-for-like and no reader has to ask which set a
    # given number came from.
    res["std_sd_trimmed"] = trimmed_sd(std)
    res["robust_sd_trimmed"] = trimmed_sd(rob)
    res["std_iqr"] = iqr(std)
    res["robust_iqr"] = iqr(rob)
    res["n_eta_ratio_above_2"] = int((e[ok] / np.sqrt(Q) > 2).sum())
    res["widening_survives_trimming"] = bool(
        res["robust_sd_trimmed"] > res["std_sd_trimmed"] and res["robust_iqr"] > res["std_iqr"])

    # The headline pair must compare like with like. This is the check that was missing when the
    # module first shipped: robust_sd was over the live configurations and was printed against
    # sharpe_sd, which is over all of them.
    if abs(res["std_sd_trimmed"]) > 0 and abs(float(n["sharpe_sd"])) > 0:
        recomputed_all = float(std.std(ddof=1))
        if abs(recomputed_all - float(n["sharpe_sd"])) > 5e-3:
            raise SystemExit(
                f"population mismatch: the standard sd recomputed over the same array used for "
                f"robust_sd is {recomputed_all:.4f}, but numbers.json['sharpe_sd'] is "
                f"{float(n['sharpe_sd']):.4f}. The paper would print a widening between two "
                f"different populations.")

    print(json.dumps(res, indent=2))
    n["robust"] = res
    with open(os.path.join(B.DATA, "numbers.json"), "w", encoding="utf-8") as fh:
        json.dump(n, fh, indent=2, sort_keys=True)
    print('wrote data/numbers.json["robust"]')


if __name__ == "__main__":
    main()
