"""Turn the grid into the paper's numbers. Writes ../data/numbers.json, the single source of truth.

Nothing in the manuscript may contain a quantity that is not a key of this file. `paper/consistency.py`
enforces that in both directions.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
ANN = math.sqrt(252.0)
GAMMA = 0.5772156649015329        # Euler-Mascheroni

FACTORS = ["model", "prompt", "trunc", "parse", "signal", "thresh", "hold", "entry",
           "weight", "neutral"]
LLM_SIDE = ["model", "prompt", "trunc"]
POST_LLM = ["parse", "signal", "thresh", "hold", "entry", "weight", "neutral"]

# The single setting a paper would report if it reported only one: the modal choice in the
# published LLM-trading pipelines coded in Table 1.
DEFAULT = {"model": "qwen2.5:7b-instruct", "prompt": "P1", "trunc": "4000", "parse": "lenient",
           "signal": "level", "thresh": "0.0", "hold": "1", "entry": "close", "weight": "equal",
           "neutral": "none"}


def load_grid():
    with open(os.path.join(DATA, "grid.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["sharpe"] = float(r["sharpe"])
        r["thresh"] = str(float(r["thresh"]))
    return rows


def expected_max_sr(mean_sr: float, var_sr: float, n: float) -> float:
    """Bailey and Lopez de Prado (2014) equation 1, in per-observation units.

    At N = 1 the equation diverges; the correct limit is the mean itself, which makes the deflated
    Sharpe ratio collapse to the probabilistic Sharpe ratio, as it should for a single trial.
    """
    if float(n) <= 1.0:
        return mean_sr
    n = max(float(n), 1.0000001)
    z1 = stats.norm.ppf(1.0 - 1.0 / n)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n * math.e))
    return mean_sr + math.sqrt(max(var_sr, 0.0)) * ((1 - GAMMA) * z1 + GAMMA * z2)


def deflated_sharpe(sr: float, var_sr: float, n: float, T: int,
                    skew: float, kurt: float) -> float:
    """Bailey and Lopez de Prado (2014) equation 2. `sr` is per-observation, `kurt` non-excess."""
    sr0 = expected_max_sr(0.0, var_sr, n)
    denom = math.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr, 1e-12))
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(T - 1) / denom))


def variance_shares(rows):
    """Balanced-design sum-of-squares partition of Sharpe variance over the ten factors."""
    sr = np.array([r["sharpe"] for r in rows])
    grand = sr.mean()
    ss_total = float(((sr - grand) ** 2).sum())
    shares, effects = {}, {}
    for f in FACTORS:
        levels = sorted({r[f] for r in rows})
        ss = 0.0
        means = {}
        for lev in levels:
            sel = np.array([r[f] == lev for r in rows])
            m = sr[sel].mean()
            means[lev] = float(m)
            ss += sel.sum() * (m - grand) ** 2
        shares[f] = ss / ss_total
        effects[f] = {"levels": means, "range": max(means.values()) - min(means.values())}
    shares["interactions"] = 1.0 - sum(shares[f] for f in FACTORS)
    return ss_total, shares, effects


def within_between(rows):
    """Dispersion attributable to swapping the model, holding every other choice fixed, and the
    reverse. More directly interpretable than a variance share, and it is what H2 is about."""
    by_harness = {}
    for r in rows:
        key = tuple(r[f] for f in FACTORS if f != "model")
        by_harness.setdefault(key, []).append(r["sharpe"])
    model_sd = np.mean([np.std(v, ddof=1) for v in by_harness.values() if len(v) > 1])
    model_range = np.mean([max(v) - min(v) for v in by_harness.values() if len(v) > 1])

    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r["sharpe"])
    harness_sd = np.mean([np.std(v, ddof=1) for v in by_model.values()])
    harness_range = np.mean([max(v) - min(v) for v in by_model.values()])
    return {"model_sd": float(model_sd), "model_range": float(model_range),
            "harness_sd": float(harness_sd), "harness_range": float(harness_range),
            "ratio_sd": float(harness_sd / model_sd) if model_sd > 0 else float("nan")}


def eigen_neff(R: np.ndarray):
    """All nonzero eigenvalues of the config correlation matrix, via the T x T Gram matrix."""
    Z = R.astype(np.float64)
    Z = Z - Z.mean(axis=1, keepdims=True)
    sd = Z.std(axis=1, ddof=0)
    live = sd > 0
    Z = Z[live] / sd[live][:, None]
    n, T = Z.shape
    g = (Z.T @ Z) / T                       # T x T, same nonzero spectrum as ZZ'/T
    lam = np.linalg.eigvalsh(g)
    lam = np.clip(lam, 0.0, None)[::-1]
    lam = lam * (n / lam.sum())             # eigenvalues of an n x n correlation matrix sum to n
    participation = float(lam.sum() ** 2 / (lam ** 2).sum())
    li_ji = float(sum(1.0 if l >= 1 else 0.0 for l in lam) + sum(l - math.floor(l) for l in lam
                                                                if l < 1))
    # sum(lam^2) is the squared Frobenius norm of the correlation matrix, i.e. sum_ij C_ij^2.
    # Removing the n diagonal ones and averaging over the n(n-1) off-diagonal entries therefore
    # gives the mean of the SQUARED correlations, NOT the mean correlation. It was returned under
    # the name mean_pairwise_corr and printed in the paper as "a mean pairwise correlation of
    # 0.05"; the true mean correlation is 0.12 and 0.05 is the mean square. Both are now returned
    # under names that say which is which, and the mean correlation is computed directly.
    mean_sq_corr = float((lam ** 2).sum() - n) / (n * (n - 1)) if n > 1 else 0.0
    # Mean off-diagonal correlation, from the row sums: sum_ij C_ij = ||Z.sum(axis=0)||^2 / T.
    colsum = Z.sum(axis=0)
    total = float((colsum @ colsum) / T)
    mean_corr = (total - n) / (n * (n - 1)) if n > 1 else 0.0
    return {"n_live": int(n), "participation_ratio": participation, "li_ji": li_ji,
            "top_eig_share": float(lam[0] / n),
            "mean_pairwise_corr": float(mean_corr),
            "mean_sq_pairwise_corr": mean_sq_corr,
            "rms_pairwise_corr": float(math.sqrt(max(mean_sq_corr, 0.0))),
            "eig_top10": [float(x) for x in lam[:10]]}


def _sb_index(T: int, rng, mean_block: float) -> np.ndarray:
    """One stationary-bootstrap index vector over T sessions."""
    p = 1.0 / mean_block
    idx = np.empty(T, dtype=int)
    i = rng.integers(T)
    for t in range(T):
        idx[t] = i
        i = rng.integers(T) if rng.random() < p else (i + 1) % T
    return idx


def paired_bootstrap(R: np.ndarray, best_i: int, def_i: int, med_i: int,
                     reps: int = 400, mean_block: float = 5.0, seed=17):
    """Resample sessions ONCE per replication and re-Sharpe every configuration on them.

    This is the check that answers the sharpest objection to the paper: that a Sharpe ratio over
    one year is noisy enough to produce the observed spread on its own. Because every
    configuration is re-evaluated on the SAME resampled sessions, the sampling error common to all
    of them cancels, and what is left is the sampling error of the between-configuration
    differences, which is the quantity the objection is actually about.
    """
    rng = np.random.default_rng(seed)
    T = R.shape[1]
    Rd = R.astype(np.float64)
    within = np.empty((reps, Rd.shape[0]), dtype=np.float32)
    for b in range(reps):
        x = Rd[:, _sb_index(T, rng, mean_block)]
        sd = x.std(axis=1, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            within[b] = np.nan_to_num(x.mean(axis=1) / np.where(sd > 0, sd, np.nan) * ANN)
    per_config_sd = within.std(axis=0, ddof=1)          # sampling sd of one configuration's Sharpe
    spread = within.std(axis=1, ddof=1)                 # between-config spread in each resample
    gap_best_med = within[:, best_i] - np.median(within, axis=1)
    gap_best_def = within[:, best_i] - within[:, def_i]
    return {
        "within_config_sd_mean": float(per_config_sd.mean()),
        "within_config_sd_median": float(np.median(per_config_sd)),
        "between_spread_mean": float(spread.mean()),
        "ratio_between_within": float(spread.mean() / max(per_config_sd.mean(), 1e-12)),
        "gap_best_median_mean": float(gap_best_med.mean()),
        "gap_best_median_sd": float(gap_best_med.std(ddof=1)),
        "gap_best_default_mean": float(gap_best_def.mean()),
        "gap_best_default_sd": float(gap_best_def.std(ddof=1)),
        "frac_best_above_median": float((gap_best_med > 0).mean()),
        "reps": reps,
    }


def stationary_bootstrap_ci(r: np.ndarray, reps: int = 2000, mean_block: float = 5.0, seed=11):
    rng = np.random.default_rng(seed)
    T = len(r)
    p = 1.0 / mean_block
    out = np.empty(reps)
    for b in range(reps):
        idx = np.empty(T, dtype=int)
        i = rng.integers(T)
        for t in range(T):
            idx[t] = i
            if rng.random() < p:
                i = rng.integers(T)
            else:
                i = (i + 1) % T
        x = r[idx]
        sd = x.std(ddof=1)
        out[b] = x.mean() / sd * ANN if sd > 0 else 0.0
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> None:
    rows = load_grid()
    R = np.load(os.path.join(DATA, "grid_returns.npy"))
    null_max = np.load(os.path.join(DATA, "null_max.npy"))
    meta = json.load(open(os.path.join(DATA, "backtest_meta.json"), encoding="utf-8"))
    T = meta["n_sessions"]
    sr = np.array([r["sharpe"] for r in rows])
    N = len(rows)
    assert R.shape[0] == N, f"grid.csv has {N} rows but grid_returns.npy has {R.shape[0]}"

    n = {}
    n["n_events"] = meta["n_events"]
    n["n_price_ok"] = meta["n_price_ok"]
    n["n_sessions"] = T
    n["first_session"] = meta["first_session"]
    n["last_session"] = meta["last_session"]
    n["n_configs"] = N
    n["n_llm_configs"] = len(meta["llm_keys"])
    n["n_post_configs"] = N // len(meta["llm_keys"])
    n["n_llm_calls"] = meta["score_meta"]["score_used"]
    n["cost_bps"] = meta["cost_bps_per_side"]
    n["num_predict"] = int(meta.get("num_predict", 0))
    n["perm_reps"] = int(len(null_max))

    # --- dispersion -----------------------------------------------------------
    n["sharpe_min"] = float(sr.min())
    n["sharpe_max"] = float(sr.max())
    n["sharpe_median"] = float(np.median(sr))
    n["sharpe_mean"] = float(sr.mean())
    n["sharpe_sd"] = float(sr.std(ddof=1))
    for q in (5, 25, 75, 95):
        n[f"sharpe_p{q}"] = float(np.percentile(sr, q))
    n["sharpe_iqr"] = n["sharpe_p75"] - n["sharpe_p25"]
    # A configuration can be well defined and still never trade: a strict parser that rejects every
    # answer from a model-prompt cell leaves it with no positions. Its return series is identically
    # zero, so its Sharpe ratio is 0/0. We record it as 0, which is the limit of a strategy that
    # holds nothing, and report how many configurations that is rather than hiding them in a bin.
    n["n_never_trades"] = int((np.abs(R).sum(axis=1) == 0).sum())
    n["frac_never_trades"] = float(n["n_never_trades"] / N)
    live = np.abs(R).sum(axis=1) > 0
    n["sharpe_median_live"] = float(np.median(sr[live]))
    n["frac_positive_live"] = float((sr[live] > 0).mean())
    n["frac_positive"] = float((sr > 0).mean())
    n["frac_above_1"] = float((sr > 1).mean())
    n["frac_below_minus1"] = float((sr < -1).mean())

    # --- reference configurations --------------------------------------------
    best_i = int(np.argmax(sr))
    med_i = int(np.argsort(sr)[len(sr) // 2])
    def_i = next((i for i, r in enumerate(rows)
                  if all(r[k] == v for k, v in DEFAULT.items())), None)
    assert def_i is not None, "the default configuration is not in the grid"
    n["sharpe_best"] = float(sr[best_i])
    n["sharpe_default"] = float(sr[def_i])
    n["best_config"] = {f: rows[best_i][f] for f in FACTORS}
    n["default_config"] = {f: rows[def_i][f] for f in FACTORS}
    for tag, i in (("best", best_i), ("default", def_i), ("median", med_i)):
        lo, hi = stationary_bootstrap_ci(R[i].astype(np.float64))
        n[f"ci_{tag}"] = [lo, hi]

    # --- variance decomposition ----------------------------------------------
    ss_total, shares, effects = variance_shares(rows)
    n["var_share"] = {k: float(v) for k, v in shares.items()}
    n["var_share_model"] = float(shares["model"])
    n["var_share_llm_side"] = float(sum(shares[f] for f in LLM_SIDE))
    n["var_share_post_llm"] = float(sum(shares[f] for f in POST_LLM))
    n["factor_effects"] = effects
    n["wb"] = within_between(rows)

    # Is the spread sampling noise? Answer it with a paired resample, not with an assertion.
    n["boot"] = paired_bootstrap(R, best_i, def_i, med_i)

    # Was the ranking manufactured by the widest levels? Re-run the partition on a trimmed grid
    # that drops the most aggressive level of the two factors a reviewer will name.
    # The same partition over only the configurations that trade. The shares are computed on the
    # full grid, which records the 1,152 never-trading configurations at a Sharpe of exactly zero,
    # so a reader is entitled to ask whether the pre-registered ordering is an artefact of that
    # zero inflation. It is not: the ordering holds and the margin widens slightly.
    Rmat = np.load(os.path.join(DATA, "grid_returns.npy")).astype(np.float64)
    live_mask = Rmat.std(axis=1, ddof=1) > 0
    live_rows = [r for r, keep in zip(rows, live_mask) if keep]
    if live_rows:
        _, lshares, _ = variance_shares(live_rows)
        n["live_only"] = {
            "n_configs": len(live_rows),
            "var_share_model": float(lshares["model"]),
            "var_share_post_llm": float(sum(lshares[f] for f in POST_LLM)),
            "var_share_interactions": float(lshares["interactions"]),
        }

    trimmed = [r for r in rows if r["thresh"] != "0.2" and r["hold"] != "5"]
    if trimmed:
        _, tshares, _ = variance_shares(trimmed)
        n["trimmed"] = {
            "n_configs": len(trimmed),
            "var_share_model": float(tshares["model"]),
            "var_share_post_llm": float(sum(tshares[f] for f in POST_LLM)),
            "sharpe_sd": float(np.std([r["sharpe"] for r in trimmed], ddof=1)),
        }

    n["cost_sweep"] = meta.get("cost_sweep", {})

    # The "no practitioner would choose that" objection: restrict the grid to settings a careful
    # researcher would plausibly defend and see whether the spread survives. Pre-specified as the
    # conservative reading of the three factors a reviewer names first.
    prudent = [r for r in rows if r["neutral"] == "market" and r["signal"] in ("demean", "xrank")
               and r["hold"] in ("1", "3") and r["parse"] == "lenient"]
    if prudent:
        srp = np.array([r["sharpe"] for r in prudent])
        _, pshares, _ = variance_shares(prudent)
        n["prudent"] = {
            "n_configs": len(prudent), "sharpe_sd": float(srp.std(ddof=1)),
            "sharpe_min": float(srp.min()), "sharpe_max": float(srp.max()),
            "sharpe_median": float(np.median(srp)),
            "range": float(srp.max() - srp.min()),
            "var_share_model": float(pshares["model"]),
            "var_share_post_llm": float(sum(pshares[f] for f in POST_LLM
                                            if f not in ("neutral", "parse"))),
        }

    # Hold the LLM side completely fixed and let only the post-LLM choices vary. This is the
    # cleanest statement of the paper's point, because it needs no comparison across models at all:
    # one model, one prompt, one truncation, and the remaining choices still span this much.
    by_llm = {}
    for r in rows:
        by_llm.setdefault((r["model"], r["prompt"], r["trunc"]), []).append(r["sharpe"])
    spreads = {k: (float(np.std(v, ddof=1)), float(min(v)), float(max(v)))
               for k, v in by_llm.items() if len(v) > 1}
    n["post_only"] = {
        "n_llm_cells": len(spreads),
        "sd_mean": float(np.mean([s for s, _, _ in spreads.values()])),
        "sd_max": float(max(s for s, _, _ in spreads.values())),
        "range_mean": float(np.mean([hi - lo for _, lo, hi in spreads.values()])),
        "range_max": float(max(hi - lo for _, lo, hi in spreads.values())),
    }

    # The scores are optimistic on average, so a configuration that does not demean or neutralise
    # is partly holding the market. That is itself an undisclosed choice worth reporting, but the
    # paper must also show what is left once it is removed, or a reviewer will read the whole
    # result as "beta or no beta". Re-run the partition inside the market-neutral half only.
    neutral_only = [r for r in rows if r["neutral"] == "market"]
    if neutral_only:
        _, nshares, _ = variance_shares(neutral_only)
        srn = np.array([r["sharpe"] for r in neutral_only])
        n["neutral_only"] = {
            "n_configs": len(neutral_only),
            "var_share_model": float(nshares["model"]),
            "var_share_post_llm": float(sum(nshares[f] for f in POST_LLM if f != "neutral")),
            "sharpe_sd": float(srn.std(ddof=1)),
            "sharpe_min": float(srn.min()), "sharpe_max": float(srn.max()),
            "sharpe_median": float(np.median(srn)),
        }

    # Do the post-LLM choices move the numerator or only the denominator? A factor that only
    # rescales volatility would be a much less interesting finding.
    for tag, key in (("mean", "mean_daily"), ("vol", "sd_daily")):
        vals = [float(r[key]) for r in rows]
        proxy = [dict(r, sharpe=v) for r, v in zip(rows, vals)]
        _, sh, _ = variance_shares(proxy)
        n[f"var_share_{tag}"] = {"model": float(sh["model"]),
                                 "post_llm": float(sum(sh[f] for f in POST_LLM))}

    # --- correlation structure and effective trials --------------------------
    eig = eigen_neff(R)
    n["neff"] = eig
    n["n_eff"] = eig["participation_ratio"]

    # --- does selection survive a split? --------------------------------------
    # The classical demonstration of backtest overfitting: pick the best configuration on the first
    # half of the window and see where it lands on the second. Cheap, intuitive, and it tests the
    # selection procedure rather than the strategy.
    half = R.shape[1] // 2
    A, Bh = R[:, :half].astype(np.float64), R[:, half:].astype(np.float64)

    def _sr(x):
        sd = x.std(axis=1, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nan_to_num(x.mean(axis=1) / np.where(sd > 0, sd, np.nan) * ANN)

    sr_a, sr_b = _sr(A), _sr(Bh)
    best_a = int(np.argmax(sr_a))
    pct_b = float(100.0 * (sr_b < sr_b[best_a]).mean())
    top = np.argsort(sr_a)[-10:]
    n["split"] = {
        "n_sessions_each": int(half),
        "spearman": float(stats.spearmanr(sr_a, sr_b).statistic),
        "is_best_sharpe": float(sr_a[best_a]),
        "oos_of_is_best": float(sr_b[best_a]),
        "oos_percentile_of_is_best": pct_b,
        "oos_median_of_is_top10": float(np.median(sr_b[top])),
        "oos_median_all": float(np.median(sr_b)),
        "frac_top10_beat_oos_median": float((sr_b[top] > np.median(sr_b)).mean()),
    }

    # How long would a researcher have to look before finding something publishable?
    for thr in (1.0, 1.5):
        frac = float((sr > thr).mean())
        n[f"tries_to_sharpe_{str(thr).replace('.', 'p')}"] = (
            float(1.0 / frac) if frac > 0 else None)

    # --- deflated Sharpe ------------------------------------------------------
    sr_daily = sr / ANN
    var_sr = float(sr_daily.var(ddof=1))
    rb = R[best_i].astype(np.float64)
    skew = float(stats.skew(rb))
    kurt = float(stats.kurtosis(rb, fisher=False))
    n["best_skew"] = skew
    n["best_kurtosis"] = kurt
    n["var_trial_sr_daily"] = var_sr
    n["dsr"] = {}
    for tag, nn in (("N1", 1.0), ("Neff", eig["participation_ratio"]),
                    ("Nli_ji", eig["li_ji"]), ("Nfull", float(N))):
        n["dsr"][tag] = deflated_sharpe(sr_daily[best_i], var_sr, nn, T, skew, kurt)
    n["psr_best"] = float(stats.norm.cdf(
        sr_daily[best_i] * math.sqrt(T - 1)
        / math.sqrt(max(1 - skew * sr_daily[best_i]
                        + (kurt - 1) / 4 * sr_daily[best_i] ** 2, 1e-12))))

    # --- exact permutation null ----------------------------------------------
    null_sd_path = os.path.join(DATA, "null_sd.npy")
    if os.path.exists(null_sd_path):
        null_sd = np.load(null_sd_path)
        n["null_grid_sd_mean"] = float(null_sd.mean())
        n["sd_ratio_real_null"] = float(n["sharpe_sd"] / max(null_sd.mean(), 1e-12))
    n["frac_events_permutable"] = float(meta.get("frac_events_permutable", float("nan")))
    n["median_events_per_session"] = float(meta.get("median_events_per_session", float("nan")))
    n["null_max_mean"] = float(null_max.mean())
    n["null_max_sd"] = float(null_max.std(ddof=1))
    n["null_max_median"] = float(np.median(null_max))
    n["null_max_p95"] = float(np.percentile(null_max, 95))
    n["null_max_max"] = float(null_max.max())
    n["perm_p_best"] = float((1 + (null_max >= sr[best_i]).sum()) / (len(null_max) + 1))
    n["perm_p_default"] = float((1 + (null_max >= sr[def_i]).sum()) / (len(null_max) + 1))
    n["perm_p_median"] = float((1 + (null_max >= np.median(sr)).sum()) / (len(null_max) + 1))

    # Does equation 1 predict the measured best-of-grid Sharpe, and at which N?
    n["emax_pred"] = {tag: float(expected_max_sr(0.0, var_sr, nn) * ANN)
                      for tag, nn in (("N1", 1.0), ("Neff", eig["participation_ratio"]),
                                      ("Nli_ji", eig["li_ji"]), ("Nfull", float(N)))}
    n["emax_measured"] = float(null_max.mean())
    n["emax_err"] = {k: float(v - n["emax_measured"]) for k, v in n["emax_pred"].items()}
    n["emax_best_N"] = min(n["emax_err"], key=lambda k: abs(n["emax_err"][k]))
    # The N that would make equation 1 exactly reproduce the measured null maximum. Equation 1 is
    # monotone in N, so bisect; record explicitly when the target lies outside the bracket rather
    # than returning the bracket endpoint as if it were a solution.
    lo, hi = 1.0000001, 1e12
    n["n_implied_saturated"] = bool(
        expected_max_sr(0.0, var_sr, hi) * ANN < n["emax_measured"])
    for _ in range(300):
        mid = math.sqrt(lo * hi)
        if expected_max_sr(0.0, var_sr, mid) * ANN < n["emax_measured"]:
            lo = mid
        else:
            hi = mid
    n["n_implied"] = float(math.sqrt(lo * hi))

    # The deflated Sharpe ratio at the MEASURED trial count. The table previously stopped at the
    # two spectral estimates and the nominal grid, so the paper argued that the permutation
    # replaces the assumption and then never showed what the correction is worth at the number the
    # permutation actually produced. It is the row a reader most wants and it costs one line.
    n["dsr"]["Nimplied"] = deflated_sharpe(sr_daily[best_i], var_sr, n["n_implied"], T, skew, kurt)

    # --- how a single reported number sits inside the grid --------------------
    n["default_percentile"] = float(100.0 * (sr < sr[def_i]).mean())
    live_mask = np.abs(R).sum(axis=1) > 0
    n["default_percentile_live"] = float(100.0 * (sr[live_mask] < sr[def_i]).mean())
    n["best_percentile"] = 100.0
    n["frac_above_null_p95"] = float((sr > n["null_max_p95"]).mean())
    n["n_above_null_p95"] = int((sr > n["null_max_p95"]).sum())
    # A reader who is handed one configuration sees a number this far from the grid median.
    n["max_minus_median"] = float(sr[best_i] - np.median(sr))
    n["default_minus_median"] = float(sr[def_i] - np.median(sr))

    # --- descriptive: how often each model answers in the requested format ----
    n["compliance"] = meta.get("compliance", {})

    # --- hypotheses -----------------------------------------------------------
    n["H1_dispersion_ge_0p5"] = bool(n["sharpe_sd"] >= 0.5)
    n["H2_post_llm_beats_model"] = bool(n["var_share_post_llm"] > n["var_share_model"])
    n["H3_neff_below_10pct"] = bool(eig["participation_ratio"] < 0.10 * N)
    n["H5_neff_closer"] = bool(abs(n["emax_err"]["Neff"]) < min(abs(n["emax_err"]["N1"]),
                                                               abs(n["emax_err"]["Nfull"])))
    n["falsified"] = bool(n["sharpe_sd"] < 0.2 or n["var_share_model"] >= n["var_share_post_llm"])

    # --- provenance of the scale-arm model ------------------------------------
    # Its release date is printed in the paper (\ScaleRelease) and is load-bearing: it is what
    # tells a reader that the scale arm, unlike the pre-registered grid, cannot exclude
    # contamination. It was previously read straight from data/model_provenance.json by
    # make_macros.py, so it never reached numbers.json and therefore never reached the
    # reviewer-facing ground truth, while the paper claimed every printed quantity is generated.
    # Route it through here like every other number.
    prov = json.load(open(os.path.join(DATA, "model_provenance.json"), encoding="utf-8"))
    n["scale_model"] = {"tag": "qwen3.6:latest",
                        "release": prov["qwen3.6:latest"]["release"],
                        "release_source": prov["qwen3.6:latest"]["release_source"]}
    # Sampling provenance. The manuscript described the event set as "every Form 8-K filed by a
    # constituent", which is what the pre-registration froze; the shipped study scores a seed-fixed
    # random subsample of it (amendment A-3, for compute). A paper about undisclosed choices cannot
    # leave its own sampling undisclosed, so the counts are routed through numbers.json like every
    # other printed quantity rather than living only in the decision ledger.
    es = json.load(open(os.path.join(DATA, "event_sample.json"), encoding="utf-8"))
    if "available" not in es:
        raise SystemExit("data/event_sample.json has no 'available' count; run code/record_sample.py "
                         "so the subsample fraction is measured rather than typed")
    n["n_events_available"] = int(es["available"])
    n["n_events_sampled"] = int(es["n"])
    n["event_sample_seed"] = int(es.get("seed", 0))

    big = "nvidia/nemotron-3-ultra-550b-a55b:free"
    if big in prov:
        n["big_model"] = {"tag": big, "release": prov[big]["release"],
                          "release_source": prov[big]["release_source"]}

    # --- everything else the manuscript prints -------------------------------
    # verify_output.txt tells reviewers that "a macro whose value disagrees with block 2 is a
    # defect ... nothing here was typed", where block 2 is this file. That was FALSE for fourteen
    # printed macros, which reached numbers.tex from cap_probe.json, paper_coding.json, the
    # decision ledger and model_provenance.json without ever passing through here. Route them all
    # through, so the ground-truth claim the artifact makes about itself is true.
    # The best-of-grid gap, raw and standardised by each grid's own dispersion. Computed here
    # rather than in make_macros.py so it appears in the analysis output like every other number.
    if "null_grid_sd_mean" in n:
        n["gap_raw"] = n["null_max_mean"] - n["sharpe_best"]
        n["gap_standardised"] = (n["null_max_mean"] / n["null_grid_sd_mean"]
                                 - n["sharpe_best"] / n["sharpe_sd"])

    n["scale_model"]["params"] = "36B"      # qwen3.6, total parameters of the local build
    if "big_model" in n:
        n["big_model"]["params"] = "550B"   # nemotron-3-ultra, total; 55B active per token
    n["main_model_params"] = "7.6B"         # the largest model in the pre-registered grid

    # The stochastic arm's decoding temperature is an INPUT the paper prints, and it was typed as
    # a literal in make_macros.py, making it the one printed quantity with no counterpart in the
    # analysis output. Read it from the module that actually used it, so the value and the run
    # cannot drift apart.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import probes as _probes  # noqa: PLC0415 - imported here to avoid a cycle at module load
    n["seed_temperature"] = float(_probes.TEMPERATURE)

    cap_path = os.path.join(DATA, "cap_probe.json")
    if os.path.exists(cap_path):
        cp = json.load(open(cap_path, encoding="utf-8"))
        cells = {k: v for k, v in cp.items() if k != "_control_holds"}
        fails = [v for v in cells.values() if not str(v.get("why", "")).startswith("CONTROL")]
        ctls = {k: v for k, v in cells.items() if str(v.get("why", "")).startswith("CONTROL")}
        n["cap_probe"] = {
            "n_cells": len(cells),
            "n_filings": int(next(iter(cells.values()))["n"]),
            "cap_low": int(next(iter(cells.values()))["cap_old"]),
            "cap_high": int(next(iter(cells.values()))["cap_new"]),
            "max_move_pct_points": max(abs(100 * (v["strict_new"] - v["strict_old"]))
                                       for v in fails),
            "control_holds": bool(cp.get("_control_holds")),
            "controls": {k: {"old": 100 * v["strict_old"], "new": 100 * v["strict_new"]}
                         for k, v in sorted(ctls.items())},
        }

    coding = json.load(open(os.path.join(DATA, "paper_coding.json"), encoding="utf-8"))
    papers = coding["papers"]
    n["num_coded_total"] = len(papers)
    n["num_coded"] = sum(1 for p in papers
                         if not any(str(v) == "unretrieved" for v in p.values()))

    for name in ("DECISIONS.md", "DECISIONS_REDACTED.md"):
        dpath = os.path.join(os.path.dirname(DATA), name)
        if os.path.exists(dpath):
            dec = open(dpath, encoding="utf-8").read()
            n["num_amendments"] = len(re.findall(r"(?m)^\*\*A-\d+ \|", dec))
            break

    if "n_events_sampled" in n:
        n["sample_pct"] = 100.0 * n["n_events_sampled"] / n["n_events_available"]

    # probe_analysis.py and power.py write their results into the same file. Preserve them, so
    # that the scripts can be run in any order; an earlier version silently deleted the probe arms
    # whenever the main analysis was re-run, and the manuscript's macros vanished with them. The
    # same hole was still open for "power" (the positive control) until the ICAIF submission
    # round: re-running analyze.py after power.py would have deleted the entire positive control
    # while every gate that reads a macro kept passing on the stale numbers.tex.
    path = os.path.join(DATA, "numbers.json")
    if os.path.exists(path):
        try:
            prev = json.load(open(path, encoding="utf-8"))
            # PRESERVE BY DEFAULT. This was a hand-maintained whitelist ("probes", then "power",
            # then "robust") and it rotted three times: each new module that wrote a top-level key
            # had its results deleted by the next run of this script, silently, with every macro
            # gate still passing on a stale numbers.tex. A whitelist of foreign keys cannot be
            # right, because this script cannot know what will be written next. Invert it: keep
            # every pre-existing key this run did not itself produce.
            for key, value in prev.items():
                if key not in n:
                    n[key] = value
        except Exception:  # noqa: BLE001 - a corrupt or absent file is not a reason to fail here
            pass

    # AFTER the preserve step, because the block this augments is written by another module and
    # only exists in `n` once it has been carried over. Computing it earlier silently skipped.
    if "oracle_dispersion" in n:
        levels = sorted(n["oracle_dispersion"]["accuracies"], key=float)
        n["oracle_dispersion"]["acc_low_pct"] = 100 * float(levels[0])
        n["oracle_dispersion"]["acc_high_pct"] = 100 * float(levels[-1])
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(n, fh, indent=2, sort_keys=True)

    print(json.dumps({k: n[k] for k in (
        "n_events", "n_configs", "sharpe_min", "sharpe_median", "sharpe_max", "sharpe_sd",
        "sharpe_default", "sharpe_best", "var_share_model", "var_share_post_llm",
        "n_eff", "null_max_mean", "perm_p_best", "emax_best_N", "n_implied",
        "H1_dispersion_ge_0p5", "H2_post_llm_beats_model", "H3_neff_below_10pct",
        "H5_neff_closer", "falsified")}, indent=2))


if __name__ == "__main__":
    main()
