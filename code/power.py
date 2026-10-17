"""The positive half of the control, plus a configuration-invariant signal statistic.

Why this exists. The permutation benchmark is a NEGATIVE control: it shows what the grid produces
when the scores carry nothing. On its own it cannot distinguish "these models carry no within-session
information" from "this harness could not detect information if it were there". The house rule for
any measurement harness is a TWO-SIDED control, and only one side was built. This module supplies
the other side and costs no model inference:

    power     inject a synthetic score of KNOWN directional accuracy, push it through the identical
              post-LLM grid and the identical permutation benchmark, and report the accuracy at
              which the pipeline first separates from its own null.
    ic        rank correlation between each LLM configuration's score and the event's raw (not
              market-adjusted) return over one session entered at the next close, the ("close", 1)
              block of backtest.precompute. One Spearman per configuration, pooled over events.
              Nearly invariant to threshold, weighting, neutralisation and entry timing, so
              it collapses 7,776 portfolio cells to 18 numbers and answers "is there signal at all"
              without any of the choices under study.
    neutral   the permutation benchmark restricted to the market-neutral half of the grid, because
              within-session permutation preserves each session's net directional exposure and a
              rising window would otherwise flow into the null through beta.

Writes its results into data/numbers.json under "power".
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as B  # noqa: E402

ACCURACIES = [0.50, 0.52, 0.55, 0.60, 0.70]   # directional accuracy of the synthetic oracle
SEED = 20260801


def setup():
    tickers, sessions, t_idx, s_idx, op, cl, spy_ret = B.load_prices()
    events = B.load_events(sessions, t_idx, s_idx)
    pre, price_ok = B.precompute(events, op, cl, len(sessions))
    t_lo = min(e["t0"] for e in events)
    t_hi = min(max(e["t0"] for e in events) + B.MAX_HOLD, len(sessions) - 1)
    win = slice(t_lo, t_hi + 1)
    spy_win = np.ascontiguousarray(spy_ret[win])
    pre = {k: tuple(np.ascontiguousarray(a[:, win]) for a in v) for k, v in pre.items()}
    return events, pre, price_ok, spy_win, spy_ret, win


def event_forward_return(events, pre, win):
    """Each event's own realised return over the reference horizon (next close, one session)."""
    R, A, _, _ = pre[("close", 1)]
    return R.sum(axis=1)


def grid_for_scores(events, pre, spy_win, price_ok, s, valid):
    """Run the 432-configuration post-LLM grid for one score vector."""
    E = len(events)
    S = np.full((1, 2, E), 0.5)
    V = np.zeros((1, 2, E), dtype=bool)
    for pi in range(2):
        S[0, pi] = s
        V[0, pi] = valid
    G, M = B.build_signals(events, S, V, price_ok)
    W, keys = B.weight_matrix(G, M)
    R, cfgs = B.run_grid(W, keys, pre, spy_win)
    return B.sharpe(R.astype(np.float64)), cfgs


def permutation_null(events, pre, spy_win, price_ok, s, valid, reps, rng):
    t0 = np.array([e["t0"] for e in events])
    groups = [np.where(t0 == x)[0] for x in np.unique(t0)]
    out = np.zeros(reps)
    for r in range(reps):
        perm = np.arange(len(events))
        for g in groups:
            if len(g) > 1:
                perm[g] = g[rng.permutation(len(g))]
        sr, _ = grid_for_scores(events, pre, spy_win, price_ok, s[perm], valid[perm])
        out[r] = sr.max()
    return out


def main() -> None:
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    events, pre, price_ok, spy_win, spy_ret, win = setup()
    fwd = event_forward_return(events, pre, win)
    rng = np.random.default_rng(SEED)
    res = {"reps": reps, "accuracies": {}}

    # ---- positive control -------------------------------------------------
    # A single synthetic draw shows the machinery CAN reject something; it does not estimate power.
    # For that we need repeated INDEPENDENT oracle datasets at each accuracy and the fraction of
    # them the benchmark rejects. `draws` controls that; with draws=1 this is a sensitivity check
    # only, and the paper must not call it power.
    draws = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    res["draws"] = draws
    tradable = price_ok & (fwd != 0)
    truth = np.sign(fwd)
    for acc in ACCURACIES:
        ps, bests = [], []
        for k in range(draws):
            keep = rng.random(len(events)) < acc
            d = np.where(keep, truth, -truth)
            s = 0.5 + 0.4 * d                 # confident but not saturated, and constant magnitude
            valid = tradable.copy()
            sr, _ = grid_for_scores(events, pre, spy_win, price_ok, s, valid)
            null = permutation_null(events, pre, spy_win, price_ok, s, valid, reps, rng)
            ps.append(float((1 + (null >= sr.max()).sum()) / (reps + 1)))
            bests.append(float(sr.max()))
        rej = float(np.mean([p <= 0.05 for p in ps]))
        res["accuracies"][f"{acc:.2f}"] = {
            "best": float(np.median(bests)), "best_min": float(min(bests)),
            "best_max": float(max(bests)), "perm_p": float(np.median(ps)),
            "p_min": float(min(ps)), "p_max": float(max(ps)),
            "rejection_rate": rej, "draws": draws,
        }
        print(f"  oracle {acc:.0%}: median best {np.median(bests):6.2f}  median p={np.median(ps):.3f}"
              f"  rejects {rej:.0%} of {draws} draws", flush=True)

    # The lowest accuracy at which the benchmark rejects in a majority of independent draws.
    strong = [a for a, v in res["accuracies"].items() if v["rejection_rate"] >= 0.8]
    res["min_detectable_accuracy"] = min(strong) if strong else None

    # ---- information coefficient, configuration-invariant ------------------
    pos = {e["accession"]: i for i, e in enumerate(events)}
    llm_keys = [(m, p, t) for m in B.MODELS for p in B.PROMPTS for t in B.TRUNCS]
    raw = {k: {} for k in llm_keys}
    with open(os.path.join(B.DATA, "scores.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            k = (r["model"], r["prompt"], r["trunc"])
            if k in raw and r["accession"] in pos:
                raw[k][r["accession"]] = r.get("raw", "")
    ics = {}
    for k, d in raw.items():
        idx = [pos[a] for a in d]
        sc = np.array([B.parse_raw(k[1], d[a], False)[0] for a in d])
        rr = fwd[idx]
        ok = np.isfinite(rr) & price_ok[idx] & (np.std(sc) > 0)
        if ok.sum() > 100 and np.std(sc[ok]) > 0:
            rho = stats.spearmanr(sc[ok], rr[ok])
            ics[f"{k[0]}|{k[1]}|{k[2]}"] = {"ic": float(rho.statistic),
                                            "p": float(rho.pvalue), "n": int(ok.sum())}
    res["ic"] = ics
    if ics:
        vals = [v["ic"] for v in ics.values()]
        res["ic_summary"] = {"min": float(min(vals)), "max": float(max(vals)),
                             "mean": float(np.mean(vals)),
                             "n_significant_at_5pct": int(sum(v["p"] < 0.05 for v in ics.values())),
                             "n_configs": len(ics)}
        print("  IC across the 18 LLM configurations: "
              f"{res['ic_summary']['min']:+.3f} to {res['ic_summary']['max']:+.3f}, "
              f"{res['ic_summary']['n_significant_at_5pct']} significant at 5%", flush=True)

    path = os.path.join(B.DATA, "numbers.json")
    n = json.load(open(path, encoding="utf-8"))
    n["power"] = res
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(n, fh, indent=2, sort_keys=True)
    print("wrote data/numbers.json['power']")


if __name__ == "__main__":
    main()
