"""Does the post-LLM dispersion survive when a REAL signal is present?

Why this exists. Every dispersion figure in the paper is measured on a grid whose scores carry no
detectable signal (the score-level IC runs -0.040 to +0.009). A reviewer can therefore say the
headline phenomenon is only ever demonstrated in a degenerate regime: of course implementation
choices move a number that is pure noise. The synthetic oracle answers this directly, because it
injects a signal of KNOWN strength through the identical post-LLM grid. The power sweep already
used it, but extracted only rejection rates and best-of-grid Sharpes, never the grid distribution.

This extracts the distribution. No permutation nulls are needed for the question "is the spread
still there", so it costs one grid evaluation per draw rather than one per replication, and runs
in seconds rather than the hour the power sweep took.

Writes data/numbers.json["oracle_dispersion"].
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import backtest as B  # noqa: E402
import power as P  # noqa: E402

ACCURACIES = [0.55, 0.60, 0.70]
DRAWS = 10
SEED = 20260805


def main() -> None:
    events, pre, price_ok, spy_win, spy_ret, win = P.setup()
    fwd = P.event_forward_return(events, pre, win)
    rng = np.random.default_rng(SEED)
    tradable = price_ok & (fwd != 0)
    truth = np.sign(fwd)

    out = {"draws": DRAWS, "accuracies": {}}
    for acc in ACCURACIES:
        sds, ranges, meds, maxs, lives = [], [], [], [], []
        for _ in range(DRAWS):
            keep = rng.random(len(events)) < acc
            d = np.where(keep, truth, -truth)
            s = 0.5 + 0.4 * d
            sr, _ = P.grid_for_scores(events, pre, spy_win, price_ok, s, tradable.copy())
            sds.append(float(sr.std(ddof=1)))
            ranges.append(float(sr.max() - sr.min()))
            meds.append(float(np.median(sr)))
            maxs.append(float(sr.max()))
            lives.append(int((sr != 0).sum()))
        out["accuracies"][f"{acc:.2f}"] = {
            "post_llm_sd_median": float(np.median(sds)),
            "post_llm_sd_min": float(min(sds)), "post_llm_sd_max": float(max(sds)),
            "range_median": float(np.median(ranges)),
            "grid_median_median": float(np.median(meds)),
            "grid_best_median": float(np.median(maxs)),
            "n_nonzero_median": int(np.median(lives)),
        }
        print(f"  oracle {acc:.0%}: post-LLM sd {np.median(sds):5.2f} "
              f"(range {min(sds):.2f}-{max(sds):.2f})  spread {np.median(ranges):5.2f}  "
              f"median {np.median(meds):+5.2f}  best {np.median(maxs):5.2f}", flush=True)

    # The comparison that answers the objection: the same 432-configuration post-LLM grid measured
    # on the real scores. If the oracle's spread is of the same order, the phenomenon is not an
    # artefact of there being no signal to disturb.
    n = json.load(open(os.path.join(B.DATA, "numbers.json"), encoding="utf-8"))
    ref = n.get("probes", {}).get("main", {})
    if ref:
        out["real_reference_post_llm_sd"] = float(ref["sharpe_sd"])
        med = out["accuracies"][f"{ACCURACIES[1]:.2f}"]["post_llm_sd_median"]
        out["ratio_oracle60_over_real"] = float(med / ref["sharpe_sd"])
        print(f"\n  real reference arm post-LLM sd: {ref['sharpe_sd']:.2f}")
        print(f"  oracle at 60% / real          : {out['ratio_oracle60_over_real']:.2f}x")
    n["oracle_dispersion"] = out
    with open(os.path.join(B.DATA, "numbers.json"), "w", encoding="utf-8") as fh:
        json.dump(n, fh, indent=2, sort_keys=True)
    print('wrote data/numbers.json["oracle_dispersion"]')


if __name__ == "__main__":
    main()
