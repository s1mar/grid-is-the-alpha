"""Render the paper's figures as vector PDFs into paper/. Reads only numbers.json and the arrays."""
from __future__ import annotations

import csv
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
PAPER = os.path.abspath(os.path.join(HERE, "..", "paper"))
GAMMA = 0.5772156649015329
ANN = math.sqrt(252.0)

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
    "font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42,  # Type 42: ACM's checker rejects Type 3 fonts
})

FACTOR_LABEL = {"model": "model", "prompt": "prompt", "trunc": "truncation", "parse": "parsing",
                "signal": "signal", "thresh": "threshold", "hold": "holding period",
                "entry": "entry timing", "weight": "weighting",
                "neutral": "neutralisation"}
LLM_SIDE = {"model", "prompt", "trunc"}


def load():
    n = json.load(open(os.path.join(DATA, "numbers.json"), encoding="utf-8"))
    with open(os.path.join(DATA, "grid.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    sr = np.array([float(r["sharpe"]) for r in rows])
    null_max = np.load(os.path.join(DATA, "null_max.npy"))
    return n, rows, sr, null_max


def fig_dispersion(n, sr, null_max):
    fig, ax = plt.subplots(figsize=(3.35, 2.15))
    lo = min(sr.min(), null_max.min()) - 0.1
    hi = max(sr.max(), null_max.max()) + 0.1
    bins = np.linspace(lo, hi, 80)
    ax.hist(sr, bins=bins, color="#4C72B0", alpha=0.85,
            label=f"{n['n_configs']:,} configurations")
    ax.hist(null_max, bins=bins, color="#C44E52", alpha=0.7,
            label=f"best of grid, scores permuted ({len(null_max)} draws)")
    # A log count axis, because a large mass of configurations that never trade would otherwise
    # flatten the whole distribution into one spike and hide the tails, which are the subject.
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.7)
    ymax = ax.get_ylim()[1]
    for k, (x, lab, style) in enumerate(((n["sharpe_default"], "default", "-"),
                                         (n["sharpe_median"], "median", "--"),
                                         (n["sharpe_best"], "best", ":"))):
        ax.axvline(x, color="k", lw=0.9, ls=style)
        ax.annotate(lab, xy=(x, ymax ** (0.48 - 0.15 * k)), xytext=(2, 0),
                    textcoords="offset points", fontsize=6, va="center", ha="left")
    ax.set_xlabel("annualised Sharpe ratio")
    ax.set_ylabel("configurations (log)")
    ax.legend(frameon=False, loc="upper left", handlelength=1.1, borderpad=0.1,
              labelspacing=0.25)
    fig.savefig(os.path.join(PAPER, "fig_dispersion.pdf"))
    plt.close(fig)


def fig_variance(n):
    shares = n["var_share"]
    items = [(FACTOR_LABEL[f], shares[f], f in LLM_SIDE) for f in FACTOR_LABEL]
    items.append(("interactions", shares["interactions"], None))
    items.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    colors = ["#8C8C8C" if s[2] is None else ("#C44E52" if s[2] else "#4C72B0") for s in items]
    ax.barh([s[0] for s in items], [100 * s[1] for s in items], color=colors, height=0.72)
    for i, s in enumerate(items):
        ax.text(100 * s[1] + 0.4, i, f"{100*s[1]:.1f}", va="center", fontsize=6)
    ax.set_xlabel("share of Sharpe variance across the grid (%)")
    ax.set_xlim(0, max(100 * s[1] for s in items) * 1.18)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#C44E52", "#4C72B0", "#8C8C8C")]
    ax.legend(handles, ["LLM-side choice", "post-LLM choice", "interactions"],
              frameon=False, loc="lower right", handlelength=1.0)
    fig.savefig(os.path.join(PAPER, "fig_variance.pdf"))
    plt.close(fig)


def fig_emax(n, sr, null_max):
    from scipy import stats
    var_sr = n["var_trial_sr_daily"]

    def emax(N):
        N = np.maximum(N, 1.0000001)
        z1 = stats.norm.ppf(1 - 1 / N)
        z2 = stats.norm.ppf(1 - 1 / (N * math.e))
        return math.sqrt(var_sr) * ((1 - GAMMA) * z1 + GAMMA * z2) * ANN

    # Equation (1) is an asymptotic in N and diverges as N approaches 1, so the curve is drawn from
    # N = 2. The N = 1 gridline is still marked, because that is where a paper reporting a single
    # configuration implicitly sits.
    hi = max(n["n_configs"] * 4, n.get("n_implied", 0) * 1.5)
    grid = np.logspace(math.log10(2), math.log10(hi), 300)
    fig, ax = plt.subplots(figsize=(3.35, 2.1))
    # No equation number in the legend. matplotlib cannot resolve a LaTeX cross-reference, so a
    # hardcoded "(1)" is a claim nothing checks; this one was wrong, pointing at the effective-trials
    # equation instead of the expected-maximum one. The caption carries the reference, where LaTeX
    # verifies it.
    ax.plot(grid, emax(grid), color="#4C72B0", lw=1.3,
            label="expected best-of-$N$, predicted")
    ax.axhline(n["emax_measured"], color="#C44E52", lw=1.3, ls="--",
               label="measured best of grid, scores permuted")
    ax.fill_between(grid, n["emax_measured"] - n["null_max_sd"],
                    n["emax_measured"] + n["null_max_sd"], color="#C44E52", alpha=0.15)
    marks = [(1, "$N{=}1$"), (n["n_eff"], "$N_{\\mathrm{eff}}$"),
             (n["n_configs"], "full grid")]
    if n.get("n_implied") and not n.get("n_implied_saturated"):
        marks.append((n["n_implied"], "$N$ implied"))
    lo_y = ax.get_ylim()[0]
    for x, lab in marks:
        ax.axvline(x, color="k", lw=0.6, ls=":")
        ax.text(x, lo_y + 0.02, f" {lab}", rotation=90, fontsize=6, va="bottom", ha="left")
    ax.set_xscale("log")
    ax.set_xlabel("number of trials $N$ assumed")
    ax.set_ylabel("annualised Sharpe")
    ax.legend(frameon=False, loc="upper left", handlelength=1.3)
    fig.savefig(os.path.join(PAPER, "fig_emax.pdf"))
    plt.close(fig)


def main() -> None:
    n, rows, sr, null_max = load()
    fig_dispersion(n, sr, null_max)
    fig_variance(n)
    # fig_emax is retained above but no longer rendered: the manuscript dropped that figure
    # for length and states the crossing point (N implied) in prose instead. Emitting it
    # would leave an orphan PDF in the artifact that no page references.
    print("wrote fig_dispersion.pdf, fig_variance.pdf")


if __name__ == "__main__":
    main()
