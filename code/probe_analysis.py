"""Evaluate the robustness arms and append their results to data/numbers.json.

    mask   the same configuration scored on filings whose issuer identity has been redacted
    seed   the same configuration scored three times with stochastic decoding
    scale  the same configuration scored by a model roughly five times larger

Every arm is run through the IDENTICAL post-LLM grid as the main study and gets its own
within-session permutation null, so the comparison is between score sources and nothing else.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as B  # noqa: E402
import probes as P  # noqa: E402

PROBE_KEY = ("qwen2.5:7b-instruct", "P1", 4000)


ARM_PROBES = ("mask", "scale", "scale550")


def score_vectors(events, probe_path, main_path, or_path=None):
    """-> dict name -> (S[1,2,E], V[1,2,E]) for the main arm and each probe arm.

    `or_path` carries the 550B arm, which was scored through OpenRouter rather than Ollama and so
    lives in its own file. It is read here rather than analysed separately so that every arm goes
    through the identical post-LLM grid and the identical within-session permutation null; a
    separate code path would make the comparison an artefact of which script ran it.
    """
    pos = {e["accession"]: i for i, e in enumerate(events)}
    E = len(events)
    raw = {}
    with open(main_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if (r["model"], r["prompt"], r["trunc"]) == PROBE_KEY and r["accession"] in pos:
                raw.setdefault("main", {})[r["accession"]] = r.get("raw", "")
    if os.path.exists(probe_path):
        with open(probe_path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["accession"] not in pos:
                    continue
                name = r["probe"] if r["probe"] in ARM_PROBES else f"seed{r['seed']}"
                raw.setdefault(name, {})[r["accession"]] = r.get("raw", "")
    if or_path and os.path.exists(or_path):
        with open(or_path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                # A rate-limited call that exhausted its retries is NOT a model that answered with
                # nothing. Including these would report a harness fault as non-compliance, which is
                # the one number this arm exists to establish. See DECISIONS.md D-042.
                if r.get("failed") or r["accession"] not in pos:
                    continue
                raw.setdefault(r["probe"], {})[r["accession"]] = r.get("raw", "")

    out = {}
    for name, d in raw.items():
        S = np.full((1, 2, E), 0.5)
        V = np.zeros((1, 2, E), dtype=bool)
        for acc, text in d.items():
            i = pos[acc]
            for pi, parse in enumerate(B.PARSES):
                s, ok = B.parse_raw(PROBE_KEY[1], text, parse == "strict")
                S[0, pi, i] = s
                V[0, pi, i] = ok
        out[name] = (S, V, len(d))
    return out


def main() -> None:
    tickers, sessions, t_idx, s_idx, op, cl, spy_ret = B.load_prices()
    events = B.load_events(sessions, t_idx, s_idx)
    pre, price_ok = B.precompute(events, op, cl, len(sessions))
    t_lo = min(e["t0"] for e in events)
    t_hi = min(max(e["t0"] for e in events) + B.MAX_HOLD, len(sessions) - 1)
    win = slice(t_lo, t_hi + 1)
    spy_win = np.ascontiguousarray(spy_ret[win])
    pre = {k: tuple(np.ascontiguousarray(a[:, win]) for a in v) for k, v in pre.items()}

    arms = score_vectors(events, os.path.join(B.DATA, "scores_probe.jsonl"),
                         os.path.join(B.DATA, "scores.jsonl"),
                         os.path.join(B.DATA, "scores_or.jsonl"))
    if "main" not in arms:
        print("no main-arm scores for the probe configuration; nothing to do")
        return

    # Each arm gets its own within-session permutation null over its own post-LLM grid, so that
    # "did this arm's scores carry anything" is answered by the same instrument as the main study
    # rather than by comparing two point estimates.
    # 300 is what the manuscript reports (\ProbePermReps) and is therefore the default. It was 200,
    # which meant an argument-less run silently produced a LOWER replication count than the paper
    # claims, and passing "200" by hand once re-ran every probe arm at reduced precision and moved
    # every printed null mean and p-value. A default that disagrees with the paper is a trap.
    DEFAULT_REPS = 300
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REPS
    if reps < DEFAULT_REPS:
        print(f"WARNING: running at {reps} replications, BELOW the {DEFAULT_REPS} the manuscript "
              f"reports. Every probe p-value and null mean will change and the gate will fail.",
              flush=True)
    t0 = np.array([e["t0"] for e in events])
    groups = [np.where(t0 == s)[0] for s in np.unique(t0)]

    res = {}
    for name, (Sarr, Varr, n_scored) in sorted(arms.items()):
        G, M = B.build_signals(events, Sarr, Varr, price_ok)
        W, keys = B.weight_matrix(G, M)
        R, _ = B.run_grid(W, keys, pre, spy_win)
        sr = B.sharpe(R.astype(np.float64))

        rng = np.random.default_rng(20260801)
        null_max = np.zeros(reps)
        for r in range(reps):
            perm = np.arange(len(events))
            for g in groups:
                if len(g) > 1:
                    perm[g] = g[rng.permutation(len(g))]
            Gp, Mp = B.build_signals(events, Sarr[:, :, perm], Varr[:, :, perm], price_ok)
            Wp, kp = B.weight_matrix(Gp, Mp)
            Rp, _ = B.run_grid(Wp, kp, pre, spy_win)
            null_max[r] = B.sharpe(Rp.astype(np.float64)).max()

        res[name] = {"n_scored": n_scored, "n_configs": int(len(sr)),
                     "sharpe_median": float(np.median(sr)), "sharpe_max": float(sr.max()),
                     "sharpe_min": float(sr.min()), "sharpe_sd": float(sr.std(ddof=1)),
                     "null_max_mean": float(null_max.mean()),
                     "perm_p": float((1 + (null_max >= sr.max()).sum()) / (reps + 1)),
                     "perm_reps": reps,
                     "strict_compliance": float(Varr[0, 0].mean())}

    # How much of the score survives redaction of the issuer's identity?
    if "mask" in arms:
        Sm, Vm, _ = arms["mask"]
        S0, V0, _ = arms["main"]
        both = (Vm[0, 1] & V0[0, 1])
        if both.sum() > 10:
            res["mask"]["score_corr_with_main"] = float(
                np.corrcoef(Sm[0, 1][both], S0[0, 1][both])[0, 1])
            res["mask"]["mean_abs_score_shift"] = float(
                np.abs(Sm[0, 1][both] - S0[0, 1][both]).mean())

    # Stochastic decoding: spread across seeds, to be compared with the spread across designs.
    seeds = [k for k in res if k.startswith("seed") and k != "seed_summary"]
    if len(seeds) >= 2:
        meds = [res[k]["sharpe_median"] for k in seeds]
        res["seed_summary"] = {"n_seeds": len(seeds),
                               "median_sharpe_range": float(max(meds) - min(meds)),
                               "median_sharpe_sd": float(np.std(meds, ddof=1))}

    path = os.path.join(B.DATA, "numbers.json")
    n = json.load(open(path, encoding="utf-8"))
    n["probes"] = res
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(n, fh, indent=2, sort_keys=True)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
