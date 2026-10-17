"""Evaluate the whole pre-registered configuration grid and write one row per configuration.

Everything here is vectorised. The grid is 18 LLM-side configurations times 432 post-LLM
configurations = 7,776 backtests, plus R permutation replications of the same grid, so a per-config
Python loop would not finish. The trick is that only the weight vector depends on the post-LLM
choices up to `(entry, hold)`, so each `(entry, hold)` block is one dense matrix product.

Outputs (all under ../data):
    grid.csv          one row per configuration: the ten factor levels and its statistics
    grid_returns.npy  (n_config, T) float32 daily portfolio returns, for the correlation work
    null_max.npy      (R,) best-of-grid Sharpe under the within-session permutation null
    backtest_meta.json
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
ET = ZoneInfo("America/New_York")

# --- frozen grid (notes/preregistration.md section 3) ------------------------
MODELS = ["qwen2.5:7b-instruct", "mistral:7b-instruct-v0.2-q4_0", "phi3:mini"]
PROMPTS = ["P1", "P2", "P3"]
TRUNCS = [1200, 4000]
PARSES = ["strict", "lenient"]
SIGNALS = ["level", "xrank", "demean"]
THRESHES = [0.00, 0.10, 0.20]
HOLDS = [1, 3, 5]
ENTRIES = ["open", "close"]
WEIGHTS = ["equal", "proportional"]
NEUTRALS = ["none", "market"]

COST = 0.0005      # 5 bp per side, fixed for every configuration
ANN = np.sqrt(252.0)
MAX_HOLD = max(HOLDS)


# --------------------------------------------------------------------- prices
def load_prices():
    rows = []
    with gzip.open(os.path.join(DATA, "prices.csv.gz"), "rt", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    tickers = sorted({r["ticker"] for r in rows} - {"SPY"})
    sessions = sorted({r["date"] for r in rows if r["ticker"] == "SPY"})
    t_idx = {t: i for i, t in enumerate(tickers)}
    s_idx = {d: i for i, d in enumerate(sessions)}
    n_t, n_s = len(tickers), len(sessions)
    op = np.full((n_s, n_t), np.nan, dtype=np.float64)
    cl = np.full((n_s, n_t), np.nan, dtype=np.float64)
    spy = np.full(n_s, np.nan)
    for r in rows:
        d = s_idx.get(r["date"])
        if d is None:
            continue
        try:
            o, c, a = float(r["open"]), float(r["close"]), float(r["adjclose"])
        except (TypeError, ValueError):
            continue
        if not (c > 0) or not (o > 0) or not (a > 0):
            continue
        if r["ticker"] == "SPY":
            spy[d] = a
            continue
        j = t_idx.get(r["ticker"])
        if j is None:
            continue
        cl[d, j] = a
        op[d, j] = o * (a / c)        # carry the split/dividend adjustment onto the open
    spy_ret = np.zeros(n_s)
    spy_ret[1:] = spy[1:] / spy[:-1] - 1.0
    return tickers, sessions, t_idx, s_idx, op, cl, spy_ret


# --------------------------------------------------------------------- events
def entry_session(acceptance: str, sessions: list[str]) -> int:
    """First regular session that opens strictly after the filing was accepted."""
    dt = datetime.strptime(acceptance, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    et = dt.astimezone(ET)
    day = et.strftime("%Y-%m-%d")
    before_open = (et.hour, et.minute) < (9, 30)
    lo, hi = 0, len(sessions)
    while lo < hi:                      # first session >= day
        mid = (lo + hi) // 2
        if sessions[mid] < day:
            lo = mid + 1
        else:
            hi = mid
    if lo >= len(sessions):
        return -1
    if sessions[lo] == day:
        return lo if before_open else lo + 1
    return lo                            # filed on a non-session day: next session


def load_events(sessions, t_idx, s_idx):
    sample_path = os.path.join(DATA, "event_sample.json")
    sample = None
    if os.path.exists(sample_path):
        sample = set(json.load(open(sample_path, encoding="utf-8"))["accessions"])
    uniq = {}
    with open(os.path.join(DATA, "docs.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            rec = {"accession": d["accession"], "ticker": d["ticker"],
                   "filing_date": d["filing_date"], "acceptance": d["acceptance"]}
            # Same dual-share-class deduplication as score.py: one filing, one event.
            prev = uniq.get(d["accession"])
            if prev is None or d["ticker"] < prev["ticker"]:
                uniq[d["accession"]] = rec
    evs = sorted(uniq.values(), key=lambda e: e["accession"])
    if sample is not None:
        evs = [e for e in evs if e["accession"] in sample]
    keep = []
    for e in evs:
        j = t_idx.get(e["ticker"])
        if j is None:
            continue
        t0 = entry_session(e["acceptance"], sessions)
        # Require room for the longest holding period under BOTH entry rules, so that every
        # configuration sees the IDENTICAL event set. A hold-dependent event set would let the
        # sample size vary with a factor level and contaminate the variance decomposition.
        if t0 < 0 or t0 + MAX_HOLD >= len(sessions):
            continue
        e["tick"] = j
        e["t0"] = t0
        keep.append(e)
    return keep


# --------------------------------------------------------------------- parsing
# A lenient parser looks for "an integer" in free text. Two things it must not do, because both are
# bugs rather than leniency: match a fragment of a longer number ("202" inside "2026"), and match
# the document type named in our own prompt ("8" inside "8-K"). See DECISIONS.md A-4.
_INT = re.compile(r"(?<![\w.])(\d{1,3})(?!\d)(?!\.\d)(?!-[A-Z])")
_LAB = re.compile(r"\b(POSITIVE|NEGATIVE|NEUTRAL)\b", re.I)
_P1_STRICT = re.compile(r"^\s*(\d{1,3})\s*\.?\s*$")
_P2_STRICT = re.compile(r"^\s*(POSITIVE|NEGATIVE|NEUTRAL)\s+(\d{1,3})\s*$", re.I)
_P3_VIEW = re.compile(r'"view"\s*:\s*"(up|down|none)"', re.I)
_P3_SCORE = re.compile(r'"score"\s*:\s*(-?\d*\.?\d+)')


def _first_in_range(text: str):
    """First integer in [0, 100] in the text, or None.

    Out-of-range integers are skipped rather than clipped. Clipping turns "S&P 500" into a maximal
    bullish score, which is not leniency, it is a bug that manufactures a signal out of boilerplate.
    """
    for m in _INT.finditer(text):
        v = int(m.group(1))
        if 0 <= v <= 100:
            return v
    return None


def _lab_score(label: str, conf: float) -> float:
    conf = min(max(conf, 0.0), 100.0) / 100.0
    if label.upper() == "POSITIVE":
        return 0.5 + 0.5 * conf
    if label.upper() == "NEGATIVE":
        return 0.5 - 0.5 * conf
    return 0.5


def parse_raw(prompt: str, raw: str, strict: bool):
    """Return (score in [0,1], valid). `valid` False means a strict parser rejects the answer."""
    raw = (raw or "").strip()
    if prompt == "P1":
        m = _P1_STRICT.match(raw)
        if m:
            return min(max(int(m.group(1)), 0), 100) / 100.0, True
        if strict:
            return 0.5, False
        v = _first_in_range(raw)
        return (v / 100.0, True) if v is not None else (0.5, True)
    if prompt == "P2":
        m = _P2_STRICT.match(raw)
        if m:
            return _lab_score(m.group(1), float(m.group(2))), True
        if strict:
            return 0.5, False
        lab = _LAB.search(raw)
        if lab:
            # Prefer a confidence stated AFTER the label. `or` would be wrong here: a confidence
            # of exactly 0 is falsy and would silently fall through to a number elsewhere.
            v = _first_in_range(raw[lab.end():])
            if v is None:
                v = _first_in_range(raw)
            return _lab_score(lab.group(1), float(v) if v is not None else 50.0), True
        return 0.5, True
    # P3
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "view" in obj and "score" in obj:
            v = str(obj["view"]).lower()
            sc = float(obj["score"])
            return (0.5 if v == "none" else 0.5 + min(max(sc, -1.0), 1.0) / 2.0), True
    except Exception:  # noqa: BLE001 - any malformed answer
        pass
    if strict:
        return 0.5, False
    mv, ms = _P3_VIEW.search(raw), _P3_SCORE.search(raw)
    if mv or ms:
        if mv and mv.group(1).lower() == "none":
            return 0.5, True
        sc = float(ms.group(1)) if ms else 0.0
        return 0.5 + min(max(sc, -1.0), 1.0) / 2.0, True
    return 0.5, True


def load_scores(events):
    """-> S[llm, parse, event], V[llm, parse, event]; llm indexes MODELS x PROMPTS x TRUNCS."""
    pos = {e["accession"]: i for i, e in enumerate(events)}
    llm_keys = [(m, p, t) for m in MODELS for p in PROMPTS for t in TRUNCS]
    llm_idx = {k: i for i, k in enumerate(llm_keys)}
    E, L = len(events), len(llm_keys)
    raw = [[None] * E for _ in range(L)]
    n_lines = n_used = 0
    with open(os.path.join(DATA, "scores.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            i = pos.get(r["accession"])
            k = llm_idx.get((r["model"], r["prompt"], r["trunc"]))
            if i is None or k is None:
                continue
            raw[k][i] = r.get("raw", "")
            n_used += 1
    S = np.full((L, 2, E), 0.5, dtype=np.float64)
    V = np.zeros((L, 2, E), dtype=bool)
    for k in range(L):
        prompt = llm_keys[k][1]
        for i in range(E):
            r = raw[k][i]
            if r is None:
                continue                     # call missing: invalid under both parsers
            for pi, parse in enumerate(PARSES):
                s, ok = parse_raw(prompt, r, parse == "strict")
                S[k, pi, i] = s
                V[k, pi, i] = ok
    return llm_keys, S, V, {"score_lines": n_lines, "score_used": n_used}


# --------------------------------------------------------------------- returns
def precompute(events, op, cl, n_sessions):
    """R[(entry,hold)] -> (E, T) daily returns; A -> active mask; F/L -> first/last active day."""
    E, T = len(events), n_sessions
    tick = np.array([e["tick"] for e in events])
    t0 = np.array([e["t0"] for e in events])
    out = {}
    ok_all = np.ones(E, dtype=bool)
    for entry in ENTRIES:
        for hold in HOLDS:
            R = np.zeros((E, T), dtype=np.float32)
            A = np.zeros((E, T), dtype=np.float32)
            F = np.zeros((E, T), dtype=np.float32)
            Lm = np.zeros((E, T), dtype=np.float32)
            ok = np.ones(E, dtype=bool)
            for i in range(E):
                j, s = tick[i], t0[i]
                if entry == "open":
                    days = list(range(s, s + hold))
                    p_entry = op[s, j]
                else:
                    days = list(range(s + 1, s + hold + 1))
                    p_entry = cl[s, j]
                if not np.isfinite(p_entry) or days[-1] >= T:
                    ok[i] = False
                    continue
                prices = [p_entry] + [cl[d, j] for d in days]
                if not all(np.isfinite(p) and p > 0 for p in prices):
                    ok[i] = False
                    continue
                for k, d in enumerate(days):
                    R[i, d] = prices[k + 1] / prices[k] - 1.0
                    A[i, d] = 1.0
                F[i, days[0]] = 1.0
                Lm[i, days[-1]] = 1.0
            out[(entry, hold)] = (R, A, F, Lm)
            ok_all &= ok
    return out, ok_all


# --------------------------------------------------------------------- signals
def build_signals(events, S, V, price_ok):
    """-> G[llm, parse, signal, event] signed signal, and M[...] tradability mask."""
    L, _, E = S.shape
    t0 = np.array([e["t0"] for e in events])
    tick = np.array([e["tick"] for e in events])
    order = np.argsort(t0, kind="stable")
    sess_groups = {}
    for i in order:
        sess_groups.setdefault(int(t0[i]), []).append(i)
    tick_order = {}
    for i in order:                       # events of one ticker, in entry-session order
        tick_order.setdefault(int(tick[i]), []).append(i)

    G = np.zeros((L, 2, len(SIGNALS), E))
    M = np.zeros((L, 2, len(SIGNALS), E), dtype=bool)
    for k in range(L):
        for pi in range(2):
            s = S[k, pi]
            v = V[k, pi] & price_ok
            # level
            G[k, pi, 0] = s - 0.5
            M[k, pi, 0] = v
            # cross-sectional rank inside the ENTRY SESSION (see DECISIONS A-1)
            g = np.zeros(E)
            m = np.zeros(E, dtype=bool)
            for _sess, idxs in sess_groups.items():
                sel = [i for i in idxs if v[i]]
                n = len(sel)
                if n < 2:
                    continue
                vals = s[sel]
                rank = np.empty(n)
                srt = np.argsort(vals, kind="stable")
                rank[srt] = np.arange(n, dtype=float)
                # average ranks over ties so that identical scores get identical signals
                uniq, inv = np.unique(vals, return_inverse=True)
                for u in range(len(uniq)):
                    sel_u = inv == u
                    rank[sel_u] = rank[sel_u].mean()
                g_sel = rank / (n - 1) - 0.5
                for a, i in enumerate(sel):
                    g[i] = g_sel[a]
                    m[i] = True
            G[k, pi, 1] = g
            M[k, pi, 1] = m
            # expanding per-ticker demeaning, strictly prior events only
            g = np.zeros(E)
            m = np.zeros(E, dtype=bool)
            for _t, idxs in tick_order.items():
                run_sum, run_n = 0.0, 0
                for i in idxs:
                    if not v[i]:
                        continue
                    if run_n > 0:
                        g[i] = s[i] - run_sum / run_n
                        m[i] = True
                    run_sum += s[i]
                    run_n += 1
            G[k, pi, 2] = g
            M[k, pi, 2] = m
    return G, M


# --------------------------------------------------------------------- grid
def sharpe(r: np.ndarray) -> np.ndarray:
    sd = r.std(axis=-1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        val = r.mean(axis=-1) / np.where(sd > 0, sd, np.nan) * ANN
    return np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)


def weight_matrix(G, M):
    """-> W (n_w, E) and the index tuples (llm, parse, signal, thresh, weight) for each row."""
    L = G.shape[0]
    rows, keys = [], []
    for k in range(L):
        for pi in range(2):
            for si in range(len(SIGNALS)):
                g = G[k, pi, si]
                m = M[k, pi, si]
                for ti, th in enumerate(THRESHES):
                    trade = m & (np.abs(g) > th)
                    d = np.sign(g) * trade
                    for wi, wname in enumerate(WEIGHTS):
                        mag = np.ones_like(g) if wname == "equal" else np.abs(g)
                        rows.append((d * mag).astype(np.float32))
                        keys.append((k, pi, si, ti, wi))
    return np.array(rows, dtype=np.float32), keys


def run_grid(W, keys, pre, spy_ret):
    """-> returns (n_config, T) float32 and the parallel list of config dicts."""
    T = spy_ret.shape[0]
    n_cfg = len(keys) * len(pre) * len(NEUTRALS)
    out = np.zeros((n_cfg, T), dtype=np.float32)
    cfgs = []
    absW = np.abs(W)
    row = 0
    for (entry, hold), (R, A, F, Lm) in pre.items():
        num = W @ R                       # (n_w, T)
        den = absW @ A
        net = W @ A
        turn = absW @ (F + Lm)
        safe = np.where(den > 0, den, 1.0)
        base = num / safe
        netx = net / safe
        cost = COST * turn / safe
        base = base - cost
        base[den <= 0] = 0.0
        netx[den <= 0] = 0.0
        for ni, neutral in enumerate(NEUTRALS):
            block = base if neutral == "none" else base - netx * spy_ret[None, :].astype(np.float32)
            out[row:row + len(keys)] = block
            for (k, pi, si, ti, wi) in keys:
                cfgs.append({"llm": k, "parse": pi, "signal": si, "thresh": ti,
                             "weight": wi, "entry": entry, "hold": hold, "neutral": ni})
            row += len(keys)
    return out, cfgs


def main() -> None:
    t_start = time.time()
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    print("loading prices...", flush=True)
    tickers, sessions, t_idx, s_idx, op, cl, spy_ret = load_prices()
    print(f"  {len(tickers)} tickers, {len(sessions)} sessions "
          f"{sessions[0]}..{sessions[-1]}", flush=True)

    events = load_events(sessions, t_idx, s_idx)
    print(f"  {len(events)} tradable events", flush=True)

    llm_keys, S, V, smeta = load_scores(events)
    print(f"  scores: {smeta['score_used']} of {smeta['score_lines']} rows matched", flush=True)
    cover = V.any(axis=1).mean(axis=1)
    print("  per-LLM-config coverage: " + ", ".join(f"{c:.2f}" for c in cover), flush=True)

    pre, price_ok = precompute(events, op, cl, len(sessions))
    print(f"  price-complete events: {int(price_ok.sum())}/{len(events)} "
          f"({time.time()-t_start:.0f}s)", flush=True)

    # Evaluate over the live window only. The price panel starts three months before the first
    # filing so that the pre-period exists for reference; scoring the strategy over sessions in
    # which no configuration can hold anything would divide every Sharpe ratio by the same
    # arbitrary constant. The window is identical for every configuration.
    t_lo = min(e["t0"] for e in events)
    t_hi = min(max(e["t0"] for e in events) + MAX_HOLD, len(sessions) - 1)
    win = slice(t_lo, t_hi + 1)
    spy_win = np.ascontiguousarray(spy_ret[win])
    pre = {k: tuple(np.ascontiguousarray(a[:, win]) for a in v) for k, v in pre.items()}
    n_win = t_hi - t_lo + 1
    print(f"  evaluation window: sessions {sessions[t_lo]}..{sessions[t_hi]} "
          f"({n_win} sessions)", flush=True)

    G, M = build_signals(events, S, V, price_ok)
    W, keys = weight_matrix(G, M)
    print(f"  {W.shape[0]} weight vectors ({time.time()-t_start:.0f}s)", flush=True)

    R, cfgs = run_grid(W, keys, pre, spy_win)
    sr = sharpe(R.astype(np.float64))
    print(f"  grid: {R.shape[0]} configurations, Sharpe "
          f"min {sr.min():.2f} / med {np.median(sr):.2f} / max {sr.max():.2f} "
          f"({time.time()-t_start:.0f}s)", flush=True)

    # Cost sensitivity. Frictions are the first thing a finance reviewer asks about, and the answer
    # is cheap: the grid is one matrix product per (entry, hold) block, so re-running it at four
    # cost levels costs seconds. Reported as a summary line, not as a factor, because the cost
    # assumption is a modelling choice we make once rather than a researcher degree of freedom.
    global COST
    cost_sweep = {}
    base_cost = COST
    for c in (0.0, 0.0005, 0.0010, 0.0025):
        COST = c
        Rc, _ = run_grid(W, keys, pre, spy_win)
        src = sharpe(Rc.astype(np.float64))
        cost_sweep[f"{c*1e4:.0f}bp"] = {
            "min": float(src.min()), "median": float(np.median(src)),
            "max": float(src.max()), "sd": float(src.std(ddof=1)),
            "frac_above_1": float((src > 1).mean())}
    COST = base_cost
    print("  cost sweep: " + ", ".join(f"{k} max {v['max']:.2f} sd {v['sd']:.2f}"
                                       for k, v in cost_sweep.items()), flush=True)

    np.save(os.path.join(DATA, "grid_returns.npy"), R)
    with open(os.path.join(DATA, "grid.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["config_id", "model", "prompt", "trunc", "parse", "signal", "thresh",
                    "hold", "entry", "weight", "neutral", "sharpe", "mean_daily", "sd_daily",
                    "n_active_days"])
        for i, c in enumerate(cfgs):
            m, p, t = llm_keys[c["llm"]]
            w.writerow([i, m, p, t, PARSES[c["parse"]], SIGNALS[c["signal"]],
                        THRESHES[c["thresh"]], c["hold"], c["entry"], WEIGHTS[c["weight"]],
                        NEUTRALS[c["neutral"]], f"{sr[i]:.6f}",
                        f"{R[i].mean():.8f}", f"{R[i].std(ddof=1):.8f}",
                        int((R[i] != 0).sum())])

    # --- exact within-session permutation null --------------------------------
    print(f"permutation null, {reps} replications...", flush=True)
    rng = np.random.default_rng(20260731)
    t0 = np.array([e["t0"] for e in events])
    groups = [np.where(t0 == s)[0] for s in np.unique(t0)]
    null_max = np.zeros(reps)
    null_sd = np.zeros(reps)     # between-configuration spread when the scores carry nothing
    for r in range(reps):
        perm = np.arange(len(events))
        for g in groups:
            if len(g) > 1:
                perm[g] = g[rng.permutation(len(g))]
        Sp, Vp = S[:, :, perm], V[:, :, perm]
        Gp, Mp = build_signals(events, Sp, Vp, price_ok)
        Wp, kp = weight_matrix(Gp, Mp)
        Rp, _ = run_grid(Wp, kp, pre, spy_win)
        srp = sharpe(Rp.astype(np.float64))
        null_max[r] = srp.max()
        null_sd[r] = srp.std(ddof=1)
        if (r + 1) % 25 == 0:
            print(f"  {r+1}/{reps} null max so far "
                  f"med {np.median(null_max[:r+1]):.2f} ({time.time()-t_start:.0f}s)", flush=True)
    np.save(os.path.join(DATA, "null_max.npy"), null_max)
    np.save(os.path.join(DATA, "null_sd.npy"), null_sd)

    # Strict-parse compliance per (model, prompt), averaged over truncation levels. This is a
    # descriptive statistic about the models, and it is what makes the `parse` factor consequential.
    # Denominator: events for which a response actually came back, not every event in the sample.
    # Dividing by the sample size would report a partial run's coverage as the model's compliance.
    answered = V[:, 0] | V[:, 1]        # lenient accepts anything that arrived, so this is "a call
    compliance = {}                     # returned something", not "the answer was well formed"
    for k, (m, p, _t) in enumerate(llm_keys):
        denom = int(answered[k].sum())
        if denom:
            compliance.setdefault(f"{m}|{p}", []).append(float(V[k, 0].sum()) / denom)
    compliance = {k: round(sum(v) / len(v), 4) for k, v in compliance.items()}

    # How much of the null is even permutable: a session with one event cannot be shuffled, so its
    # score stays attached to its filing and the null is conservative to that extent.
    sess_sizes = np.array([len(g) for g in groups])
    permutable = float((sess_sizes[sess_sizes > 1]).sum() / max(sess_sizes.sum(), 1))

    meta = {
        "n_events": len(events), "n_price_ok": int(price_ok.sum()), "compliance": compliance,
        "cost_sweep": cost_sweep,
        "frac_events_permutable": permutable,
        "num_predict": __import__("score").NUM_PREDICT,
        "median_events_per_session": float(np.median(sess_sizes)),
        "n_sessions": n_win, "first_session": sessions[t_lo], "last_session": sessions[t_hi],
        "n_panel_sessions": len(sessions),
        "n_configs": R.shape[0], "reps": reps, "cost_bps_per_side": COST * 1e4,
        "llm_keys": llm_keys, "score_meta": smeta,
        "coverage_per_llm_config": [float(c) for c in cover],
        "runtime_sec": round(time.time() - t_start, 1),
    }
    with open(os.path.join(DATA, "backtest_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps({k: meta[k] for k in
                      ("n_events", "n_price_ok", "n_configs", "reps", "runtime_sec")}, indent=2))


if __name__ == "__main__":
    main()
