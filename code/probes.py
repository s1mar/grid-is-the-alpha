"""Two robustness probes that need the GPU, run after the main grid.

    mask   entity-masked scoring: is the model reading the filing, or recognising the issuer?
           Glasserman and Lin report that masking company identifiers changes measured
           performance, so asserting that our post-cutoff window settles contamination is not
           enough; the masked arm measures it.
    seed   stochastic-decoding scoring: how much Sharpe dispersion does sampling noise produce,
           against the dispersion that design choices produce? The main grid is greedy-decoded, so
           without this the paper cannot say the dispersion it reports is design rather than noise.

Both write to data/scores_probe.jsonl with a `probe` tag, so nothing here can contaminate the
main grid's score file.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score as S  # noqa: E402

OUT = os.path.join(S.DATA, "scores_probe.jsonl")
PROBE_MODEL = "qwen2.5:7b-instruct"
PROBE_PROMPT = "P1"
PROBE_TRUNC = 4000
SEEDS = [101, 202, 303]
TEMPERATURE = 0.7
CONCURRENCY = 16

# The scale arm. Roughly five times the parameter count of the largest model in the main grid, and
# the largest that runs at a usable rate on one consumer GPU. `think` is disabled explicitly:
# with reasoning left on, the model spends its whole token budget in the thinking channel and
# returns an EMPTY answer, which would have been silently recorded as a non-compliant response.
SCALE_MODEL = "qwen3.6:latest"
SCALE_THINK = False

_STOP = {"inc", "inc.", "corp", "corp.", "corporation", "company", "co", "co.", "the", "group",
         "holdings", "plc", "ltd", "&", "of", "and", "international", "american", "national"}


def mask_text(text: str, company: str, ticker: str) -> str:
    """Redact the issuer's identity from the filing text as far as a regex can."""
    tokens = [t for t in re.split(r"[^A-Za-z0-9&.]+", company) if t and t.lower() not in _STOP]
    pats = [re.escape(company)] + [re.escape(t) for t in tokens if len(t) > 2]
    pats.append(r"\b" + re.escape(ticker) + r"\b")
    out = text
    for p in sorted(pats, key=len, reverse=True):
        out = re.sub(p, "[REDACTED]", out, flags=re.I)
    return out


def done_keys() -> set:
    keys = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                keys.add((r["accession"], r["probe"], r.get("seed", 0)))
    return keys


def call(model, prompt_text, seed, temperature, think=None):
    payload = {
        "model": model, "prompt": prompt_text, "stream": False,
        "options": {"temperature": temperature, "seed": seed, "num_predict": S.NUM_PREDICT},
    }
    if think is not None:
        payload["think"] = think
    body = json.dumps(payload).encode()
    req = urllib.request.Request(S.OLLAMA, data=body,
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return r.get("response", "")


def run(jobs, label):
    lock = threading.Lock()
    state = {"n": 0, "err": 0, "t0": time.time()}
    with open(OUT, "a", encoding="utf-8") as fh:
        def work(job):
            rec = dict(job["rec"])
            try:
                rec["raw"] = call(job.get("model", PROBE_MODEL), job["prompt"], job["seed"],
                                  job["temperature"], job.get("think"))[:400]
            except Exception as exc:  # noqa: BLE001
                rec["raw"] = ""
                rec["error"] = type(exc).__name__
            with lock:
                fh.write(json.dumps(rec) + "\n")
                state["n"] += 1
                state["err"] += "error" in rec
                if state["n"] % 500 == 0:
                    fh.flush()
                    el = time.time() - state["t0"]
                    print(f"  {label} {state['n']}/{len(jobs)} "
                          f"({state['n']/el:.2f}/s, {el/60:.1f} min)", flush=True)
        with ThreadPoolExecutor(CONCURRENCY) as ex:
            list(ex.map(work, jobs))
    print(f"{label}: {state['n']} calls, {state['err']} errors, "
          f"{(time.time()-state['t0'])/60:.1f} min")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", choices=["mask", "seed", "scale", "all"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    events = S.load_events()
    sample = json.load(open(os.path.join(S.DATA, "event_sample.json"), encoding="utf-8"))
    keep = set(sample["accessions"])
    events = [e for e in events if e["accession"] in keep]
    if args.limit:
        events = events[: args.limit]
    have = done_keys()
    print(f"probe events: {len(events)}; {len(have)} probe calls already on disk")

    if args.probe in ("mask", "all"):
        jobs = []
        for e in events:
            if (e["accession"], "mask", 0) in have:
                continue
            masked = dict(e)
            masked["company"] = "the issuer"
            masked["ticker"] = "XXX"
            masked["text"] = mask_text(e["text"], e["company"], e["ticker"])
            jobs.append({"prompt": S.build_prompt(masked, PROBE_PROMPT, PROBE_TRUNC), "seed": S.SEED,
                         "temperature": 0.0,
                         "rec": {"accession": e["accession"], "ticker": e["ticker"],
                                 "model": PROBE_MODEL, "prompt": PROBE_PROMPT,
                                 "trunc": PROBE_TRUNC, "probe": "mask", "seed": S.SEED}})
        print(f"mask: {len(jobs)} calls")
        run(jobs, "mask")

    if args.probe in ("scale", "all"):
        jobs = []
        for e in events:
            if (e["accession"], "scale", S.SEED) in have:
                continue
            jobs.append({"prompt": S.build_prompt(e, PROBE_PROMPT, PROBE_TRUNC), "seed": S.SEED,
                         "temperature": 0.0, "model": SCALE_MODEL, "think": SCALE_THINK,
                         "rec": {"accession": e["accession"], "ticker": e["ticker"],
                                 "model": SCALE_MODEL, "prompt": PROBE_PROMPT,
                                 "trunc": PROBE_TRUNC, "probe": "scale", "seed": S.SEED}})
        print(f"scale: {len(jobs)} calls")
        run(jobs, "scale")

    if args.probe in ("seed", "all"):
        jobs = []
        for sd in SEEDS:
            for e in events:
                if (e["accession"], "seed", sd) in have:
                    continue
                jobs.append({"prompt": S.build_prompt(e, PROBE_PROMPT, PROBE_TRUNC), "seed": sd,
                             "temperature": TEMPERATURE,
                             "rec": {"accession": e["accession"], "ticker": e["ticker"],
                                     "model": PROBE_MODEL, "prompt": PROBE_PROMPT,
                                     "trunc": PROBE_TRUNC, "probe": "seed", "seed": sd}})
        print(f"seed: {len(jobs)} calls")
        run(jobs, "seed")


if __name__ == "__main__":
    main()
