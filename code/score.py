"""Run the pre-registered LLM-side grid over every 8-K event and store RAW model output.

Parsing is deliberately NOT done here. `parse` is one of the pre-registered post-LLM factors, so
the raw string has to survive to the analysis stage; a runner that parsed on the way in would
silently collapse one arm of the grid.

Design notes that matter:
  * The work is chunked by event, and every chunk runs all three models, so an interrupted run
    still leaves a BALANCED design rather than one complete model and two empty ones.
  * Resumable on the exact key (accession, model, prompt, trunc).
  * Concurrency is bounded; Ollama holds one model at a time and this process owns the lock.

    python score.py            # run everything still missing
    python score.py --limit 40 # smoke test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
OUT = os.path.join(DATA, "scores.jsonl")
OLLAMA = "http://localhost:11434/api/generate"

# --- the frozen LLM-side grid (notes/preregistration.md section 3) -----------
MODELS = ["qwen2.5:7b-instruct", "mistral:7b-instruct-v0.2-q4_0", "phi3:mini"]
PROMPTS = ["P1", "P2", "P3"]
TRUNCS = [1200, 4000]

CONCURRENCY = 16
CHUNK = 500          # events per chunk; each chunk is completed by all three models
NUM_PREDICT = 64     # enough for a verbose model to finish a sentence containing its answer
SEED = 7
EVENT_LIMIT = 3000   # pre-declared random subsample; see DECISIONS.md A-3

P1 = """You are an experienced equity research analyst. Below is an excerpt from a Form 8-K filed \
with the SEC by {company} (ticker {ticker}).

Estimate the probability that this company's stock will OUTPERFORM the S&P 500 index over the next \
trading day.

FILING EXCERPT:
{text}

Respond with a single integer from 0 to 100 and nothing else."""

P2 = """Read the excerpt from a Form 8-K filed with the SEC by {company} (ticker {ticker}) and \
classify its likely short-term effect on the filer's share price relative to the market.

FILING EXCERPT:
{text}

Answer on one line in exactly this format:
LABEL CONFIDENCE
where LABEL is one of POSITIVE, NEGATIVE, NEUTRAL and CONFIDENCE is an integer from 0 to 100."""

P3 = """You are a quantitative analyst. Given the excerpt from a Form 8-K filed with the SEC by \
{company} (ticker {ticker}), output a single JSON object and nothing else:

{{"view": "up" | "down" | "none", "score": <number between -1 and 1>}}

Use "none" when the filing carries no tradable information. A positive score means the stock is \
likely to outperform the market; a negative score means it is likely to underperform.

FILING EXCERPT:
{text}"""

TEMPLATES = {"P1": P1, "P2": P2, "P3": P3}


def load_events() -> list[dict]:
    """Assemble one text blob per event: item codes, then exhibits, then the primary document.

    Substance first. An earnings 8-K's primary document is a cover page whose only content is a
    pointer to Exhibit 99.1, so truncating a cover-first assembly would hand every model an empty
    filing and then blame the model. This ordering is FIXED, not a factor: it is the assembly most
    favourable to the signal at every truncation length, which is the conservative direction for a
    paper arguing that the measured signal is fragile.
    """
    with open(os.path.join(DATA, "universe.csv"), encoding="utf-8") as fh:
        names = {r["ticker"]: r["name"] for r in csv.DictReader(fh)}

    exhibits = {}
    ex_path = os.path.join(DATA, "exhibits.jsonl")
    if os.path.exists(ex_path):
        with open(ex_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    try:
                        r = json.loads(line)
                    except Exception:  # noqa: BLE001 - torn last line of a running fetch
                        continue
                    exhibits[r["accession"]] = r["text"]

    events = {}
    with open(os.path.join(DATA, "docs.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            blob = (f"8-K ITEMS: {d['items']}\n\n"
                    + exhibits.get(d["accession"], "") + "\n\n" + d["text"]).strip()
            rec = {"accession": d["accession"], "ticker": d["ticker"],
                   "company": names.get(d["ticker"], d["ticker"]), "text": blob}
            # One filing, one event. Three CIKs in the universe carry two share classes each
            # (GOOG/GOOGL, FOX/FOXA, NWS/NWSA), so the same accession arrives twice and would be
            # traded twice on one piece of news. Keep the alphabetically first ticker: a fixed
            # rule, applied identically to every configuration.
            prev = events.get(d["accession"])
            if prev is None or d["ticker"] < prev["ticker"]:
                events[d["accession"]] = rec
    events = sorted(events.values(), key=lambda e: e["accession"])
    random.Random(20260731).shuffle(events)   # fixed shuffle -> partial runs are unbiased
    return events


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
                except Exception:  # noqa: BLE001 - a torn final line from a killed run
                    continue
                keys.add((r["accession"], r["model"], r["prompt"], r["trunc"]))
    return keys


def build_prompt(ev: dict, prompt: str, trunc: int) -> str:
    return TEMPLATES[prompt].format(
        company=ev["company"], ticker=ev["ticker"], text=ev["text"][:trunc])


def call(model: str, prompt_text: str) -> tuple[str, dict]:
    body = json.dumps({
        "model": model, "prompt": prompt_text, "stream": False,
        "options": {"temperature": 0.0, "seed": SEED, "num_predict": NUM_PREDICT},
    }).encode()
    req = urllib.request.Request(OLLAMA, data=body,
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return r.get("response", ""), {
        "ptok": r.get("prompt_eval_count"), "otok": r.get("eval_count")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only this many events (smoke test)")
    args = ap.parse_args()

    events = load_events()
    events = events[: args.limit or EVENT_LIMIT]
    with open(os.path.join(DATA, "event_sample.json"), "w", encoding="utf-8") as fh:
        json.dump({"n": len(events), "limit": args.limit or EVENT_LIMIT,
                   "accessions": [e["accession"] for e in events]}, fh)
    have = done_keys()
    total = len(events) * len(MODELS) * len(PROMPTS) * len(TRUNCS)
    print(f"events {len(events)} x {len(MODELS)}x{len(PROMPTS)}x{len(TRUNCS)} = {total} calls; "
          f"{len(have)} already on disk")

    lock = threading.Lock()
    fh = open(OUT, "a", encoding="utf-8")
    state = {"n": 0, "err": 0, "t0": time.time()}

    def work(job):
        ev, model, prompt, trunc = job
        try:
            raw, meta = call(model, build_prompt(ev, prompt, trunc))
            rec = {"accession": ev["accession"], "ticker": ev["ticker"], "model": model,
                   "prompt": prompt, "trunc": trunc, "raw": raw[:400], **meta}
        except Exception as exc:  # noqa: BLE001
            rec = {"accession": ev["accession"], "ticker": ev["ticker"], "model": model,
                   "prompt": prompt, "trunc": trunc, "raw": "", "error": type(exc).__name__}
        with lock:
            fh.write(json.dumps(rec) + "\n")
            state["n"] += 1
            if "error" in rec:
                state["err"] += 1
            if state["n"] % 500 == 0:
                fh.flush()
                el = time.time() - state["t0"]
                print(f"  {state['n']} calls, {state['err']} errors, "
                      f"{state['n']/el:.2f} calls/s, elapsed {el/60:.1f} min", flush=True)

    for start in range(0, len(events), CHUNK):
        chunk = events[start: start + CHUNK]
        print(f"\n--- chunk {start//CHUNK + 1}/{-(-len(events)//CHUNK)} "
              f"({len(chunk)} events) ---", flush=True)
        for model in MODELS:                       # model outermost within a chunk: 1 load each
            jobs = [(ev, model, p, t) for ev in chunk for p in PROMPTS for t in TRUNCS
                    if (ev["accession"], model, p, t) not in have]
            if not jobs:
                continue
            print(f"  {model}: {len(jobs)} calls", flush=True)
            with ThreadPoolExecutor(CONCURRENCY) as ex:
                list(ex.map(work, jobs))
            fh.flush()

    fh.close()
    el = time.time() - state["t0"]
    print(f"\ndone: {state['n']} calls, {state['err']} errors, {el/60:.1f} min")
    if state["err"] > 0.01 * max(state["n"], 1):
        print("WARNING: error rate above 1%")
        sys.exit(1)


if __name__ == "__main__":
    main()
