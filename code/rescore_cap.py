"""Is the zero format compliance of some model-prompt cells a MODEL fact or a DECODE-CAP artifact?

Why this exists. `mistral|P1` and `mistral|P2` score exactly 0.0% strict compliance across 3,000
filings, and `phi3|P1` and `phi3|P3` are near zero. Those cells produce the 1,152 never-trading
configurations and the manuscript's concrete example of a model x parsing interaction. Inspecting
the raw strings shows they average 290 characters against a 64-token cap and end mid-word: the
model was still explaining and never reached its answer. If a larger cap recovers compliance, then
the interaction the paper displays is partly a property of an unablated harness constant, and the
sentence "a compliance rate is a property of the weights that produced it" is incomplete.

TWO-SIDED CONTROL. The cells that already comply are re-scored at the SAME higher cap. They must
STAY compliant. If they do not, the experiment changed something other than the cap and no
conclusion may be drawn from the failing cells either.

Everything is local and free. Writes data/cap_probe.json; touches nothing the paper reads until
the result is inspected.

    python rescore_cap.py [n_events] [cap]
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import backtest as B  # noqa: E402
import score as S  # noqa: E402

OUT = os.path.join(S.DATA, "cap_probe.json")
TRUNC = 4000
SEED = 20260805

# Grouped BY MODEL: Ollama holds one model at a time, so interleaving models would evict and
# reload many gigabytes per call. Each tuple is (model, prompt, why it is here).
CELLS = [
    ("mistral:7b-instruct-v0.2-q4_0", "P1", "failing: 0.0% strict at cap 64"),
    ("mistral:7b-instruct-v0.2-q4_0", "P2", "failing: 0.0% strict at cap 64"),
    ("mistral:7b-instruct-v0.2-q4_0", "P3", "CONTROL: 74.8% strict at cap 64, must stay high"),
    ("phi3:mini", "P1", "failing: 0.3% strict at cap 64"),
    ("phi3:mini", "P3", "failing: 6.6% strict at cap 64"),
    ("phi3:mini", "P2", "CONTROL: 95.0% strict at cap 64, must stay high"),
]

_lock = threading.Lock()


def call(model: str, prompt_text: str, cap: int) -> str:
    body = json.dumps({
        "model": model, "prompt": prompt_text, "stream": False,
        "options": {"temperature": 0.0, "num_predict": cap, "seed": S.SEED},
    }).encode("utf-8")
    req = urllib.request.Request(S.OLLAMA, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r).get("response", "")


def main() -> None:
    n_events = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    print(f"cap probe: {n_events} events, num_predict {S.NUM_PREDICT} -> {cap}")

    events = S.load_events()
    rng = __import__("random").Random(SEED)
    sample = rng.sample(events, min(n_events, len(events)))

    # Baseline compliance at the shipped cap, from the scores the paper already uses, restricted
    # to exactly these events so the two rates are over the same filings.
    accs = {e["accession"] for e in sample}
    base = {}
    with open(os.path.join(S.DATA, "scores.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["trunc"] != TRUNC or r["accession"] not in accs:
                continue
            base.setdefault((r["model"], r["prompt"]), []).append(r.get("raw", ""))

    results = {}
    for model, prompt, why in CELLS:
        t0 = time.time()
        outs = [None] * len(sample)

        def work(i: int) -> None:
            outs[i] = call(model, S.build_prompt(sample[i], prompt, TRUNC), cap)

        with ThreadPoolExecutor(max_workers=S.CONCURRENCY) as ex:
            list(ex.map(work, range(len(sample))))

        new_ok = sum(1 for t in outs if B.parse_raw(prompt, t or "", True)[1])
        b = base.get((model, prompt), [])
        old_ok = sum(1 for t in b if B.parse_raw(prompt, t, True)[1])
        rec = {
            "why": why,
            "n": len(sample),
            "cap_old": S.NUM_PREDICT, "cap_new": cap,
            "strict_old": (old_ok / len(b)) if b else None,
            "strict_new": new_ok / len(sample),
            "mean_chars_new": sum(len(t or "") for t in outs) / len(sample),
            "seconds": round(time.time() - t0, 1),
        }
        results[f"{model}|{prompt}"] = rec
        o = "n/a" if rec["strict_old"] is None else f"{100 * rec['strict_old']:5.1f}%"
        print(f"  {model.split(':')[0]:8s} {prompt}  {o} -> {100 * rec['strict_new']:5.1f}%   "
              f"({rec['seconds']:.0f}s)  [{why}]", flush=True)

    # The control decides whether any of this is interpretable.
    controls = {k: v for k, v in results.items() if v["why"].startswith("CONTROL")}
    ok = all(v["strict_new"] >= 0.60 for v in controls.values())
    results["_control_holds"] = ok
    print()
    if ok:
        print("CONTROL HOLDS: already-compliant cells stayed compliant at the larger cap, so a "
              "change in the failing cells is attributable to the cap.")
    else:
        print("CONTROL FAILED: a cell that complied at cap 64 no longer complies. The larger cap "
              "changed something else as well; DO NOT draw a conclusion from the failing cells.")

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
