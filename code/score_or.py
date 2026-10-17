"""The 550B scale arm, run over OpenRouter's free tier.

Why this exists. The most predictable reviewer objection to this paper is that its three scoring
models are all under 8B parameters, so a null result shows only that small models cannot read an
8-K. The 36B arm answers part of that. This arm extends the ladder to 550B at zero cost.

Design constraints, each of which is load-bearing:
  * IDENTICAL events. The arm scores exactly the accessions already present in data/scores.jsonl,
    read from that file rather than re-derived from the subsample rule, so the comparison cannot
    drift if the sampling code ever changes.
  * IDENTICAL prompt and truncation to the 36B arm (P1 @ 4000, temperature 0), so the two scale
    arms differ in the model and nothing else.
  * IDENTICAL decode cap (score.NUM_PREDICT). Reasoning is disabled rather than the cap raised,
    because the paper states the cap is the same for every model; see the spike note in
    spike_or.py for what happens when it is left on.
  * Resumable on (accession, probe). An interrupted run loses nothing.
  * Every call goes through spend_guard for the key, the guard and the ledger, free or not.

    python score_or.py --limit 20     # throughput probe: measures rate and 429s, commits nothing
    python score_or.py                # everything still missing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

import score as S  # noqa: E402

# `spend_guard` is the spend ledger and approval guard, which lives outside this package and is NOT
# shipped: it records the operator's approvals. Inside the released artifact the import fails, and
# it used to fail at module level, so a reviewer opening this file could not even import it to read
# how the arm was run. The arm's raw outputs ship in data/scores_or.jsonl, so re-running is
# optional. Without spend_guard, the stub below uses YOUR OpenRouter key from the OPENROUTER_API_KEY
# environment variable and refuses any model that is not a ":free" endpoint, so it cannot spend.
try:
    import spend_guard  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - only in the released artifact
    class _NoSpend:
        """Present so the module imports and reads; calls only free endpoints, with your key."""

        @staticmethod
        def guard(model, *_a, **_k):
            if not str(model).endswith(":free"):
                raise SystemExit(f"refusing {model}: the released runner calls free endpoints only")
            if not os.environ.get("OPENROUTER_API_KEY"):
                raise SystemExit(
                    "set OPENROUTER_API_KEY to your own OpenRouter key to re-score this arm. Its raw "
                    "outputs already ship in data/scores_or.jsonl, so re-scoring is optional, and a "
                    "new run may differ: the serving provider and its precision were not recorded.")

        @staticmethod
        def api_key():
            return os.environ["OPENROUTER_API_KEY"]

        @staticmethod
        def record(*_a, **_k):
            return 0.0

    spend_guard = _NoSpend()  # type: ignore[assignment]

MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
PROBE = "scale550"
PROMPT, TRUNC = "P1", 4000
OUT = os.path.join(S.DATA, "scores_or.jsonl")
URL = "https://openrouter.ai/api/v1/chat/completions"

# Measured: 4 workers gives 40/min with zero 429s; 8 gives 82/min but rate-limits about 10% of
# calls into retry exhaustion. Throughput is not worth a hole in the arm.
CONCURRENCY = 4
MAX_RETRY = 5

_lock = threading.Lock()
_stats = {"ok": 0, "empty": 0, "failed": 0, "http": 0, "retry": 0, "rate_limited": 0}


def target_accessions() -> list[str]:
    """The accessions the MAIN grid actually scored, in file order, deduplicated.

    Read from scores.jsonl rather than recomputed from the subsample rule: the arm has to sit on
    the same events as the grid it is being compared against, and a rule can be edited.
    """
    seen, out = set(), []
    with open(os.path.join(S.DATA, "scores.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            a = json.loads(line)["accession"]
            if a not in seen:
                seen.add(a)
                out.append(a)
    return out


def done_keys() -> set:
    """Accessions with a REAL answer from the model.

    A call that exhausted its retries is not done. It was previously written with raw="" and no
    usage, which is byte-identical to a model that replied with nothing, so the parser would have
    scored a rate-limited call as a non-compliant response. At concurrency 8 that was 10% of the
    arm. Failed rows carry failed=true, are excluded here so a resume retries them, and are
    counted separately at the end.
    """
    keys = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("failed"):
                    continue
                keys.add((r["accession"], r.get("probe")))
    return keys


def call_or(prompt_text: str, timeout: int = 180) -> tuple[str, dict, bool]:
    """One call, with backoff on 429/5xx.

    Returns (raw, usage, failed). `failed` distinguishes "the harness never got an answer" from
    "the model answered with nothing", which the caller must keep distinct: only the second is a
    compliance fact about the model.
    """
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.0,
        "max_tokens": S.NUM_PREDICT,
        "reasoning": {"enabled": False},
    }).encode("utf-8")
    for attempt in range(MAX_RETRY):
        spend_guard.guard(MODEL)
        req = urllib.request.Request(URL, data=body, headers={
            "Authorization": f"Bearer {spend_guard.api_key()}",
            "Content-Type": "application/json",
        })
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.load(r)
        except urllib.error.HTTPError as e:
            with _lock:
                _stats["http"] += 1
                if e.code == 429:
                    _stats["rate_limited"] += 1
            if e.code in (429, 500, 502, 503, 504) and attempt < MAX_RETRY - 1:
                with _lock:
                    _stats["retry"] += 1
                time.sleep(min(60, 3 * 2 ** attempt))
                continue
            return "", {}, True
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt < MAX_RETRY - 1:
                with _lock:
                    _stats["retry"] += 1
                time.sleep(min(60, 3 * 2 ** attempt))
                continue
            return "", {}, True
        usage = payload.get("usage", {}) or {}
        spend_guard.record(MODEL, usage, tag=PROBE, script=__file__)
        choices = payload.get("choices") or []
        # A 200 with no choices is an upstream failure, not a model that declined to answer. It
        # has no usage block either. Recording it as an empty answer would put a harness fault
        # into the arm's compliance rate, which is the number this arm exists to report. Only a
        # choice that is PRESENT and empty is a compliance fact about the model.
        if not choices:
            if attempt < MAX_RETRY - 1:
                with _lock:
                    _stats["retry"] += 1
                time.sleep(min(60, 3 * 2 ** attempt))
                continue
            return "", usage, True
        raw = (choices[0].get("message", {}) or {}).get("content", "")
        return raw, usage, False
    return "", {}, True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="score at most N events; use for a throughput probe")
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    a = ap.parse_args()

    events = {e["accession"]: e for e in S.load_events()}
    want = target_accessions()
    missing_text = [x for x in want if x not in events]
    if missing_text:
        raise SystemExit(f"{len(missing_text)} scored accessions have no event text; refusing to "
                         f"run a partial arm (first: {missing_text[:3]})")
    have = done_keys()
    todo = [x for x in want if (x, PROBE) not in have]
    if a.limit:
        todo = todo[:a.limit]
    print(f"grid events {len(want)}, already done {len(want) - len([x for x in want if (x, PROBE) not in have])}, "
          f"to run now {len(todo)}, concurrency {a.concurrency}")
    if not todo:
        return

    t0 = time.time()
    fh = open(OUT, "a", encoding="utf-8")

    def work(acc: str) -> None:
        ev = events[acc]
        raw, usage, failed = call_or(S.build_prompt(ev, PROMPT, TRUNC))
        with _lock:
            if failed:
                _stats["failed"] += 1
            elif raw.strip():
                _stats["ok"] += 1
            else:
                _stats["empty"] += 1
            fh.write(json.dumps({"accession": acc, "model": MODEL, "prompt": PROMPT,
                                 "trunc": TRUNC, "probe": PROBE, "raw": raw,
                                 "usage": usage, "failed": failed}) + "\n")
            fh.flush()
            n = _stats["ok"] + _stats["empty"] + _stats["failed"]
            if n % 25 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  {n}/{len(todo)}  {n / el * 60:.0f}/min  ok={_stats['ok']} "
                      f"empty={_stats['empty']} failed={_stats['failed']} "
                      f"429={_stats['rate_limited']} "
                      f"retry={_stats['retry']}", flush=True)

    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        list(ex.map(work, todo))
    fh.close()

    el = time.time() - t0
    done = _stats["ok"] + _stats["empty"] + _stats["failed"]
    print(f"\ndone {done} in {el / 60:.1f} min ({done / el * 60:.0f}/min)")
    print(f"  ok {_stats['ok']}, empty {_stats['empty']}, FAILED {_stats['failed']}, "
          f"429s {_stats['rate_limited']}, retries {_stats['retry']}")
    if _stats["empty"]:
        print("  WARNING: empty answers are recorded as non-compliant by the parser. If this "
              "count is large the arm is measuring a broken call, not a model refusal.")
    remaining = len(want) - len(done_keys())
    if remaining > 0:
        print(f"  {remaining} events remain; at this rate about "
              f"{remaining / max(done / el, 1e-9) / 60:.0f} min more.")


if __name__ == "__main__":
    main()
