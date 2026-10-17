"""Feasibility spike for the 550B scale arm. ONE call. Prints the raw answer and the usage.

House rule: no script here POSTs to OpenRouter on its own terms. The key comes from
spend_guard.api_key(), the call is gated by spend_guard.guard() and recorded by spend_guard.record(), so
this arm shows up in the operator's call log like everything else even though the model is free.

Run before committing to 2,978 calls:
    python spike_or.py [model]
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

import backtest as B  # noqa: E402
import score as S  # noqa: E402
try:
    import spend_guard  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - only in the released artifact
    # See the same guard in score_or.py. The spend ledger is not shipped; this file is here to
    # document why the 550B arm runs with its reasoning channel disabled, so it must at least
    # import and read.
    from score_or import spend_guard  # type: ignore[no-redef]

MODEL = sys.argv[1] if len(sys.argv) > 1 else "nvidia/nemotron-3-ultra-550b-a55b:free"
URL = "https://openrouter.ai/api/v1/chat/completions"


def call_or(model: str, prompt_text: str, timeout: int = 180):
    spend_guard.guard(model)
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.0,
        "max_tokens": S.NUM_PREDICT,
        # Reasoning OFF, for the same reason SCALE_THINK is False on the 36B arm: with it on the
        # model spends the whole token budget in the thinking channel and returns an EMPTY answer,
        # which the pipeline would record as a non-compliant response rather than as a broken call.
        # Raising max_tokens instead would break comparability, since the paper states the decode
        # cap is identical for every model.
        "reasoning": {"enabled": False},
    }).encode("utf-8")
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {spend_guard.api_key()}",
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.load(r)
    usage = payload.get("usage", {}) or {}
    spend_guard.record(model, usage, tag="spike-550b", script=__file__)
    choices = payload.get("choices") or []
    if not choices:
        return "", usage, payload
    return (choices[0].get("message", {}) or {}).get("content", ""), usage, payload


def main() -> None:
    events = S.load_events()
    print(f"events loaded: {len(events)}")
    ev = events[0]
    prompt_text = S.build_prompt(ev, "P1", 4000)
    print(f"model        : {MODEL}")
    print(f"event        : {ev['accession']}  {ev['ticker']}  ({len(ev['text'])} chars of text)")
    print(f"prompt chars : {len(prompt_text)}")
    print("-" * 70)
    try:
        raw, usage, payload = call_or(MODEL, prompt_text)
    except urllib.error.HTTPError as e:  # noqa: PERF203 - the body is the diagnostic
        print(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:600]}")
        raise SystemExit(1)
    print(f"RAW ANSWER   : {raw!r}")
    print(f"usage        : {usage}")
    # The whole arm is worthless if the answer does not parse under the SAME parser the grid uses.
    # parse_raw lives in backtest, not score: guarding this with hasattr(S, ...) made the check
    # vacuous and it printed None/None as though the parse had merely failed.
    for strict in (True, False):
        val, ok = B.parse_raw("P1", raw, strict)
        print(f"parse strict={strict}: value={val} compliant={ok}")


if __name__ == "__main__":
    main()
