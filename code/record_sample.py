"""Record how many events the subsample was drawn FROM, measured rather than typed.

`score.py` wrote data/event_sample.json with the drawn accessions and the cap, but not the size of
the pool it drew from. That made the sampling fraction unstateable without hardcoding a number,
and the manuscript consequently described the event set as "every Form 8-K filed by a constituent"
when it is a seed-fixed random subsample (amendment A-3). This recomputes the pool from the same
loader `score.py` uses and folds the count into the same file, so the fraction traces.

Idempotent. Preserves the accession list.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score as S  # noqa: E402

PATH = os.path.join(S.DATA, "event_sample.json")


def main() -> None:
    es = json.load(open(PATH, encoding="utf-8"))
    pool = S.load_events()
    es["available"] = len(pool)
    es["seed"] = S.SEED_SAMPLE if hasattr(S, "SEED_SAMPLE") else 20260731
    drawn = set(es.get("accessions", []))
    have = {e["accession"] for e in pool}
    missing = drawn - have
    if missing:
        raise SystemExit(f"{len(missing)} sampled accessions are not in the pool; the loader and "
                         f"the recorded sample disagree (first: {sorted(missing)[:3]})")
    with open(PATH, "w", encoding="utf-8") as fh:
        json.dump(es, fh, indent=2, sort_keys=True)
    print(f"pool {es['available']} events, drew {es['n']} "
          f"({100 * es['n'] / es['available']:.1f}%), seed {es['seed']}")


if __name__ == "__main__":
    main()
