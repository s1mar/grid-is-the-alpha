"""Index-membership audit of the universe list. Post hoc, added at camera-ready. It revises decision
D-011 in the decision ledger; it is not a pre-registration amendment, because no analysis changes.

The universe is the Wikipedia "List of S&P 500 companies" as `collect.py` fetched it on 2026-07-31,
one day after the last session. A list taken at the end of the window contains names that joined
the index during the window and omits names that left it. This script measures both, from a pinned
copy of the page, and changes nothing in the study: no result is recomputed and no event is excluded.

What it establishes, and how:
  * Which page was fetched. collect.py did not record a revision id. Revision 1365256480
    (dated 2026-07-21) was the live revision from that date until the next edit on
    2026-08-03 (MediaWiki revision history, read 2026-10-17), and parsing it with
    collect.parse_constituents rebuilds data/universe.csv row for row. The second fact is asserted
    on every run; the first is recorded here and in the decision ledger.
  * Joiners: listed names whose date_added falls inside [first session, last session].
  * Pre-addition filings: tradable events (backtest.load_events, the set every result uses) of a
    joiner whose filing date is strictly before that firm's date_added.
  * Removals: rows of the revision's change table dated inside the window that name a removed
    ticker. Their filings are not in the pool, which holds filings of the listed names only, so
    they are counted as names. Their filings were never collected and are not counted.

    python code/membership_audit.py           # read the pinned copy, write numbers.json["membership"]
    python code/membership_audit.py --fetch   # also re-fetch the revision and check it against the pin
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import backtest as B  # noqa: E402
import collect as C  # noqa: E402

REVISION = 1365256480
REVISION_DATE = "2026-07-21"
# The date collect.py fetched the page: the modification date of data/universe.csv (2026-07-31),
# corroborated by a byte-identical copy in the 2026-08-01 backup of the working tree.
FETCHED = "2026-07-31"
PIN = os.path.join(B.DATA, f"sp500_wikipedia_rev{REVISION}.wikitext.gz")
PIN_SHA256 = "f0b067837459eb10e82585d21d6c95432ff1411aad5960d406c9699d9d707d1a"  # of the wikitext
API = ("https://en.wikipedia.org/w/api.php?action=parse&oldid={rev}&prop=wikitext&format=json")


def read_pin() -> str:
    raw = gzip.open(PIN, "rb").read()
    got = hashlib.sha256(raw).hexdigest()
    if got != PIN_SHA256:
        raise SystemExit(f"{PIN} has sha256 {got}, expected {PIN_SHA256}")
    return raw.decode("utf-8")


def removals(wikitext: str, lo: str, hi: str) -> list[dict]:
    """Change-table rows dated in [lo, hi] that name a removed ticker.

    Rows are split on the table's row separator and read by column position (date, added ticker,
    added name, removed ticker, removed name, reason), because a row with only a removal leaves the
    added cells empty, and a symbol search cannot tell which side of the row a ticker sits on.
    """
    i = wikitext.index('id="changes"')
    block = wikitext[i: wikitext.index("|}", i)]
    out = []
    for part in re.split(r"\n\|-\s*\n", block)[3:]:          # the first three are the header rows
        part = re.sub(r"<ref[^>]*/>", "", part)
        part = re.sub(r"<ref[^>]*>.*?</ref>", "", part, flags=re.S)
        part = re.sub(r"<!--.*?-->", "", part, flags=re.S).strip()
        if not part.startswith("|"):
            continue
        cells = [c.strip() for c in part[1:].replace("\n|", "||").split("||")]
        if len(cells) < 6:
            raise SystemExit(f"change-table row with {len(cells)} cells: {cells}")
        try:
            day = datetime.strptime(cells[0], "%B %d, %Y").strftime("%Y-%m-%d")
        except ValueError as exc:
            raise SystemExit(f"change-table row with an unreadable date: {cells[0]!r}") from exc
        if lo <= day <= hi and cells[3]:
            out.append({"date": day, "removed": cells[3], "added": cells[1]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="re-fetch the revision and compare")
    args = ap.parse_args()

    wikitext = read_pin()
    if args.fetch:
        live = json.loads(C._get(API.format(rev=REVISION)))["parse"]["wikitext"]["*"]
        same = hashlib.sha256(live.encode("utf-8")).hexdigest() == PIN_SHA256
        print(f"re-fetched revision {REVISION}: {'identical to' if same else 'DIFFERS from'} the pin")
        if not same:
            raise SystemExit(1)

    with open(C.UNIVERSE, encoding="utf-8") as fh:
        universe = list(csv.DictReader(fh))
    rebuilt = C.parse_constituents(wikitext)
    if rebuilt != universe:
        raise SystemExit(f"revision {REVISION} does not rebuild data/universe.csv "
                         f"({len(rebuilt)} parsed rows against {len(universe)})")
    # The rebuild check cannot see a row the parser drops, because both sides use the same rule.
    # Count the table's rows independently: each opens with a ticker template.
    s = wikitext.index('id="constituents"')
    openers = [ln.strip() for ln in wikitext[s: wikitext.index("|}", s)].split("\n")
               if re.match(r"^\|\{\{", ln.strip())]
    unparsed = [re.search(r"\|([A-Za-z0-9.\-]+)\}\}", ln).group(1) for ln in openers
                if not C.SYMBOL_ROW.match(ln)]
    if len(openers) != len(universe) + len(unparsed):
        raise SystemExit(f"{len(openers)} table rows, {len(universe)} parsed, {len(unparsed)} unparsed")

    path = os.path.join(B.DATA, "numbers.json")
    n = json.load(open(path, encoding="utf-8"))
    tickers, sessions, t_idx, s_idx, _op, _cl, _spy = B.load_prices()
    events = B.load_events(sessions, t_idx, s_idx)
    # The window the paper prints (\FirstSession to \LastSession). NOT sessions[0]: the price file
    # starts three months earlier for warm-up, and using it counted 25 joiners where the window has 21.
    lo, hi = n["first_session"], n["last_session"]
    added = {u["ticker"]: u["date_added"] for u in universe}
    joiners = {t: d for t, d in added.items() if lo <= d <= hi}
    pre = [e for e in events if e["ticker"] in joiners and e["filing_date"] < joiners[e["ticker"]]]
    gone = removals(wikitext, lo, hi)

    out = {
        "revision": REVISION,
        "revision_date": REVISION_DATE,
        "fetched": FETCHED,
        "wikitext_sha256": PIN_SHA256,
        "window": [lo, hi],
        "n_listed": len(openers),
        "unparsed_tickers": unparsed,
        "n_constituents": len(universe),
        "n_joiners": len(joiners),
        "n_events": len(events),
        "n_pre_addition_events": len(pre),
        "pre_addition_pct": 100.0 * len(pre) / len(events),
        "n_pre_addition_firms": len({e["ticker"] for e in pre}),
        "n_removals": len(gone),
        "removed_tickers": sorted(g["removed"] for g in gone),
    }
    if n["n_events"] != out["n_events"]:
        raise SystemExit(f"load_events gave {out['n_events']} events, numbers.json says "
                         f"{n['n_events']}: the audit is not over the set the results use")
    n["membership"] = out
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(n, fh, indent=2, sort_keys=True)
    for k, v in out.items():
        print(f"  {k:24s} {v}")
    print('wrote data/numbers.json["membership"]')


if __name__ == "__main__":
    main()
