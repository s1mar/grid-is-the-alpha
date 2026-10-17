"""Collect every input the ICAIF study needs, from public sources only, resumably.

Stages (each is idempotent and skips work already on disk):
    universe  S&P 500 constituent list from Wikipedia  -> data/universe.csv
    prices    daily OHLCV from the Yahoo chart endpoint -> data/prices.csv.gz
    filings   8-K metadata from the SEC submissions API -> data/filings.jsonl
    docs      primary 8-K document text from EDGAR      -> data/docs.jsonl
    check     the correctness gauntlet over all of the above

Nothing here needs an API key, an account, or a paid feed. Run:
    python collect.py all
"""
from __future__ import annotations

import csv
import gzip
import html
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
os.makedirs(DATA, exist_ok=True)

# The SEC's automated-access policy requires a contact string in the User-Agent. It is read from
# the environment rather than hard-coded, because a hard-coded address is an author identity that
# ships inside the released artifact and defeats double-blind review even when the PDF is clean.
# Set SEC_CONTACT to your own email before running.
UA = os.environ.get("SEC_CONTACT_UA") or (
    "ICAIF2026 academic study " + os.environ.get("SEC_CONTACT", "anonymous@example.org"))

# --- frozen sample window (see notes/preregistration.md) ---------------------
FILING_START = "2025-08-01"   # first 8-K filing date considered
FILING_END = "2026-07-24"     # last 8-K filing date considered
PRICE_START = "2025-05-01"    # earlier, so the pre-period liquidity filter has data
PRICE_END = "2026-07-31"      # later, so every event has forward returns

SEC_SLEEP = 0.12              # <= 10 requests/second, the SEC's stated ceiling
DOC_CHARS = 24000             # characters of filing text retained on disk


def _get(url: str, timeout: int = 60, tries: int = 4) -> bytes:
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"}
            )
            resp = urllib.request.urlopen(req, timeout=timeout)
            body = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
            return body
        except Exception as exc:  # noqa: BLE001 - network, retry on anything
            last = exc
            time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"GET failed after {tries} tries: {url}: {last}")


def _epoch(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


# --------------------------------------------------------------------------- universe
UNIVERSE = os.path.join(DATA, "universe.csv")


def stage_universe() -> None:
    if os.path.exists(UNIVERSE):
        print(f"universe: cached {UNIVERSE}")
        return
    # Pinned to revision 1365256480, the live page on the 2026-07-31 fetch date. The original
    # fetch used page=List_of_S%26P_500_companies, which returns whatever the page says today;
    # this revision rebuilds the shipped universe.csv row for row (membership_audit.py checks it).
    url = ("https://en.wikipedia.org/w/api.php?action=parse"
           "&oldid=1365256480&prop=wikitext&format=json")
    wikitext = json.loads(_get(url))["parse"]["wikitext"]["*"]
    rows = parse_constituents(wikitext)
    if len(rows) < 450:
        raise RuntimeError(f"universe parse looks wrong: only {len(rows)} rows")
    with open(UNIVERSE, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["ticker", "name", "sector", "date_added", "cik"])
        w.writeheader()
        w.writerows(rows)
    print(f"universe: {len(rows)} constituents -> {UNIVERSE}")


# A constituents-table row opens with a ticker template. Only these templates are parsed: the
# pinned revision has one row in another template ({{BZX link|CBOE}}), which this rule drops, so
# Cboe Global Markets is not in the universe. membership_audit.py counts and names such rows.
SYMBOL_ROW = re.compile(r"^\|\{\{(?:Nyse|Nasdaq|Bats|NYSE|NASDAQ)Symbol\|([A-Za-z0-9.\-]+)\}\}")


def parse_constituents(wikitext: str) -> list[dict]:
    """The constituents table of the Wikipedia page, as universe.csv rows. Split out of
    stage_universe so membership_audit.py rebuilds the list with this exact rule."""
    start = wikitext.index('id="constituents"')
    end = wikitext.index("|}", start)
    block = wikitext[start:end]

    rows, current = [], None
    for line in block.split("\n"):
        line = line.strip()
        m = SYMBOL_ROW.match(line)
        if m:
            current = [m.group(1)]
            continue
        if current is not None and line.startswith("|") and "||" in line:
            cells = [c.strip() for c in line.lstrip("|").split("||")]
            # security || sector || sub-industry || hq || date added || cik || founded
            if len(cells) >= 6:
                name = re.sub(r"\[\[([^\]|]+)\|?[^\]]*\]\]", r"\1", cells[0]).strip()
                rows.append({
                    "ticker": current[0],
                    "name": name,
                    "sector": cells[1],
                    "date_added": cells[4],
                    "cik": cells[5].zfill(10),
                })
            current = None
    return rows


def load_universe() -> list[dict]:
    with open(UNIVERSE, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------- prices
PRICES = os.path.join(DATA, "prices.csv.gz")


def _yahoo_symbol(ticker: str) -> str:
    return ticker.replace(".", "-")


def stage_prices() -> None:
    if os.path.exists(PRICES):
        print(f"prices: cached {PRICES}")
        return
    tickers = [r["ticker"] for r in load_universe()] + ["SPY"]
    p1, p2 = _epoch(PRICE_START), _epoch(PRICE_END)
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["ticker", "date", "open", "high", "low", "close", "adjclose", "volume"])
    ok = fail = 0
    for i, t in enumerate(tickers):
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{_yahoo_symbol(t)}"
               f"?period1={p1}&period2={p2}&interval=1d&events=div%7Csplit")
        try:
            body = json.loads(_get(url))
            res = body["chart"]["result"][0]
            ts = res["timestamp"]
            q = res["indicators"]["quote"][0]
            adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
            for k, epoch in enumerate(ts):
                day = datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d")
                w.writerow([t, day, q["open"][k], q["high"][k], q["low"][k],
                            q["close"][k], adj[k], q["volume"][k]])
            ok += 1
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"  price fail {t}: {type(exc).__name__}")
        if i % 50 == 0:
            print(f"  prices {i}/{len(tickers)}")
        time.sleep(0.15)
    with gzip.open(PRICES, "wt", encoding="utf-8", newline="") as fh:
        fh.write(out.getvalue())
    print(f"prices: {ok} ok, {fail} failed -> {PRICES}")


# --------------------------------------------------------------------------- filings
FILINGS = os.path.join(DATA, "filings.jsonl")


def _dedup_jsonl(path: str, key: str) -> None:
    """Rewrite a jsonl append-log keeping one record per key, dropping any torn final line.

    A killed run can leave a half-written line, and the retry then appends the same record again.
    Without this the uniqueness assertion in `check` fires on an artefact of the interruption
    rather than on a real defect.
    """
    if not os.path.exists(path):
        return
    keep, order = {}, []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001 - torn line
                continue
            if rec[key] not in keep:
                order.append(rec[key])
            keep[rec[key]] = line
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for k in order:
            fh.write(keep[k] + "\n")
    os.replace(tmp, path)


def _seen_keys(path: str, key: str) -> set:
    _dedup_jsonl(path, key)
    seen = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        seen.add(json.loads(line)[key])
                    except Exception:  # noqa: BLE001 - a torn last line
                        pass
    return seen


class _Throttle:
    """Token bucket shared by every worker thread. The SEC's stated ceiling is 10 requests/s."""

    def __init__(self, per_second: float = 8.0) -> None:
        import threading
        self._gap = 1.0 / per_second
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            due = max(now, self._next)
            self._next = due + self._gap
        delay = due - time.monotonic()
        if delay > 0:
            time.sleep(delay)


THROTTLE = _Throttle()


def _get_throttled(url: str) -> bytes:
    THROTTLE.wait()
    return _get(url)


def _run_pool(jobs: list, fn, out_path: str, label: str, workers: int = 8) -> None:
    """Map `fn` over `jobs` concurrently, appending each returned record as it lands."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    lock = threading.Lock()
    state = {"n": 0, "t0": time.time()}
    with open(out_path, "a", encoding="utf-8") as fh:
        def wrapped(job):
            rec = fn(job)
            with lock:
                fh.write(json.dumps(rec) + "\n")
                state["n"] += 1
                if state["n"] % 250 == 0:
                    fh.flush()
                    el = time.time() - state["t0"]
                    print(f"  {label} {state['n']}/{len(jobs)} "
                          f"({state['n']/el:.1f}/s, {el/60:.1f} min)", flush=True)
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(wrapped, jobs))


def stage_filings() -> None:
    universe = load_universe()
    done = _seen_keys(FILINGS, "cik")
    todo = [r for r in universe if r["cik"] not in done]
    print(f"filings: {len(done)} companies cached, {len(todo)} to fetch")
    with open(FILINGS, "a", encoding="utf-8") as fh:
        for i, row in enumerate(todo):
            cik = row["cik"]
            try:
                sub = json.loads(_get(f"https://data.sec.gov/submissions/CIK{cik}.json"))
            except Exception as exc:  # noqa: BLE001
                print(f"  filings fail {row['ticker']}: {type(exc).__name__}")
                time.sleep(SEC_SLEEP)
                continue
            recent = sub["filings"]["recent"]
            events = []
            for k, form in enumerate(recent["form"]):
                if form != "8-K":
                    continue
                fdate = recent["filingDate"][k]
                if not (FILING_START <= fdate <= FILING_END):
                    continue
                events.append({
                    "accession": recent["accessionNumber"][k],
                    "filing_date": fdate,
                    "acceptance": recent["acceptanceDateTime"][k],
                    "items": recent["items"][k],
                    "primary_doc": recent["primaryDocument"][k],
                    "report_date": recent["reportDate"][k],
                })
            fh.write(json.dumps({
                "cik": cik, "ticker": row["ticker"], "sector": row["sector"],
                "oldest_recent": recent["filingDate"][-1] if recent["filingDate"] else None,
                "n_recent": len(recent["form"]),
                "events": events,
            }) + "\n")
            fh.flush()
            if i % 50 == 0:
                print(f"  filings {i}/{len(todo)}")
            time.sleep(SEC_SLEEP)
    print("filings: done")


def load_filings() -> list[dict]:
    rows = []
    with open(FILINGS, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------- docs
DOCS = os.path.join(DATA, "docs.jsonl")

_TAG = re.compile(r"(?s)<[^>]+>")
_DROP = re.compile(r"(?is)<(script|style|head)[^>]*>.*?</\1>")
_WS = re.compile(r"[ \s]+")


def strip_html(raw: bytes) -> str:
    text = raw.decode("utf-8", "ignore")
    text = _DROP.sub(" ", text)
    text = _TAG.sub(" ", text)
    text = html.unescape(text)
    return _WS.sub(" ", text).strip()


def stage_docs() -> None:
    wanted = []
    for company in load_filings():
        for ev in company["events"]:
            wanted.append((company["cik"], company["ticker"], ev))
    done = _seen_keys(DOCS, "accession")
    todo = [x for x in wanted if x[2]["accession"] not in done]
    print(f"docs: {len(wanted)} filings, {len(done)} cached, {len(todo)} to fetch")

    def one(job):
        cik, ticker, ev = job
        acc = ev["accession"].replace("-", "")
        url = (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{ev['primary_doc']}")
        try:
            text = strip_html(_get_throttled(url))
        except Exception as exc:  # noqa: BLE001
            text = ""
            print(f"  doc fail {ticker} {ev['accession']}: {type(exc).__name__}")
        return {
            "accession": ev["accession"], "ticker": ticker, "cik": cik,
            "filing_date": ev["filing_date"], "acceptance": ev["acceptance"],
            "items": ev["items"], "url": url,
            "n_chars_full": len(text), "text": text[:DOC_CHARS],
        }

    _run_pool(todo, one, DOCS, "docs")
    print("docs: done")


# --------------------------------------------------------------------------- exhibits
EXHIBITS = os.path.join(DATA, "exhibits.jsonl")

# The header index escapes its own SGML, so the document table arrives as &lt;TYPE&gt;.
_EXDOC = re.compile(r"&lt;TYPE&gt;(EX-99[^\n<]*)\n(?:.*?)&lt;FILENAME&gt;([^\n<]+)\n", re.S)


def stage_exhibits() -> None:
    """Fetch the EX-99 exhibits.

    An earnings 8-K's primary document is usually a two-page cover that says the substance is
    attached as Exhibit 99.1. Reading only the primary document would hand the model a stub and
    then blame the model for having no view.
    """
    wanted = []
    for company in load_filings():
        for ev in company["events"]:
            wanted.append((company["cik"], company["ticker"], ev))
    done = _seen_keys(EXHIBITS, "accession")
    todo = [x for x in wanted if x[2]["accession"] not in done]
    print(f"exhibits: {len(wanted)} filings, {len(done)} cached, {len(todo)} to fetch")

    def one(job):
        cik, ticker, ev = job
        acc_raw, acc = ev["accession"], ev["accession"].replace("-", "")
        base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}"
        names, texts = [], []
        try:
            head = _get_throttled(f"{base}/{acc_raw}-index-headers.html").decode("utf-8", "ignore")
            for _type, fname in _EXDOC.findall(head):
                if fname.lower().endswith((".htm", ".html", ".txt")):
                    names.append(fname)
            for fname in names[:3]:
                texts.append(strip_html(_get_throttled(f"{base}/{fname}")))
        except Exception as exc:  # noqa: BLE001
            print(f"  exhibit fail {ticker} {acc_raw}: {type(exc).__name__}")
        blob = "\n\n".join(t for t in texts if t)
        return {
            "accession": acc_raw, "ticker": ticker,
            "exhibit_files": names[:3], "n_chars_full": len(blob),
            "text": blob[:DOC_CHARS],
        }

    _run_pool(todo, one, EXHIBITS, "exhibits")
    print("exhibits: done")


# --------------------------------------------------------------------------- check
def stage_check() -> None:
    """The correctness gauntlet. Every assertion here failed loudly at least once."""
    import collections

    fails = []

    def need(cond: bool, msg: str) -> None:
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(msg)

    print("universe")
    uni = load_universe()
    need(450 <= len(uni) <= 520, f"constituent count {len(uni)} in [450,520]")
    need(len({r["ticker"] for r in uni}) == len(uni), "tickers unique")
    need(all(len(r["cik"]) == 10 and r["cik"].isdigit() for r in uni), "CIKs are 10-digit")

    print("prices")
    with gzip.open(PRICES, "rt", encoding="utf-8") as fh:
        px = list(csv.DictReader(fh))
    tick_days = collections.Counter(r["ticker"] for r in px)
    need("SPY" in tick_days, "SPY present")
    spy_days = tick_days["SPY"]
    need(280 <= spy_days <= 330, f"SPY has {spy_days} trading days for a ~15-month window")
    covered = sum(1 for t in {r["ticker"] for r in uni} if tick_days.get(t, 0) >= 0.95 * spy_days)
    need(covered >= 0.90 * len(uni), f"{covered}/{len(uni)} tickers have >=95% of SPY's days")
    need(all(r["date"] <= PRICE_END for r in px), "no price beyond PRICE_END")

    print("filings")
    fil = load_filings()
    need(len(fil) >= 0.95 * len(uni), f"{len(fil)}/{len(uni)} companies have a filings record")
    evs_all = [e for c in fil for e in c["events"]]
    # Three CIKs in the S&P 500 carry two ticker classes each (GOOG/GOOGL, FOX/FOXA, NWS/NWSA),
    # so their filings arrive once per class. One filing is one event; the analysis keeps the
    # alphabetically first ticker. Assert the duplication has exactly that explanation.
    owners = collections.defaultdict(set)
    for c in fil:
        for e in c["events"]:
            owners[e["accession"]].add((c["cik"], c["ticker"]))
    multi_cik = {c["cik"] for c in fil
                 if sum(1 for o in fil if o["cik"] == c["cik"]) > 1}
    multi = sorted(c["ticker"] for c in fil if c["cik"] in multi_cik)
    dup_acc = {a for a, v in owners.items() if len(v) > 1}
    unexplained = [a for a in dup_acc
                   if len({cik for cik, _ in owners[a]}) != 1
                   or next(iter(owners[a]))[0] not in multi_cik]
    need(bool(multi_cik), "the dual-class check actually found the multi-ticker CIKs it tests")
    need(not unexplained,
         f"every duplicated accession is one filing by one dual-class CIK; "
         f"unexplained: {unexplained[:5]}")
    evs = list({e["accession"]: e for e in evs_all}.values())
    print(f"  info  {len(evs_all)} filing rows, {len(evs)} unique accessions, "
          f"{len(dup_acc)} duplicated; dual-class tickers: {multi}")
    need(all(FILING_START <= e["filing_date"] <= FILING_END for e in evs), "filing dates in window")
    # The submissions API caps its "recent" block at 1000 filings. A company whose oldest recent
    # filing postdates the window start has EITHER hit that cap (events silently missing) OR is a
    # new registrant (nothing missing). Only the first is a defect, so test the cap, not the date.
    truncated = [c["ticker"] for c in fil
                 if c["n_recent"] >= 1000 and c["oldest_recent"]
                 and c["oldest_recent"] > FILING_START]
    need(not truncated, f"no company hit the 1000-filing cap inside the window "
                        f"(would silently drop events); offenders: {truncated[:8]}")
    newcos = [c["ticker"] for c in fil
              if c["n_recent"] < 1000 and c["oldest_recent"]
              and c["oldest_recent"] > FILING_START]
    if newcos:
        print(f"  info  {len(newcos)} new registrant(s) with no pre-window filings: {newcos[:8]}")

    print("docs")
    docs = list({json.loads(l)["accession"]: json.loads(l)
                 for l in open(DOCS, encoding="utf-8") if l.strip()}.values())
    need(len(docs) == len(evs), f"docs {len(docs)} == unique filings {len(evs)}")
    empty = sum(1 for d in docs if len(d["text"]) < 200)
    need(empty <= 0.03 * max(len(docs), 1), f"{empty}/{len(docs)} docs shorter than 200 chars")
    lens = sorted(len(d["text"]) for d in docs)
    if lens:
        print(f"  info  doc text length p10/p50/p90 = "
              f"{lens[len(lens)//10]}/{lens[len(lens)//2]}/{lens[9*len(lens)//10]}")
        print(f"  info  {len(evs)} events, {len({e['filing_date'] for e in evs})} distinct dates")

    print("exhibits")
    ex = list({json.loads(l)["accession"]: json.loads(l)
               for l in open(EXHIBITS, encoding="utf-8") if l.strip()}.values())
    need(len(ex) == len(evs), f"exhibit records {len(ex)} == unique filings {len(evs)}")
    combined = {d["accession"]: len(d["text"]) for d in docs}
    for d in ex:
        combined[d["accession"]] = combined.get(d["accession"], 0) + len(d["text"])
    short = sum(1 for v in combined.values() if v < 600)
    need(short <= 0.10 * max(len(combined), 1),
         f"{short}/{len(combined)} events have under 600 chars of combined text")
    cl = sorted(combined.values())
    print(f"  info  combined text p10/p50/p90 = "
          f"{cl[len(cl)//10]}/{cl[len(cl)//2]}/{cl[9*len(cl)//10]}")
    with_ex = sum(1 for d in ex if d["text"])
    print(f"  info  {with_ex}/{len(ex)} events carry at least one EX-99 exhibit")

    print()
    if fails:
        print(f"GAUNTLET FAILED: {len(fails)} check(s)")
        for f in fails:
            print("  - " + f)
        sys.exit(1)
    print("GAUNTLET PASSED")


STAGES = {
    "universe": stage_universe,
    "prices": stage_prices,
    "filings": stage_filings,
    "docs": stage_docs,
    "exhibits": stage_exhibits,
    "check": stage_check,
}

if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    order = (["universe", "prices", "filings", "docs", "exhibits", "check"]
             if args == ["all"] else args)
    for name in order:
        print(f"\n=== {name} ===")
        STAGES[name]()
