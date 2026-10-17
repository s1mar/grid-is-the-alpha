# Runsheet: regenerate every number in the paper from scratch

No paid feed and no account. One consumer GPU. Total wall clock is dominated by the LLM scoring
pass; everything downstream runs in minutes.

**One arm needs an API key and one does not ship its runner's dependencies.** The 550B scale arm
(`score_or.py`) was scored against a hosted free-tier endpoint, so re-running it needs your own key
and your own accounting; the operator's spend guard is deliberately not shipped. You do not have to
re-run it: its raw model outputs are in `data/scores_or.jsonl`, and every number derived from them
regenerates from that file alone. The same holds for the decode-cap ablation
(`data/cap_probe.json`, 1,800 local calls).

```
python code/collect.py all            # universe, prices, 8-K metadata, filing text, gauntlet
python code/score.py                  # the 18-pass LLM grid over the event subsample
python code/record_sample.py          # record the pool the subsample was drawn from
python code/probes.py mask            # entity-masked arm
python code/probes.py seed            # stochastic-decoding arm
python code/probes.py scale           # the 36B scale arm
#   the 550B arm and the cap ablation are OPTIONAL; their outputs ship. To re-run them:
#   python code/score_or.py           #   set OPENROUTER_API_KEY to your own OpenRouter key (free endpoint)
#   python code/rescore_cap.py        #   1,800 local Ollama calls
python code/backtest.py 1000          # the 7,776-configuration grid + 1000 permutation replications
python code/analyze.py                # -> data/numbers.json, the single source of truth
python code/probe_analysis.py         # all probe arms incl. 36B and 550B, 300 replications each
python code/robust_sharpe.py          # autocorrelation-robust annualisation (Lo 2002)
python code/power.py 200 10           # synthetic-oracle positive control, 10 draws per accuracy
python code/oracle_dispersion.py      # dispersion under a known signal
python code/membership_audit.py       # index joiners and removals, from the pinned Wikipedia revision
python code/make_macros.py            # -> paper/numbers.tex
python code/make_verify_output.py     # -> paper/verify_output.txt, the reviewer-facing ground truth
python code/figures.py                # -> paper/fig_*.pdf
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
python paper/consistency.py           # the gate
python paper/consistency.py --selftest # prove the gate still catches injected defects
python code/check_tables.py           # every table cell against numbers.json
```

Order matters in one place: `analyze.py` preserves any top-level key it did not itself produce, so
the modules above may be run in any order relative to each other, but `make_macros.py` must run
after all of them.

## Editing the manuscript: the double gate

`paper/consistency.py` checks the document as it stands. That is structurally blind to the most
common defect in a heavily revised paper: a sentence that was CORRECT and became wrong when it was
edited. There is nothing in the finished document to compare it against. So edits get their own
gate, and the document gate refuses to pass until it has been used.

```
python code/editcheck.py snapshot   # BEFORE touching paper/main.tex
#   ... make your edits ...
python code/editcheck.py check      # every changed sentence, with flags to re-read
#   ... read each flagged sentence and fix or accept it ...
python code/editcheck.py snapshot   # re-snapshot once reviewed
python paper/consistency.py         # now passes; it fails while the snapshot is stale
```

`check` compares against the snapshot, pairs each reworded sentence with the one it replaced, and
flags only the changed ones for: a macro dropped by the reword (this is how a number vanishes
silently), a bare numeral introduced, a count word that disagrees with the list it introduces, a
universal quantifier newly asserted, a claim about the permuted scores missing its "within-session"
qualifier, a demonstrative opener whose antecedent also moved, the "and neither is X" inversion, and
a comparison "from" something with no "to".

A flag is a sentence to re-read, not a verdict. `python code/editcheck.py --selftest` re-injects
every one of those defects, all of which reached a built PDF at least once here.

## What each stage needs and produces

| stage | inputs | outputs | wall clock |
|---|---|---|---|
| `collect.py universe` | Wikipedia constituent table, revision 1365256480 | `data/universe.csv` | seconds |
| `membership_audit.py` | `data/sp500_wikipedia_rev1365256480.wikitext.gz`, universe, prices, docs | `data/numbers.json["membership"]` | seconds |
| `collect.py prices` | Yahoo Finance chart endpoint | `data/prices.csv.gz` | ~3 min |
| `collect.py filings` | SEC submissions API, 1 request per company | `data/filings.jsonl` | ~2 min |
| `collect.py docs` | EDGAR primary documents | `data/docs.jsonl` | ~13 min |
| `collect.py exhibits` | EDGAR EX-99 exhibits | `data/exhibits.jsonl` | ~30 min |
| `collect.py check` | all of the above | the gauntlet, exits 1 on any failure | seconds |
| `score.py` | Ollama at `localhost:11434` | `data/scores.jsonl`, `data/event_sample.json` | ~12 h |
| `probes.py` | same | `data/scores_probe.jsonl` | ~1.5 h |
| `backtest.py` | prices, docs, scores | `data/grid.csv`, `grid_returns.npy`, `null_max.npy` | ~5 min |
| `analyze.py` | the above | `data/numbers.json` | ~2 min |

Every network stage is idempotent and resumable: rerunning skips whatever is already on disk, and
a killed run's torn final line is repaired rather than duplicated.

## The rate limits that matter

The SEC asks for a contact string in the `User-Agent` and at most ten requests per second. The
collector sends the contact string and throttles to eight per second through a shared token bucket,
so the concurrency setting cannot violate the limit no matter how it is tuned.

## Determinism

* LLM decoding in the main grid is greedy with a fixed seed. The seed probe is the only
  stochastic arm and it is deliberately so.
* The event subsample, the permutation null, both bootstraps and the figure rendering all take
  fixed seeds, listed at the top of each file.
* `backtest.py` and everything after it are deterministic given `data/`.

## Where a number lives

`data/numbers.json` is the only place a paper quantity may come from. `code/make_macros.py` turns
it into `paper/numbers.tex`, and `paper/consistency.py` fails if the manuscript contains a
quantity outside that macro system, or if `numbers.tex` no longer matches what `make_macros.py`
generates from `numbers.json`.

## Reusing just the audit

`code/harness_audit.py` is standalone. It needs numpy and scipy and the `(n_configs, T)` return
matrix a grid already produced:

```
python code/harness_audit.py data/grid_returns.npy
```
