# The Grid Is the Alpha

Code, data and stored model outputs for the paper

> Simarjot Khanna. 2026. The Grid Is the Alpha: Measuring Effective Multiplicity and Permutation
> Nulls in LLM Event-Driven Trading. In *7th ACM International Conference on AI in Finance
> (ICAIF '26)*, November 14 to 17, 2026, Milan, Italy. ACM. https://doi.org/10.1145/3858063.3858869

The paper evaluates one LLM event-driven trading pipeline over a pre-registered grid of 7,776
configurations, measures how many independent trials that grid is worth, and benchmarks the best
configuration against a within-session permutation null run over the same grid.

## What is here

| path | contents |
|---|---|
| `paper/` | LaTeX source, the generated macro and table files, the built PDF, and `consistency.py`, the gate that checks every printed claim |
| `code/` | collection, scoring, the grid backtest, the analysis, and `harness_audit.py`, a standalone audit for any grid |
| `data/numbers.json` | every number the paper prints, produced by the analysis scripts |
| `data/` (other files) | the S&P 500 universe and the pinned Wikipedia revision it was parsed from, daily prices, the stored outputs of every model arm, the grid return matrices and the permutation draws |
| `notes/preregistration.md` | the frozen pre-registration |
| `DECISIONS_REDACTED.md` | the decision ledger, including the seven declared pre-registration amendments |
| `REPRODUCE.md` | the full runsheet, from data collection to the PDF |
| `LICENSE` | MIT for the code; terms for the third-party data |

## Requirements

Python 3.10 or later with `numpy`, `scipy` and `matplotlib`; on Windows also `tzdata`. Tested with
Python 3.13, numpy 2.5, scipy 1.18 and matplotlib 3.11.

```
pip install numpy scipy matplotlib tzdata
```

## Check the paper's numbers without re-running anything

These read only the shipped files.

```
python code/trace_macros.py      # every printed number is defined and listed in the ground truth
python code/check_tables.py      # every table cell against data/numbers.json
python paper/consistency.py      # the claim gate
python code/harness_audit.py data/grid_returns.npy   # the effective-trials audit on the shipped grid
```

## Re-run the study

`REPRODUCE.md` lists every stage, its inputs, its outputs and its running time. Collection uses
public sources only (SEC EDGAR, Wikipedia, Yahoo Finance) and needs a contact string for the SEC in
the `SEC_CONTACT` environment variable. Scoring uses a local Ollama server. The 550B arm was scored
through OpenRouter's free tier; its raw outputs ship in `data/scores_or.jsonl`, so re-scoring it is
optional and needs your own key in `OPENROUTER_API_KEY`.
