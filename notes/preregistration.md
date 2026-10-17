# Pre-registration: nuisance degrees of freedom in an LLM event-driven trading pipeline

**Frozen 2026-07-31, before any portfolio return was computed.** The paper's subject is undisclosed
researcher freedom, so a grid chosen after seeing the answers would be an instance of the thing the
paper criticises. `paper/consistency.py` gates the shipped grid against this file; any deviation
must appear in `DECISIONS.md` as an amendment with a reason, and the amendment count must be
declared in the paper.

Status at freeze time: universe, 8-K metadata and prices are collected; **no score has been
computed, no return has been computed, and no Sharpe ratio exists on disk.**

---

## 1. Data

| item | value |
|---|---|
| universe | S&P 500 constituents listed on Wikipedia's constituent table, fetched 2026-07-31: **502 tickers** |
| events | every `8-K` filed by a universe member with `filingDate` in [2025-08-01, 2026-07-24] |
| event count at freeze | **6,350** across 243 distinct filing dates |
| text | the filing's primary document plus its EX-99 exhibits, HTML-stripped |
| prices | Yahoo Finance daily OHLCV + adjusted close, 2025-05-01 to 2026-07-31 |
| benchmark | SPY |

The sample window starts **after the pretraining cutoff of all three scoring models**
(`qwen2.5:7b-instruct`, `mistral:7b-instruct-v0.2`, `phi3:mini`; all released in or before 2024),
so results cannot be produced by memorised outcomes.

## 2. The pipeline

For each event `e` (ticker `i`, acceptance timestamp `T_e`, text `X_e`):

1. **Score.** An LLM reads a truncation of `X_e` under a prompt and emits a raw string; a parser
   maps it to `s_e` in [0,1].
2. **Signal.** `s_e` is transformed into a signed signal `g_e`.
3. **Position.** A direction, a size, an entry session and a holding period turn `g_e` into a
   position.
4. **Portfolio.** Positions are pooled into one daily return series; annualised Sharpe is computed.

### Tradeability rule (FIXED, not a knob)

An event is tradable only in the first regular session that opens **strictly after** `T_e`
(acceptance timestamp, converted from UTC to America/New_York). Filings accepted before 09:30 ET on
a trading day are tradable that same session; all others at the next session. No configuration may
use information dated at or after its own entry price.

### Transaction costs (FIXED, not a knob)

5 basis points per side, charged on the traded notional at entry and at exit. Applied identically
to every configuration. The zero-cost grid is reported once as a robustness line.

## 3. The frozen factor grid

**LLM-side factors (each level costs a separate inference pass): 3 x 3 x 2 = 18 passes/event.**

| factor | levels | note |
|---|---|---|
| `model` | `qwen2.5:7b-instruct`, `mistral:7b-instruct-v0.2-q4_0`, `phi3:mini` | 3 |
| `prompt` | `P1` analyst probability 0-100; `P2` three-way label with confidence; `P3` JSON with an explicit "no view" option | 3 |
| `trunc` | 1200, 4000 characters of filing text | 2 |

**Post-LLM factors (free; they re-use the same scores): 2 x 3 x 3 x 3 x 2 x 2 x 2 = 432.**

| factor | levels | note |
|---|---|---|
| `parse` | `strict` (answer must be a bare number; otherwise the event is dropped), `lenient` (first number anywhere; otherwise neutral) | 2 |
| `signal` | `level` (`s - 0.5`), `xrank` (cross-sectional rank within the filing date, centred), `demean` (`s` minus that ticker's running mean) | 3 |
| `thresh` | 0.00, 0.10, 0.20 deadband on `|g|` | 3 |
| `hold` | 1, 3, 5 trading days | 3 |
| `entry` | next `open`, next `close` | 2 |
| `weight` | `equal`, `proportional` to `|g|` | 2 |
| `neutral` | `none`, `market` (subtract net exposure times the SPY return) | 2 |

**Total configurations: 18 x 432 = 7,776.** Every configuration is evaluated on the identical event
set, identical prices and identical window.

### Why these are called *undisclosed* choices

Not that they are economically meaningless: a five-day hold is not a one-day hold. The claim is
narrower and checkable: (i) no theory in the LLM-trading literature determines any of these levels,
and (ii) papers reporting LLM trading results overwhelmingly report a single setting of each without
justification or sensitivity. The paper substantiates (ii) with a coding of recent published
pipelines rather than asserting it.

## 4. The reported statistics

* **Annualised Sharpe** `mean/sd * sqrt(252)` over the 243-session daily portfolio return series.
* **Dispersion**: sd, and the 5th/50th/95th percentiles, of Sharpe across the 7,776 configurations.
* **Variance decomposition**: balanced-design sum-of-squares share of Sharpe variance attributable
  to each of the 10 factors, plus the interaction residual.
* **Effective number of independent trials** `N_eff`: the participation ratio
  `(sum lambda)^2 / sum lambda^2` of the eigenvalues of the 7,776 x 7,776 correlation matrix of
  daily configuration returns (primary), with the Li-Ji eigenvalue count as a secondary estimate.
* **Deflated Sharpe ratio** (Bailey and Lopez de Prado) for the best-of-grid configuration, computed
  at `N = 7,776` and at `N = N_eff`.
* **Exact permutation null**: `R = 200` replications in which each event's score is permuted among
  events filed on the same date, destroying the event-to-score link while preserving the daily
  score distribution and the number of positions per day. The entire post-LLM grid is re-run on each
  replication, giving the null distribution of the best-of-grid Sharpe and an exact p-value.
* **Reference configurations**: the `default` configuration (the modal choice in the literature:
  lenient parse, level signal, no deadband, 1-day hold, close entry, equal weight, no
  neutralisation), the `median` configuration, and the `best` configuration.

## 5. Hypotheses, stated before the answers exist

* **H1 (dispersion).** The standard deviation of annualised Sharpe across the 7,776 configurations
  is at least 0.5.
* **H2 (harness over model).** The share of Sharpe variance attributable to `model` is smaller than
  the share attributable to the union of the seven post-LLM factors.
* **H3 (correlated grid).** `N_eff` is below 10% of the nominal grid size, so neither `N = 1` nor a
  Bonferroni-style `N = 7,776` is the right multiplicity correction.
* **H4 (selection).** Reported as a two-sided test, not a directional prediction: the permutation
  p-value of the best-of-grid Sharpe is reported whatever it is. If the LLM pipeline's best-of-grid
  Sharpe is significant against the permutation null, that is reported as such and the paper's claim
  narrows to the size of the selection premium.
* **H5 (approximation quality).** The deflated Sharpe ratio computed at `N = N_eff` is closer to the
  exact permutation p-value than the deflated Sharpe ratio computed at `N = 7,776` or at `N = 1`.

### Falsification

The paper's central claim is dead, and will be reported as dead, if **either**
`sd(Sharpe) < 0.2` (there is no meaningful harness dispersion to correct for)
**or** the `model` factor accounts for a larger variance share than all post-LLM factors combined
(the model, not the harness, is what matters).

## 6. What is deliberately NOT varied

Random seed and decoding temperature are fixed (`temperature = 0`, `seed = 7`) so that the reported
dispersion is attributable to design choices rather than to sampling noise; a separate seed-variation
check on one configuration bounds the residual stochastic contribution. The universe, the sample
window, the benchmark, the cost assumption and the tradeability rule are fixed for every
configuration, so none of them can generate between-configuration dispersion.
