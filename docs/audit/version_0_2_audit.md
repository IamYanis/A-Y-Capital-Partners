# Version 0.2 quantitative research audit — 13 October 2026

## Conclusion and scope

PR #2 can be reviewed as a research MVP after the defects below are fixed and its
checks pass. It is **not competition-ready**: eligible instruments/rules, GBP risk
and realistic execution remain unverified or unimplemented. Scores are historical
price evidence; stable ranks are not proof of predictive returns.

Audited baseline: ee83e25 on feature/version-0.2. Code inspection covered signals,
market data/cache, macro vintages, portfolio accounting, configuration and outputs.
Tests include adversarial future-data changes and independent share-ledger accounting.
Empirical diagnostics use the existing immutable Yahoo snapshots: 14 assets,
1,506 sessions, 2020-10-13 through 2026-10-12. Derived evidence and per-asset snapshot
hashes are in version_0_2/. No synthetic data is presented as historical evidence.

## Critical defects — fixed on the existing feature branch

### C1. Expired macro vintages could be reused

`available_as_of` previously selected records by vintage start but ignored
`realtime_end`. An observation that ceased to be valid without a replacement
continued to appear known. It also carried values beyond the snapshot's as-of
coverage. This is a critical temporal-data integrity defect for future historical
macro use; **the current strategy was unaffected because macro is chart-only**.

Regression cases first failed on the original code. Fixes:

- Validate interval ends and reject overlapping vintage intervals.
- Apply the conservative one-day UTC delay to the entire information clock,
  including interval expiry, not just release starts.
- Return missing for an expired latest period; do not resurrect an earlier value.
- Return missing beyond a snapshot's declared as-of coverage.
- Reject provider observation/vintage dates outside the requested historical range
  before caching, and check the same bounds on cache reads.

Tests cover expiry without replacement, delayed revision transitions, retrieval
coverage, reversed/overlapping intervals and provider future-vintage contamination.
FRED real-time periods are inclusive at both boundaries; date-level vintage records
still do not establish actual intraday publication times. Live FRED verification is
pending an API key. No macro predictive relationships were assumed or added.

## Accounting issue — fixed

### M1. Costs used approximate trade quantities

The previous method charged c × sum(abs(target - old)) and then assigned target
weights to reduced NAV. Actual purchases therefore differed from charged notional.
This was documented as an approximation, but could become material with high costs
or repeated rotations. The accepted cost configuration included high rates.

The simulator now solves the self-financing equation:

`u + c * sum(abs(u * target - old)) = 1`

Here u is post-fee/pre-fee NAV, and old is drifted pre-trade weight. Fees and turnover
use actual fractional trade notional. A cash purchase has u = 1/(1+c); a complete
asset rotation has u = (1-c)/(1+c). This changes prior numerical backtests slightly.
Independent currency/share-ledger bisection agrees with the portfolio implementation.
No slippage, spread, impact or whole-share restrictions were added.

Infinite externally supplied scores now fail validation rather than entering ranks.

## 1. Look-ahead bias

**Pass, conditional on the supplied price snapshot.** All production metrics use
trailing prices or contemporaneous cross-sectional ranks. No centered windows,
negative shifts or full-history fitting occur in signal generation. Future-price
perturbation tests preserve earlier metrics, scores, rebalance decisions and returns.
The 200-session moving average requires its full trailing window.

A signal based on close t is available after the actual UTC close. Execution occurs
at close t+1, and new holdings first earn t+1-to-t+2 returns. Trades log both times.
This avoids same-close execution and attributing pre-fill returns to selected assets.

Caveats: the API accepts caller-supplied scores, whose provenance it cannot prove;
the audited guarantee applies to the provided calculate_signals pipeline. Current
vendor adjusted history is not a point-in-time vintage database. Historical vendor
corrections and corporate-action revisions can alter old observations. A causal
formula alone cannot repair those source limitations.

The split is a configured fraction of the requested evaluation sample, not a learned
parameter. Freeze dates, universe, weights and cutoff before any future evaluation.
These diagnostic periods have now been inspected and must not be reused as untouched
OOS evidence after tuning the model.

## 2. Data integrity and macro lags

- Explicit Yahoo auto-adjustment is requested. An additional live download with
  auto_adjust=False compared its Adj Close against the cached adjusted panel for
  every proxy. Maximum relative difference: 1.162034e-6 (~1.2 ppm). Small differences
  indicate vendor retrieval/rounding/revision effects; exact equality is not promised.
- All 14 cached series contain 1,506 finite positive prices, with zero missing
  observations on the common session schedule. No invented market values or missing
  value filling is used. See adjustment_crosscheck.csv for per-asset evidence.
- Real exchange holidays, early closes and UTC close times are checked. Incomplete
  sessions are excluded. Internal/trailing gaps fail; leading unavailable history
  is logged and trimmed to the latest first available quote across assets.
- **Minor limitation:** leading missing history is not independently verified as
  fund inception. It could also represent a provider outage. Before extending the
  historical universe, verify inception/delisting dates and point-in-time membership.
- Static surviving ETFs introduce survivorship/selection bias. The common sample
  starts after the 2020 pandemic crash; that regime is not covered. Warmup further
  limits scored observations to 2021-07-29 onward.
- Macro latest snapshots remain descriptive only. They are rejected by historical
  as-of queries. Revised vintages enter only after conservative next-UTC-day
  availability, with the new expiry/coverage safeguards. Intraday release calendars,
  publication completeness and live FRED retrieval remain unverified.
- YAML quote currency is user-supplied metadata, not a provider-certified unit check.
  A unit change or erroneous native-currency declaration could pass positive-price
  checks. Add provider currency/corporate-action diagnostics before competition use.

## 3. Backtesting and benchmarks

**Timing and holdings accounting pass tested invariants.** Old holdings earn returns
through the execution close, weights drift between rebalances, and fixed top-N
weights are reset at scheduled eligible sessions. No holiday fills occur. Strategy
and equal-weight benchmark share a rebalance schedule. SPY is buy-and-hold with an
entry fee. The independent ledger test verifies fee-funded trades and net returns.

IS and OOS are independent accounts starting in cash; charts remain separate. A
pre-start signal is permitted because it is already observable. Entry fees apply to
all accounts; no terminal liquidation is assumed. Sharpe uses zero risk-free rate,
and CAGR/volatility use 252-session annualization. Short/nonstandard samples can make
these estimates noisy; report calendar-time CAGR as an additional diagnostic later.

**Minor limitations:** fixed rules are sequentially evaluated, not nested trained
walk-forward models; no parameter estimation is currently performed. Cash earns
zero, spreads/impact/liquidity are absent, and adjusted-price fills represent a
fractional total-return proxy rather than actual executable share-price orders.
Benchmarks are comparisons within the same USD proxy world, not verified competition
benchmarks. None of these results establishes operational execution quality.

Cost sensitivity for the inspected OOS account (descriptive, not tuned):

| Cost per traded notional | CAGR | Sharpe | Maximum drawdown |
|---|---:|---:|---:|
| 0 bps | 10.24% | 0.805 | -12.08% |
| 10 bps | 9.02% | 0.719 | -12.69% |
| 25 bps | 7.21% | 0.591 | -13.61% |
| 50 bps | 4.26% | 0.378 | -15.11% |

Annualized buy-plus-sell turnover is approximately 11.1 times NAV, making execution
cost assumptions economically material. This is historical vendor-data research,
not a forecast or an expected competition return.

## 4. Signal stability across historical slices

Calendar years are diagnostic slices, not fitted macro regimes. 2021 includes only
post-warmup observations; 2026 ends on 12 October. 2022 covers a negative SPY period
(-18.18%) with higher observed volatility (24.24% annualized), while 2023/2024 have
positive SPY returns and lower volatility. Labels use ex-post observations solely
for audit comparisons and do not enter any production decision.

| Slice | Scored sessions | Mean daily score rank persistence | Risk rank persistence | Mean 20-session forward rank IC | Mean 40-session forward rank IC |
|---|---:|---:|---:|---:|---:|
| 2021 partial | 109 | 0.947 | 0.997 | -0.215 | -0.170 |
| 2022 | 251 | 0.962 | 0.997 | 0.042 | 0.068 |
| 2023 | 250 | 0.950 | 0.996 | 0.002 | -0.048 |
| 2024 | 252 | 0.953 | 0.997 | -0.063 | -0.153 |
| 2025 | 250 | 0.946 | 0.997 | -0.029 | -0.012 |
| 2026 partial | 195 | 0.952 | 0.998 | -0.014 | -0.073 |

Persistence is cross-sectional Spearman between consecutive session rankings. Forward
IC compares a signal at t with cross-asset returns from next-close entry t+1 to
exit t+1+h (h=20/40 sessions, approximate 4/8 weeks). Forward endpoints remain within
each slice, with incomplete horizons excluded. These **ex-post** evaluation returns
never enter production signals. Windows overlap; no significance, independent-sample
or causal/predictive claim follows from their mean correlations.

Findings: score/momentum ordering is fairly persistent, and risk ordering is even
more persistent. Top-four day-to-day retention averages 84–93%. Changing both risk
weights by ±20% preserves approximately 95–99% of selected names on average. However,
forward IC changes sign across slices and is often negative: stable scores do not
show robust directional predictive reliability.

**Minor model issue:** 60-day relative strength is a common monotonic transformation
of 60-day return when all assets share the same positive benchmark denominator.
Their cross-sectional rank correlation is exactly 1.0 in this sample. Separate
weights double-count this signal (momentum60=2, relative strength=1 out of total
absolute weight 10). Other trend/momentum components are correlated too. The
configuration is left unchanged to avoid post-audit optimization. Consolidate
redundant components in a preregistered future model, then assess fresh OOS data.

## 5. USD/GBP exposure — competition-use blocker

This engine reports USD returns and risk. It does **not** convert the USD account
into GBP or include USD/GBP fluctuations in strategy/benchmark NAV. For an unhedged
USD asset with X measured as GBP per USD:

`1 + r_GBP = (1 + r_USD) * (X_t / X_(t-1))`

GBP covariance and drawdowns must be estimated after aligned conversion. Merely
changing report_currency or multiplying prices by a constant is not a currency
exposure model. The existing to_gbp utility supports aligned rates, but the cross-
asset pipeline does not wire FX into its simulation. Underlying economic currency
exposure can differ from listing currency; UUP is a dollar basket proxy, not a
GBP/USD conversion series. Hedged and unhedged share classes require distinct data.

Required before competition allocations: verify the competition valuation currency,
obtain reliable FX at matching valuation times, calculate all portfolios/benchmarks
in that currency and add currency limits if official rules require them.

## 6. Instrument availability and eligibility — competition-use blocker

SPY, QQQ, IWM, VGK, EEM, SGOV, IEF, TLT, GLD, DBC, XLE, LQD, HYG and UUP are configured
research ETF proxies. **UMushroom platform availability and UK 2026 competition
eligibility are unverified for every one of these instruments.** Unknown does not
mean unavailable or ineligible; it means no verified mapping has been entered.

Public checks did not yield a reliable current eligible-instrument list. The
UMushroom page at the UK competition path returned FAQ content describing a Swiss
2025 challenge (CHF currency and 2025 dates); it must not supply UK 2026 rules or fill
assumed constraints. General website listings would not establish competition
eligibility either. No rules from this page were imported.

Obtain the official current competition universe and rule set, then map each proxy
to the exact offered instrument/ISIN, share class, quote currency, exchange and
hedging status. Equivalent asset classes are not equivalent instruments. Enter
verified eligibility, position/exposure/cash limits and scoring/valuation mechanics
before translating any research rank into an allocation recommendation.

## Recommended improvements, in order

1. Official competition instrument/rule mapping and valuation-currency verification.
2. Audited GBP FX integration; provider unit/corporate-action/inception checks and
   point-in-time universe/data coverage.
3. Freeze a new evaluation cutoff and preregister a nonredundant signal model;
   reserve genuinely untouched OOS periods, with blocked confidence intervals and
   multiple-testing controls for horizon/model comparisons.
4. Stress test longer adverse regimes, universe changes and realistic execution
   costs; add cash interest, spreads/impact and share/valuation constraints.
5. Verify FRED publication timestamps and coverage, then consider macro predictive
   research only in a separately tested, release-aware experiment.
6. Address dependency deprecation warnings and optional calendar-time metrics.

## Reproduction and validation

From the repository root:

```bash
uv sync --locked --extra dev
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked python scripts/audit_cross_asset.py
# Optional live same-vendor adjusted-close check:
uv run --locked python scripts/audit_adjustments.py
uv run --locked aycapital-research --source synthetic --start 2018-01-01 --end 2024-01-01
```

The audit script uses the original market request/cache by default and retrieves
real data if that cache is absent. A refreshed vendor snapshot may produce different
results; provenance.json records the hashes used here. It writes only descriptive
regime/cost diagnostics, not fitted parameters. The adjustment comparison is a
separate live same-vendor check with its own timestamp. All 60 tests and Ruff passed locally; both the original demo and audited market
report ran successfully. Full validation results are recorded in PR #2; dependency timedelta warnings are known and not suppressed.

## Primary sources

- [FRED inclusive real-time validity periods](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html)
- [FRED observation and vintage API](https://fred.stlouisfed.org/docs/api/fred/series_observations.html)
- [Yahoo adjustment/download interface](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)
- [UMushroom public UK-path page checked](https://umushroom.com/de/wettbewerb/uk-investmentcompetition)
