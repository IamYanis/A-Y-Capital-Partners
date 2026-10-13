# Cross-asset Phase 1 methodology

## Architecture

`aycapital.cross_asset` adds configuration, immutable snapshots, market providers,
causal signals, walk-forward paper simulation, macro retrieval and report generation.
Existing Version 0.1 statistics and strict price validation are reused. Optional
Streamlit presentation reads saved reports only. Production logic stays in src/;
notebooks and the dashboard must not implement separate research calculations.

## Data

The Yahoo adapter explicitly requests `auto_adjust=True` daily closes. Provider
interfaces allow replacement without changing signal or simulation functions.
Adjusted prices are vendor estimates, not verified total-return indices. Cash
flows, withholding and FX hedging cannot be inferred from these data.

The initial proxies are US-listed USD funds, including proxies for non-US underlying
exposures. USD quote currency is distinct from underlying economic exposure. All
use an XNYS session schedule including holidays, DST and early closes. Completed
sessions only are accepted. Calendar metadata and actual UTC close timestamps are
exported. The CLI defaults to the Europe/London date as exclusive end.

At least five calendar years are requested; the configured default is six. Each
instrument's raw series is cached with request parameters, retrieval timestamp,
source and content digest. Refresh writes new immutable CSV/metadata snapshots and
atomically changes a request pointer, preserving older snapshots. Reproduction uses
the same date range, config and cached snapshot; record metadata.json and archive
outputs before refreshing. Yahoo can revise history despite adjusted prices.

Leading unavailable observations (e.g. before fund inception) are logged; the panel
begins at the latest first available asset quote. At least 201 common completed
sessions are required. Internal and trailing missing sessions, duplicates, invalid
prices or timestamps fail validation. No fabricated fallback or silent forward-fill
is allowed. Different listing calendars/currencies require explicit future adapters.

## Signal definitions

All metrics at t depend on prices through the completed t close only:

| Metric | Definition |
|---|---|
| 5/20/60/120-session momentum | P[t] / P[t-n] - 1 |
| 20/50/200-session trend | P[t] / mean(last n prices) - 1 |
| Volatility-adjusted momentum | 60-session momentum / annualized 60-session volatility |
| Relative strength | (1 + 60-session return) / (1 + benchmark 60-session return) - 1 |
| 20/60-session volatility | Sample standard deviation of last n simple returns × sqrt(252) |
| Benchmark correlation | Correlation of last 60 simple returns against SPY by default |
| Rolling maximum drawdown | Worst peak-to-trough loss entirely within last 60 prices |
| Downside volatility | sqrt(mean(min(simple return, 0)^2)) over 60 sessions × sqrt(252) |

Annualization is configurable. Windows count sessions, not calendar days. Signal
exports mark the session and its actual close UTC; data becomes usable **after**
that close. A 200-price average first exists after 200 observations; return windows
need an additional starting price. Zero volatility produces undefined risk-adjusted
momentum rather than an infinite score.

Each configured metric becomes a contemporaneous cross-sectional midrank scaled
from zero to one: (rank - 1)/(available asset count - 1). Ties share rank; a sole
available value is neutral at 0.5. Negative weights invert preference (e.g. lower
volatility); scores average preferences with absolute-weight normalization.
Missing required metrics leave an asset unscored. The backtest requires complete
universe scores after warmup and rejects later gaps. Stable ticker insertion order
breaks ranking ties for top-N selection. No learned transforms or full-sample fitting.
Scores compare peers, not expected returns, and can favor an asset in a declining market.

## Walk-forward accounting

Fixed ex-ante ranking weights and top-N parameters are applied sequentially. The
configured fraction of eligible evaluation sessions defines the in-sample boundary;
the remainder is out of sample. There is no fitted model in Phase 1, and the split
is not evidence of predictive accuracy. Changing parameters after seeing either
segment contaminates evaluation. Rolling history preceding the OOS boundary is
permitted to compute OOS signals, but future observations never enter those signals.

A signal computed after session t close is filled at session t+1 close. Old holdings
earn the t-to-t+1 return; new holdings first earn t+1-to-t+2. The model does not assume
access to the next opening price or same-close execution. Exchange sessions determine
fills, excluding holidays and using actual early close times. A paper-trade log
records signal availability, fill timestamp, weights, turnover and costs.

Strategy holds equal weights in top N scores and rebalances every configured number
of sessions. SPY is buy-and-hold; equal-weight benchmark rebalances on the same
schedule as the strategy. All accounts begin in cash; no pre-fill asset returns are
earned. Both IS and OOS restart at cash NAV one, using a pre-start signal for their
first next-close fill. Curves are shown separately, not spliced into one account.
Cash earns zero; long-only research weights sum to one after fills.

Between rebalances weights drift with realized returns. Traded notional is
sum(abs(target - pre-trade drifted weight)), without halving. Initial cash-to-assets
turnover is one; a complete asset rotation is two. Proportional bps costs apply to
this pre-trade NAV notional, after the old holdings' session return. Target weights
then describe post-cost NAV. This common approximation does not solve exact
fee-dependent trade quantities and omits slippage, market impact and spreads.

CAGR uses compounded net return raised to periods_per_year / number_of_sessions.
Sharpe is annualized arithmetic mean / sample standard deviation with zero risk-free
rate. Annualized volatility and maximum drawdown reuse Version 0.1 conventions,
including starting NAV one. Total and annualized turnover use buy-plus-sell notional.
Performance includes entry costs in every account/split; no forced final liquidation.

## Macro data and publication safeguards

FRED series: DGS2, DGS10, BAMLH0A0HYM2, CPIAUCSL, PCEPI and DTWEXBGS. The 10Y–2Y
spread is calculated only where yields share dates. Inflation YoY comparisons use
matching calendar months 12 months earlier, not the prior 12 daily rows. No period
imputation is used. Each macro indicator gets its own chart to retain units.

Latest mode requests a snapshot as of a specified date. Such history can contain
revisions and is labelled descriptive/chart-only; available_as_of rejects it.
Vintage mode requests FRED observation real-time intervals through the as-of date
with pagination, preserving observation date, vintage date and validity end. Vintage
records dated before their observation periods are rejected as unsuitable release
metadata. As-of queries use only vintages available before the timestamp and then
choose the most recently reported period, incorporating revisions only after their
availability. Published missing values stay missing rather than reviving old values.

FRED supplies date-level vintage information, not necessarily intraday release
instants. Availability is conservatively set to UTC midnight the following day.
This is a documented delay, not an invented actual release time. It cannot repair
missing vintage history; reliable intraday release calendars are Phase 2 work.
Macro retrieval/plots are available but macro is not a strategy input in Phase 1.
FRED_API_KEY stays in environment memory, not config, cache or logs.

## Primary references

- [Yahoo download API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)
- [FRED observation API and real-time fields](https://fred.stlouisfed.org/docs/api/fred/series_observations.html)
- [FRED real-time periods](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html)
- [Exchange calendar implementation](https://github.com/gerrymanoim/exchange_calendars)

## Phase 2

Map proxies to verified competition instruments and constraints; add point-in-time
universes and corporate actions; model GBP FX; evaluate nested walk-forward parameter
selection, sensitivity, multiple-testing controls and benchmark robustness; improve
cost/impact simulation and macro publication timestamp coverage. No brokerage execution.
