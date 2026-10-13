# Cross-asset GBP benchmarks and scenario analysis

## Existing v0.2 components reused

| Existing component | Reuse |
|---|---|
| cross_asset.market | Yahoo provider, real exchange sessions, strict price validation and cached native prices |
| cross_asset.cache | Immutable FX snapshots, request pointers and SHA-256 provenance |
| data.to_gbp | GBP price conversion after explicit timestamp alignment |
| cross_asset.signals | 5/20/60/120-session momentum, trends, volatility, correlations, relative strength and historical scores |
| statistics | Simple and rolling returns, cumulative wealth, volatility, drawdowns and historical risk ratios |
| covariance | Ledoit–Wolf estimate for current-allocation volatility |
| cross_asset.backtest | Audited close-to-close accounting, drift, rebalancing and self-financing fees |
| config/cross_asset.yaml | Existing research proxies, asset-class labels, calendar and signal parameters |

The new modules add FX availability/alignment, benchmark composition, explicit
terminal scenario calculations and artifact orchestration. The simulator gains a
fixed-allocation entry point around the same ledger; existing walk_forward behavior
and its regression tests are preserved. No duplicate return/risk calculator or
market downloader was imported from the prototype.

## Prototype integration

Source: supplied cross_asset_scenario_engine.zip. Its literal four scenario shock
sets and MODEL_WEIGHTS were translated from percentages and class names to decimal
returns and existing ticker identifiers in config/scenarios.yaml. No attachment code
or compiled bytecode was executed. Disinflation/soft landing, persistent (sticky)
inflation, recession and risk-on rebound remain manually chosen assumptions.

The prototype has 13 proxies; v0.2 also includes UUP. To require complete scenario
coverage, UUP's native USD return is explicitly assumed zero in every default
scenario. This added illustrative assumption is editable and **not an empirical
estimate**. UUP's native dollar-basket exposure is separate from GBP/USD translation.
Two additional illustrative allocations demonstrate defensive/growth comparisons.
No model weights are recommendations or competition constraints.

## Launch locally

From the repository root:

```bash
uv sync --locked --extra dev --extra dashboard
# Live provider data, six years requested by default (requires internet):
uv run --locked aycapital-scenarios
uv run --locked --extra dashboard streamlit run scripts/scenario_dashboard.py
```

The dashboard URL is http://localhost:8501. It reads outputs/scenarios by default.
For fixed dates matching the validated live run:

```bash
uv run --locked aycapital-scenarios --start 2020-10-13 --end 2026-10-13 --output outputs/scenarios
```

An explicit offline synthetic illustration is also available:

```bash
uv run --locked aycapital-scenarios --source synthetic --start 2018-01-01 --end 2024-01-01 --output outputs/scenarios_synthetic
uv run --locked --extra dashboard streamlit run scripts/scenario_dashboard.py
```

Select outputs/scenarios_synthetic in the sidebar. Source labels prominently mark
synthetic prices/FX; a live failure never substitutes synthetic data. The dashboard
never performs downloads. Regenerate artifacts to update data; --refresh preserves
previous immutable snapshots while fetching new ones. Generated outputs are ignored
by Git. The existing research dashboard and CLI remain available.

## Historical GBP evidence

All initial research funds are USD quotes. Yahoo GBPUSD=X is **USD per GBP**; convert
P_GBP = P_USD / GBPUSD. Changing FX affects returns, volatility, correlations and
relative performance, not merely displayed units. Currency/unit declarations still
require user verification; this adapter rejects native currencies other than USD.

Yahoo daily FX dates do not provide verified release instants synchronized with
exchange closes. The explicit conservative convention makes a quote available at
UTC midnight after its observation date. For each actual asset session close,
backward matching chooses the most recent quote already available. It may reuse a
quote over a weekend/holiday, but never silently fills or interpolates observations.
Maximum age is six calendar days by default;
unavailable/older quotes fail. This includes the inspected 2025 Easter gap: the
2025-04-22 equity close uses the 2025-04-17 FX observation under the conservative
availability convention (5.83 calendar days old). This extended reuse is explicitly
warned and may distort synchronous GBP risk; reduce the configured limit to reject
it or supply higher-quality timestamped FX rather than inventing a quote.
Missing raw FX quotes are retained in cache and logged,
then explicitly excluded from the usable quote set; no replacement observation is invented.
If Yahoo supplies an extra observation beyond the exclusive end, it is retained in
raw cache, logged and explicitly excluded from conversion. Neither missing nor
out-of-range values can become invented dated observations. The
matched observation date, availability timestamp, quote and age are exported in
fx_alignment.csv and shown in the dashboard, including historical selected dates.

This is an asynchronous, dated-FX research valuation series, **not** a verified
simultaneous equity/FX closing NAV. The delay can affect measured correlations and
risk. Precisely timestamped synchronous FX is a future data improvement; no daily
FX observation is assumed to have been known before its conservative availability.
Asset prices remain vendor-adjusted total-return approximations, subject to revisions.
Point-in-time corporate actions, historical universe membership and currency metadata
are not independently certified. The v0.2 audit limitations continue to apply.

Historical metrics include 5/20/40/60-session compounded GBP returns, 20/60-session
volatility, sample maximum drawdown, trailing 60-price drawdown, benchmark correlation,
trends, momentum and relative strength. Existing risk and score conventions are
unchanged. Return horizons count trading sessions, not calendar days. Zero-risk
ratios or risk-adjusted scores can be undefined and are left missing. A complete
200-session trend needs the full observation window.

The historical score ranks peers using existing v0.2 weights; it is not a calibrated
20–40-session return forecast. Its known redundant relative-strength/momentum ranking
is retained without post-hoc tuning. The dashboard's 20/40 window selects **trailing
historical evidence**, while scenario assumptions have their own fixed horizon.

## Historical benchmarks and allocations

US equities use SPY by default. The equal-weight cross-asset benchmark holds all
configured proxies and rebalances on the existing configured session interval
(default 20), with default 10 bps fees from the existing research config. SPY is
buy-and-hold. Benchmark relatives compare compounded returns as growth ratios:
(1 + asset return)/(1 + benchmark return) - 1.

Configured allocations use the same existing ledger: start in cash, enter at the
second session close, drift between rebalances, and solve fee-funded fractional
trades. Costs, calendar and rebalance interval are inherited from cross_asset.yaml.
Equal-weight and custom comparisons rebalance on the same schedule; SPY has entry
fees only. All portfolios are long-only, fully invested after entry and have no
GBP cash balance after fills. Zero cost is configurable; no forced final liquidation.

These are ex-post fixed-allocation histories, **not untouched out-of-sample strategy
validation**. Editing weights today and inspecting their past path cannot establish
performance of a historical decision rule. The GBP comparison reuses the ledger but
does not retrain or duplicate the existing walk-forward momentum strategy.
Current-allocation volatility uses a trailing 60-session shrinkage covariance;
historical path statistics use their full displayed sample. Both are empirical
historical estimates, not forward conditional scenario estimates.

## Hypothetical scenarios

config/scenarios.yaml explicitly declares kind: hypothetical. Each scenario contains
one native-USD terminal simple return for **every** configured asset and one
GBP-per-USD FX return. Units are fractions (0.08 means 8%). All default scenarios
cover 40 trading sessions; configuration can specify a 20–40-session horizon, but
changing the horizon does not automatically rescale or calibrate shocks.

For native shock r and GBP-per-USD shock f:

`GBP asset shock = (1 + r) * (1 + f) - 1`

Positive f means the USD strengthens versus GBP. Scenario portfolio return is the
sum of initial weights times GBP asset shocks. It describes a terminal buy-and-hold
stress with no intervening fees, trading, cash flows or path. Portfolio weights must
be finite, nonnegative, reference known assets and sum to one. Invalid weights are
rejected, never automatically normalized. Missing shocks are rejected, never treated
as zero. Asset/FX shocks must currently exceed -100%.

Dashboard edits change assumptions for the selected scenario and an illustrative
allocation. Those edits live in the session; YAML is the reproducible configuration,
and export buttons save the resulting tables. Asset scenario ordering is simply
ordering of assumed shocks. A worst configured scenario is neither VaR nor a
probability, and a terminal loss is not an interim maximum drawdown. Scenarios do
not carry likelihoods or validate predictions. Hypothetical comparisons, historical
metrics and empirical covariance estimates occupy separate labelled dashboard views.

## Proxy eligibility

Every configured ETF is a research proxy. UMushroom availability and UK 2026
competition eligibility remain unverified. No mapping or trading eligibility is
inferred from ticker, asset-class equivalence, chart availability or scenario rank.
Official instruments/ISINs, share classes, hedging, valuation currency and constraints
must be verified before any competition allocation. No brokerage execution exists.

## Empirical scenario calibration — future enhancement

1. Define historical regimes using publication-aware macro data and freeze definitions
   before evaluation; reuse v0.2 vintage/release guards.
2. Fit conditional joint asset/FX returns on training periods only, with adequate
   regime samples, selection-bias controls and uncertainty intervals.
3. Compare estimated scenarios against simple unconditional/benchmark baselines on
   genuinely untouched OOS periods; preserve a record of tested horizons/models.
4. Use blocked evaluation for overlapping 20/40-session outcomes, check regime
   instability and control multiple testing. Reassess costs and instrument coverage.
5. Add calibrated scenarios as a separately labelled provenance-bearing mode only
   after validation. Do not relabel current manual shocks as estimated relationships.

No empirical-calibration engine is implemented in this change.

## Validation

```bash
uv run --locked --extra dashboard pytest -q
uv run --locked ruff check .
uv run --locked aycapital-demo
uv run --locked aycapital-scenarios --source synthetic --start 2018-01-01 --end 2024-01-01 --output outputs/scenarios_synthetic
```

Tests cover prototype FX cancellation, future-quote exclusion, invalid/stale/missing
FX, existing cache/provider reuse, missing prices, shock compounding, incomplete
scenarios, invalid weights, weighted portfolio returns, benchmark identity and source/
observation-date exports. Optional Streamlit AppTest exercises scenario switching,
20/40 windows and historical date selection. With no dashboard extra, that one test
is skipped. Existing tests continue to exercise the shared ledger and causal signals.
Validation on this branch: 80 tests passed with the dashboard extra; Ruff and
whitespace checks passed. Both synthetic and live GBP artifact generation succeeded,
as did the original demo and live/synthetic dashboard interaction checks. Known
pandas/calendar NumPy timedelta deprecation warnings remain visible.
