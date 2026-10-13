# A-Y Capital Partners — Version 0.2

A Python quantitative research foundation for the Umushroom UK Investment
Competition 2026. Competition rules and eligibility have **not been configured**.
Example tickers and portfolio limits are demonstrations, not competition rules or
investment recommendations. No real trade execution is implemented.

## Install

Install Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).
From the repository root:

```bash
uv sync --locked --extra dev
uv run --locked pytest
uv run --locked ruff check .
uv run --locked aycapital-demo
```

`.python-version` selects Python 3.12; `uv.lock` records exact resolved dependencies.
The project supports Python 3.12+; verification uses the locked Python 3.12 environment.
Research/dashboard extras are optional: `uv sync --locked --extra dev --extra research`.
An alternative, unlocked installation is `python -m pip install -e '.[dev]'` in
an existing Python 3.12+ virtual environment.

## Demonstration

The default demo uses seeded **synthetic prices**, labelled as such. It prints
simple returns, historical statistics, annualized sample and Ledoit–Wolf
covariances, and constrained minimum-variance weights. Outputs and provenance
are written to `outputs/demo/`; generated files are ignored by Git.

```bash
uv run --locked aycapital-demo --config config/demo.toml --output outputs/example
# Optional real provider request (requires internet; availability is not guaranteed):
uv run --locked aycapital-demo --source yahoo
# Equivalent convenience script:
uv run --locked python scripts/demo.py
```

Edit `config/demo.toml` to change example tickers, dates (end exclusive),
annualization, risk-free rate and maximum weight. Yahoo mode requires an explicit
GBP-per-quote-unit factor for each asset. Constant factors suit known GBP/GBp
quotes only: for foreign currencies, use the Python `to_gbp` API with reliable,
date-aligned FX rates. Never infer currency or units from a ticker suffix.
Competition rules remain unconfigured; enter verified `eligible_assets` and
translate official constraints into the supported settings before competition use.
Setting `rules_configured` is a user assertion, not certification of compliance.

Optional exposure constraints:

```toml
[portfolio.exposure_caps.example_group]
assets = ["VOD.L", "HSBA.L"]
max_weight = 0.7
```

## Architecture and conventions

- `src/aycapital/data.py`: strict price validation, adjusted Yahoo loading,
  SHA-256-checked CSV snapshots with UTC retrieval metadata, explicit GBP conversion,
  and deterministic synthetic examples. Use `refresh=True` to request a new snapshot.
- `statistics.py`: simple/log price returns, compounded and rolling simple returns,
  rolling and annualized volatility, drawdowns, skewness, excess kurtosis, Sharpe
  and Sortino. Sample volatility uses ddof=1; annualization defaults to 252 periods.
  Annual effective risk-free rates convert to per-period simple rates. Sortino uses
  the root mean square of negative excess returns across all observations.
  Zero-risk ratios are NaN; short samples may have undefined higher moments.
- `covariance.py`: sample (ddof=1) and centered Ledoit–Wolf shrinkage covariance.
  Both are annualized; Ledoit–Wolf uses its estimator's population convention.
- `portfolio.py`: fully invested, long-only minimum variance and optional
  mean-variance with supplied annual expected returns, maximum position weights,
  eligibility and group exposure caps. CLARABEL solves the convex program; failed
  or infeasible results are rejected. Expected returns are never inferred as forecasts.
- `tests/`: deterministic math checks, analytical optimization solutions,
  constraint failures, FX conversion, mocked provider/cache behavior and demo integration.
- `config/`, `docs/`, `notebooks/`, `scripts/`: settings, methodology, future exploratory
  work and the demonstration entry point. Notebooks must not duplicate production logic.

## Research limitations

Yahoo adjusted prices approximate corporate-action-adjusted returns; they are not
verified total-return indices. Data may be revised, incomplete, rate limited or
subject to provider usage restrictions. Quote currencies and calendars require
verification. Missing observations are rejected; choose and document a common
calendar explicitly rather than silently filling or dropping data. Cached refreshes
replace the same request snapshot; archive it separately before refreshing if needed.

Static example universes are not point-in-time universes and may suffer survivorship
bias. This milestone contains no predictive signals, backtest, performance forecast,
transaction costs or trading. Optimization reflects the supplied historical risk
estimate; it does not evaluate out-of-sample portfolio performance. Sample covariance
can be singular and historical risk can change. Competition restrictions beyond the
supported eligibility/position/group caps must be added before making allocations.

## Version 0.2 candidates

Point-in-time universe and competition-rule ingestion, audited FX/total-return data,
lead-lag research with multiple-testing controls, shifted signals and walk-forward
backtests with costs, benchmarks and sensitivity analysis. These are future work.

## Cross-Asset Research Engine — Phase 1

The new engine extends the Version 0.1 library; all original APIs and the demo
remain available. It ranks 14 configurable liquid US-listed cross-asset proxies.
These USD instruments are **not** an eligible competition universe. The scores
summarize price evidence, without predicting a calibrated 4–8 week return.

```bash
uv sync --locked --extra dev
# Real adjusted prices: six years requested by default, ending before today.
uv run --locked aycapital-research
# Fixed dates and explicit output directory for reproducible research:
uv run --locked aycapital-research --start 2020-10-13 --end 2026-10-13 \
  --output outputs/cross_asset_live
# Entirely offline, clearly labelled synthetic demonstration:
uv run --locked aycapital-research --source synthetic \
  --start 2018-01-01 --end 2024-01-01 --output outputs/cross_asset_synthetic
# Optional FRED charts; set FRED_API_KEY in your environment first:
uv run --locked aycapital-research --macro
# --refresh downloads new snapshots while retaining previous snapshot files.
```

Open `outputs/cross_asset/report.html` in a browser for a standalone interactive
report (Plotly is embedded; no internet needed to view it). Reports also export
ranked latest signals in CSV/JSON, per-instrument signal history, 60-session rolling
correlations, strategy/benchmark equity curves, turnover, performance statistics,
paper rebalance records and provenance. Synthetic and live runs should use separate
output directories. Repeating the same configuration/dates with the same market
snapshots reproduces the numerical outputs; generation timestamps naturally differ.

`config/cross_asset.yaml` defines asset classes, currencies, listing calendars,
benchmark, history length, signed ranking weights, fixed backtest parameters and
FRED series. All initial proxies use USD quotes and a common XNYS calendar. Mixed
calendars or currencies are rejected rather than silently conflated. The existing
`to_gbp` API can handle explicit synchronized FX conversion; this engine currently
reports USD research, **not GBP portfolio exposure**.

Signal definitions and assumptions are documented in
[docs/cross_asset.md](docs/cross_asset.md). In brief, rolling price metrics are ranked
against peers at each date, with no full-history normalization. Post-close signals
execute at the next session close; the first earned return follows that fill.
In-sample and out-of-sample results are separate accounts using fixed rules;
there is no automatic hyperparameter fitting or claim of predictive validation.

### Optional saved-results dashboard

The dashboard reads completed artifacts and never downloads or trades:

```bash
uv sync --locked --extra dev --extra dashboard
uv run --locked --extra dashboard streamlit run scripts/cross_asset_dashboard.py
```

Select the appropriate output directory in its sidebar. Synthetic results retain
explicit source labels. The richer interactive signal and macro figures remain in
the standalone HTML report.

### Verification

```bash
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked aycapital-demo
uv run --locked aycapital-research --source synthetic \
  --start 2018-01-01 --end 2024-01-01 --output outputs/verification
```

Tests use deterministic synthetic observations and mocked market/FRED providers.
They test exchange holidays/early closes, missing/incomplete data, immutable caches,
release dates and revisions, signal formulae, future-price perturbations, delayed
execution, drift-aware turnover, costs, split accounts and report integration.
Calendar/pandas dependencies currently emit NumPy timedelta deprecation warnings;
they do not prevent the test suite from passing.

### Phase 1 limits and Phase 2 priorities

- Vendor adjustments are total-return approximations with no historical vintage
  guarantees. Static proxies introduce survivorship/selection bias. A cache snapshot
  improves reproducibility but does not make prices point-in-time corporate-action data.
- Missing internal/trailing sessions fail validation. Leading unavailable history
  is logged and trimmed to the common sample. This can shorten history below five
  years for newly listed proxies; five-plus years are requested where available.
- Next-close fills omit intraday spreads, impact, liquidity limits and realistic
  order execution. Costs solve a self-financing proportional buy-plus-sell notional fee equation; cash earns zero. This is a research approximation.
- FRED needs an API key. Latest snapshots are chart-only and cannot enter earlier
  backtests. Vintage mode uses actual FRED real-time dates with a conservative
  next-UTC-day delay because intraday publication times are unavailable. Macro is
  never used in the Phase 1 strategy. Complete vintage coverage is not guaranteed.
- Official competition rules, GBP exposure, position sizing and execution remain
  unimplemented in this engine. No live trading or brokerage integration exists.
- Phase 2: verified competition mapping/constraints, point-in-time universe and
  prices, audited GBP FX, nested walk-forward parameter selection, costs/slippage
  sensitivity, macro release-timestamp validation and multiple-testing controls.

## Version 0.2 quantitative audit

See [the research audit](docs/audit/version_0_2_audit.md) for temporal-data fixes,
independent fee-accounting checks, historical regime diagnostics and competition-use
limitations. Macro as-of queries now honor validity ends and retrieval coverage.
Backtest fees solve self-financing trade quantities, so older output artifacts must
be regenerated. Rankings are persistent but forward rank correlations vary across
periods; predictive reliability is not established. GBP conversion and verified
UMushroom instrument/rule mapping remain required before competition allocations.

## GBP benchmark and scenario dashboard

The scenario module reuses v0.2 data, signals, risk and audited backtesting. It adds
GBP price evidence, US-equity/equal-weight benchmarks and editable **hypothetical**
macro shocks/portfolio comparisons. Historical rankings do not validate 20–40-session
forecasts; scenarios are manually chosen assumptions. Research proxies remain
separate from unverified UMushroom-eligible instruments.

```bash
uv sync --locked --extra dev --extra dashboard
uv run --locked aycapital-scenarios
uv run --locked --extra dashboard streamlit run scripts/scenario_dashboard.py
```

Open http://localhost:8501. The default saved data directory is outputs/scenarios.
For an explicit offline illustration, first run:

```bash
uv run --locked aycapital-scenarios --source synthetic --start 2018-01-01 --end 2024-01-01 --output outputs/scenarios_synthetic
```

Then choose outputs/scenarios_synthetic in the sidebar. See
[scenario assumptions and methodology](docs/scenarios.md) for FX availability,
benchmark accounting, prototype integration and future empirical calibration.
Edit config/scenarios.yaml for shock assumptions and portfolio weights. Equity/FX
observation dates and data-quality warnings are displayed; dated FX valuations are
not synchronized closing NAVs.
