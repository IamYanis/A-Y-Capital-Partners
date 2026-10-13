# Development instructions

Version 0.1 is a research library, not a trading system. Use Python 3.12+ and
src/aycapital. Keep provider I/O in data.py; pure historical calculations in
statistics.py and covariance.py; allocation constraints in portfolio.py. demo.py
orchestrates these components using TOML configuration. Add future modules only
when requested. Research extras reserve dependencies for later milestones.

## Coding and verification

Use clear functions, explicit units, labelled pandas objects, descriptive errors,
and docstrings describing conventions. Avoid global state and implicit downloads.
Use deterministic synthetic fixtures and monkeypatch provider calls in tests.
Run `uv sync --locked --extra dev`, `uv run --locked pytest`,
`uv run --locked ruff check .`, and `uv run --locked aycapital-demo`.
Update README and uv.lock for dependency changes. Work on feature branches;
do not merge or push without user instruction.

## Quantitative safeguards

- Never fabricate market observations, realized performance, or forecasts.
- Mark simulated data and historical research explicitly.
- Competition rules and eligible assets remain unspecified until supplied.
  Example tickers and caps are illustrative; reject ineligible/infeasible inputs.
- Preserve provider provenance and cached snapshots; validate missing data,
  asset order, calendars, currencies and quote units. Never silently fill gaps.
- Yahoo adjustments are estimates: verify corporate actions and total returns.
- GBP FX must be aligned and expressed as GBP per quote unit; GBp is pence.
- Future backtests must shift signals, separate training and evaluation, include
  costs, and use point-in-time universes where possible. Report survivorship bias.
- Fit transforms and models on training observations only. Control multiple
  testing and compare against benchmarks in future research milestones.
- Annualize consistently; distinguish simple and log returns. No real trades.
- Current optimization uses data already observed and demonstrates allocations;
  it does not constitute a backtest or a forecast.

## Cross-asset Phase 1

The cross_asset package extends Version 0.1. Use config/cross_asset.yaml for proxy
metadata, signed ranking weights, fixed backtest settings and macro series. Keep
provider I/O separate from causal calculations. market.py uses actual exchange
sessions and retains immutable cache snapshots. Signals are post-close; backtest
fills are the following session close, earning returns only after that fill.
Turnover must reflect weight drift and buy-plus-sell notional, including cash entry.
IS/OOS accounts are independent; never splice their NAVs into one continuous curve.

Run the existing checks plus:
`uv run --locked aycapital-research --source synthetic --start 2018-01-01 --end 2024-01-01`.
The optional dashboard reads saved artifacts only. Test future-price perturbation
invariance, real calendar holidays and macro release/revision availability. Latest
FRED snapshots must never enter earlier backtests. Date-only vintages become
available the following UTC day; do not invent intraday release times. Keep secrets
out of logs and cached metadata. USD research is not GBP portfolio risk modeling.
Competition eligibility remains unspecified. Do not claim predictive accuracy.
