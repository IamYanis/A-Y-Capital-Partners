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
