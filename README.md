# A-Y Capital Partners — Version 0.1

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
