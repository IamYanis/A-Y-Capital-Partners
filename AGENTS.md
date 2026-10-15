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

Audit safeguards: honor macro vintage validity ends and retrieved as-of coverage,
with the same conservative delay for release and expiry. Costs must solve
self-financing fractional trade quantities rather than charge pre-fee target changes.
Maintain independent share-ledger and expired-vintage regression tests. Descriptive
regime diagnostics use future returns only for evaluation and never for signals;
overlapping horizons do not provide independent significance tests.

Scenario extensions: use cross_asset.fx for explicit available-at FX matching,
data.to_gbp for conversion, benchmarks.py for composition of existing metrics, and
scenarios.py for hypothetical terminal shocks. Reuse the backtest ledger through
fixed_weight_history; do not implement another portfolio simulator. Preserve old
walk_forward behavior. Every scenario needs complete asset shocks; never impute a
missing shock or normalize invalid weights silently. Native/FX shocks are fractions.
Keep hypothetical stress ranks separate from historical scores and any future
empirical calibration. Show asset/FX observation dates and asynchronous-valuation
warnings. Dashboard reads saved data only. Run dashboard integration tests with
`uv run --locked --extra dashboard pytest -q` and scenario CLI smoke checks.

Version 0.4 forecasting: build features from existing GBP snapshots and signal/
macro availability APIs. Every training label must end strictly before the decision
session; targets enter next close. Fit normalizers, regime thresholds and models
inside the eligible train window. Evaluate horizon-spaced nonoverlapping outcomes;
calibrate residual distributions only with matured earlier OOS errors. Use paired
block uncertainty and full-family multiplicity correction; never promote a failed
baseline comparison, unsupported regime or synthetic result to predictive skill.
Keep reduced macro scope explicit, and reject revised macro snapshots. Forecast
outputs must include source dates, input hashes, calibration counts and evidence
status. Update the existing dashboard instead of creating a second forecast app.

Fixed-rule allocation research: keep config/allocation_research.yaml frozen and
its registered hash checked before CLI analysis. Protocol commit ef17a02 precedes
strategy results. Never move historical_end/prospective_start to consume the
sealed period or interpret retrospective rolling windows as a fresh holdout.
Any new study needs separate registration and no retuning on the existing test
history. No additional ML in this phase. Reuse allocation_history and the existing
ledger; decision rows execute next close. Use GBP risk estimates, same targets
for USD attribution, and explicit assumed costs. Macro snapshots must carry known-
by dates; whole-vintage calendar-month calculations prevent revised-denominator
leakage. Unknown macro is never a benign regime. Regime diagnostics overlap and
are not independent cycles. Zero-hurdle Sharpe is not excess GBP cash Sharpe.
Run `uv run --locked aycapital-allocations --input outputs/scenarios` only on the
frozen historical snapshot; do not replace it with current data. The prospective
252-session period starts 2026-10-15 and remains unscored until separately reviewed.
