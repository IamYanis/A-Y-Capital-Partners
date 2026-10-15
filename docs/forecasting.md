# Version 0.4 — Cross-Asset Forecasting Engine

## Scope and reuse

The forecasting package extends the existing GBP benchmark/scenario dashboard. It
reads its saved GBP prices, FX alignment and provenance rather than downloading a
second price dataset or rebuilding return/risk functions. Feature construction
reuses cross_asset.signals and the FRED vintage-aware available_as_of lookup.
Existing hypothetical shocks, portfolio comparisons and the audited simulation
remain unchanged. Forecast accuracy is not a simulated trading strategy or net P&L.

The new layers are forecasting/config.py (fixed design), features.py (as-of inputs
and targets), engine.py (purged walk-forward models), uncertainty.py (paired circular
moving-block inference), validation.py (metrics and conservative evidence gate),
pipeline.py (exports/report) and dashboard.py (embedded saved-results tab).

## Run

From the repository root, with Python 3.12+ and uv installed:

```bash
uv sync --locked --extra dev --extra dashboard
# Reuse the GBP pipeline; refresh date-stamped inputs before forecasting:
uv run --locked aycapital-scenarios
uv run --locked aycapital-forecast --input outputs/scenarios
uv run --locked --extra dashboard streamlit run scripts/scenario_dashboard.py
```

Open http://localhost:8501 and select **Forecasts & validation**. The existing
historical/scenario tabs remain available. The dashboard checks the price, FX and
metadata hashes and refuses stale forecast artifacts if their inputs changed.
Forecasts always identify their own decision date independently of the historical
slider. Prediction/validation runs occur in the CLI, never implicitly in the UI.

This local checkout already has its environment installed, so equivalent commands
without a globally installed uv are:

```bash
.venv/bin/aycapital-scenarios
.venv/bin/aycapital-forecast --input outputs/scenarios
.venv/bin/streamlit run scripts/scenario_dashboard.py
```

config/forecasting.yaml specifies all settings. They are fixed ex ante for a run;
there is no full-sample grid search. A null training window expands; an integer
rolling window must cover minimum training history plus the maximum label purge.
Changing settings after inspecting results constitutes a new research experiment,
not fresh OOS confirmation on the same history.

## Features and macro availability

Per-asset GBP momentum (5/20/60/120 sessions), 20/60-session volatility, moving-average
trends, downside volatility, trailing drawdown and relative strength are reused from
the existing signal engine. Common features include benchmark trend/volatility,
GBP-per-USD changes computed from already-available aligned FX, quote age, and HYG
versus LQD relative momentum. That last feature is a **price-based credit proxy**,
not a fabricated observed credit spread.

Vintage FRED inputs include 2Y/10Y yields, their spread, high-yield option-adjusted
credit spread, CPI and broad dollar index. Each level/change uses only the vintage
known under the existing conservative next-UTC-day release/expiry convention.
Changes compare known levels 20 sessions apart, not purported inflation YoY. Monthly
macro levels are not backfilled to their observation months. Missing or expired
observations remain missing. Revised/latest descriptive snapshots are rejected.

FRED_API_KEY is read from the process environment. With macro_policy: optional,
unavailable vintage series produce explicit warnings and a reduced feature scope;
no revised data, invented spreads or zero-filled macro features are substituted.
With macro_policy: require, missing required series stop the run. macro_policy: off
explicitly disables macro retrieval. Macro provenance is exported. Configured series
that exist but have gaps remain required feature columns; affected training/test
rows are excluded rather than imputed with future information.

The delivered empirical validation run had no FRED key: its scope is reduced_price_fx.
It tests the price/FX model family, **not the unobserved macro-enabled model family**.
Publication-aware macro feature construction is tested with deterministic vintage
fixtures; a full real-data macro run remains pending credentials and source coverage.

## Targets and purged walk-forward design

For horizon h in 5, 10, 20 and 40 trading sessions, the target at decision t is:

`P_GBP[t+1+h] / P_GBP[t+1] - 1`

The decision occurs after the actual t close, entry is the following session close,
and the outcome matures h sessions after entry. Benchmark excess target is the
arithmetic asset return minus SPY return over that exact interval. Self-outperformance
of SPY against itself is exactly zero and is excluded from the skill gate's relative
classification requirement. Future entry/exit session labels use the same exchange
calendar; unforeseen closures remain a schedule limitation.

At origin t, every training row s must satisfy `s + h + 1 < t`. The extra strict
inequality conservatively excludes even labels ending at today's close. Eligible
feature rows must be finite; required unavailable features do not become zeros.
Rolling windows use only eligible positions inside the configured recency window.
StandardScaler and all models fit inside that training set. Volatility regime
thresholds are the median of eligible training observations, never the full sample.

Test origins are spaced by h sessions. Their return intervals share boundary prices
but no return increments. Earlier test outcomes enter later fitting or residual
calibration only after their actual label endpoints have passed. Prequential updates
using matured old test outcomes are legitimate online learning, not reuse for an
untouched prospective holdout. Future-price/feature/revision perturbation tests verify
invariance of earlier predictions, including intervals and probabilities.

## Models and reference forecasts

- **Historical average:** expanding/rolling matured-label mean plus smoothed
  unconditional positive/outperformance frequencies, recomputed at every origin.
- **Conditional average:** historical means and frequencies within the current
  benchmark trend/volatility state (up/down trend, high/low training-relative vol).
  At least 100 training sessions and 10 nonoverlapping outcome blocks are required
  by default. Sparse states visibly fall back to unconditional estimates, are marked
  unsupported and cannot be interpreted as validated conditional forecasts.
- **Ridge:** fixed-alpha standardized linear regressions for asset and excess returns.
  Its positive/outperformance probabilities come from previously matured OOS return/
  excess residuals, and are absent before enough such residuals exist.
- **Logistic:** fixed-C standardized classifiers for positive return and benchmark
  outperformance. Expected returns mix the training positive/negative outcome means
  using fitted probabilities. Sparse single-class cases use flagged unconditional
  frequencies; they are not described as fitted classifiers with demonstrated skill.

Historical frequencies use Beta(1,1) smoothing. Logistic probabilities are native
model outputs, not post-hoc isotonic/Platt fits on future labels. Calibration curves
and Brier scores evaluate them OOS. Directional accuracy thresholds probabilities
at 0.5 on matched probability observations; return-sign accuracy is separately
reported. Expected simple asset returns are floored at -100% for economic support.

Alternative-regime conditional averages are exported when their past sample supports
estimation. Their regime-specific skill remains explicitly unvalidated. Subgroup
OOS errors are descriptive; no subgroup is promoted to credible based on a tiny
sample or a full-history regime definition.

## Uncertainty, metrics and evidence gate

Intervals use **matured nonoverlapping OOS residuals**, not optimistic training-fit
residuals. Default 80% predictive intervals require 20 earlier residual origins;
conditional-average intervals additionally match the current regime label. Empirical
samples/quantiles are approximate under distributional stability. They include
realized prior prediction errors but do not guarantee distribution-free conditional
coverage or fully quantify parameter uncertainty after structural breaks. No exact
conformal coverage claim is made.

Paired loss improvements use circular moving blocks (default five evaluation origins),
retaining serial dependence even after label overlap is removed. Seeded 4,999 bootstrap
replicates estimate 95% mean-improvement intervals and one-sided centered-null p-values.
Loss/coverage confidence intervals are marginal, not simultaneous; Holm controls
multiplicity for the screen. This approximate inference assumes weak/local stationarity; block length and small
sample sensitivity remain research limitations. It is not an IID bootstrap.

Metrics include return/excess MAE, probability-based directional/outperformance accuracy,
positive/outperformance Brier losses, descriptive five-bin calibration error, interval
coverage/width and bootstrap coverage confidence intervals. Benchmarks are evaluated
on the same probability rows as the corresponding model. Overlap exists in training
labels, but uncertainty and test sample counts never treat daily overlapping labels
as independent OOS outcomes.

The gate requires:

1. At least 40 OOS origins, matched probability outcomes and scored intervals.
2. At least 80% of OOS forecasts with adequate regime/class/training-block support.
3. A coverage-bootstrap lower confidence bound at least nominal coverage minus 10
   percentage points (a fixed conservative screening tolerance, not a coverage theorem).
4. Positive joint improvement in MAE and both relevant Brier endpoints. The maximum
   one-sided endpoint p-value tests the intersection alternative: **all** must improve.
5. Holm family-wise correction across all configured assets × horizons × three
   non-baseline models, including unsupported combinations. No cherry-picked horizon.

A passing flag says **retrospective OOS screen passed; prospective validation still
required**. Otherwise the interface explicitly says **insufficient evidence**, with
sample/coverage/baseline reasons. The historical-average reference never makes a
skill claim. Synthetic runs cannot establish market skill even if their scores are
excellent. Unit loss fixtures test that the gate can pass strong improvements and
reject a tied baseline; they never enter the real validation report.

## Validation report and artifacts

Every run writes forecasts/validation_report.md and full CSVs inside its input folder:
latest_forecasts, oos_predictions, validation, calibration, regime_validation,
conditional_regime_forecasts and predictive_samples. Metadata records configuration,
input SHA-256 hashes, macro provenance, feature names, source dates and limitations.
Missing probabilities/intervals are empty, not fabricated numbers.

The committed report and per-asset results in docs/validation/version_0_4/ reproduce
the delivered 14-asset, 1,508-session GBP snapshot through 2026-10-14. Under the fixed
joint screen, **none of the 168 non-baseline asset/horizon/model combinations show
credible OOS skill**. The 56 historical-average rows are reference forecasts. At
20/40 sessions, scored interval/probability sample sizes are inadequate. At shorter
horizons, failed joint baseline improvements or uncertain interval coverage prevent
claims. Do not select isolated attractive point estimates as reliable forecasts.

## Continuing limitations and next steps

The saved price dataset is a current vendor vintage, not a point-in-time corporate-
action database; the static proxy universe has selection/survivorship bias. GBP
valuations inherit the documented asynchronous dated-FX policy and bounded holiday
quote reuse. Macro release times are conservative date-level conventions, not audited
intraday publication timestamps. Regime changes can invalidate residual distributions.

The project already inspected much of this history in prior audits; the delivered
backtest is retrospective OOS rather than a fresh prospective test. No executable
portfolio policy, costs-based P&L or validated scenario calibration is inferred.
All assets remain research proxies with unverified UMushroom availability/eligibility.

Next steps: obtain vintage macro credentials/coverage, verify synchronous FX and
point-in-time universe/prices, preregister parameters and a future cutoff, collect
more long-horizon evidence, test block/calibration-window sensitivity on training
research data, and evaluate a frozen forecast/decision policy prospectively.

## Primary references

- [scikit-learn temporal splits and gap](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [Training-only Pipeline transformations](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [Probability calibration and proper scores](https://scikit-learn.org/stable/modules/calibration.html)
- [FRED real-time periods](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html)

## Delivered checks

90 tests passed with the dashboard extra, Ruff and whitespace checks passed, and
the original demo plus live GBP/forecast dashboard checks succeeded. Tests cover
strict label purging, future feature/price invariance, vintage revisions, unavailable
FX, matured residuals, sparse samples, baseline rejection, synthetic no-skill gates,
regime support and stale-snapshot detection. Existing dependency deprecation warnings
remain visible. CI also exercises the forecast CLI with a deliberately small synthetic
smoke configuration; that configuration is not the committed market experiment.
