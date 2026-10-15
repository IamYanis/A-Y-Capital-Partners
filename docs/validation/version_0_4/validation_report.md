# Version 0.4 forecast-validation report

Decision snapshot: **2026-10-14**. Source: **yahoo**.
Feature scope: **reduced_price_fx**.

## Credible skill assessment

Retrospective research only; no prospective validation or investment recommendation.
Design settings were fixed for this run, but the history has already been inspected in earlier project work. A fresh future holdout is still required.

**No asset-class/horizon/model combination demonstrated credible skill under the configured joint, multiplicity-adjusted screen.**

## Evidence by horizon and model

| Horizon | Model | Asset combinations | Adequate OOS | Screen passed |
|---:|---|---:|---:|---:|
| 5 | conditional_average | 14 | 14 | 0 |
| 5 | historical_average | 14 | 14 | 0 |
| 5 | logistic | 14 | 14 | 0 |
| 5 | ridge | 14 | 14 | 0 |
| 10 | conditional_average | 14 | 14 | 0 |
| 10 | historical_average | 14 | 14 | 0 |
| 10 | logistic | 14 | 14 | 0 |
| 10 | ridge | 14 | 14 | 0 |
| 20 | conditional_average | 14 | 0 | 0 |
| 20 | historical_average | 14 | 0 | 0 |
| 20 | logistic | 14 | 0 | 0 |
| 20 | ridge | 14 | 0 | 0 |
| 40 | conditional_average | 14 | 0 | 0 |
| 40 | historical_average | 14 | 0 | 0 |
| 40 | logistic | 14 | 0 | 0 |
| 40 | ridge | 14 | 0 | 0 |

## Design and safeguards

- Targets: GBP simple returns from next-session-close entry to exit after h sessions.
- Purged expanding/rolling fits: label endpoints precede the decision session.
- Scalers, models and regime volatility thresholds fit on eligible training rows only.
- OOS origins are spaced h sessions apart; label return intervals do not overlap.
- Models: regime-conditional historical averages, fixed Ridge and logistic regressions.
- Mean/frequency baselines use the same matured training set.
- Sparse regime/class support triggers explicit fallback and insufficient evidence flags.
- Intervals use previously matured, nonoverlapping OOS residuals only.
- Approximate residual intervals assume stability; coverage is not guaranteed.
- Paired moving-block bootstrap retains serial dependence across evaluation origins. MAE and both Brier losses must improve; self-outperformance is exactly zero.
- Holm correction includes all configured asset/horizon/model candidates.
- Regime subgroup metrics are descriptive, with no separate skill claim.
- Calibration tables, MAE, directional accuracy, Brier losses, benchmark-excess MAE, interval coverage/width and paired confidence intervals are exported in CSV.
- Forecast accuracy is not strategy P&L; a trading policy needs separate validation.

## Limitations and unavailable inputs

- GBP valuation uses latest already-available dated FX quotes, not synchronized equity/FX closes; daily FX dates are conservatively usable the next UTC day.
- Bounded backward FX alignment can span weekends/holidays; source dates are exported.
- FX quote age exceeds four calendar days on 2 session(s); maximum observed age is 5.83 days. This may distort synchronized-currency risk estimates.
- Vendor adjusted prices approximate total returns and can be revised.
- Static proxy universe; common history excludes unavailable leading observations.
- Historical portfolio comparisons are ex-post illustrations, not OOS validation.
- Scenario shocks are manually chosen assumptions, not forecasts or calibrated estimates.
- UMushroom instrument availability, eligibility and rules remain unverified.
- Provider returned 1 FX date(s) outside the requested interval; raw quotes retained and explicitly excluded from conversion.
- Provider FX contains 6 missing quote dates; raw cache retains them, and matching uses only valid dated quotes.
- DGS2: vintage retrieval unavailable; not replaced with revised history.
- DGS10: vintage retrieval unavailable; not replaced with revised history.
- BAMLH0A0HYM2: vintage retrieval unavailable; not replaced with revised history.
- CPIAUCSL: vintage retrieval unavailable; not replaced with revised history.
- DTWEXBGS: vintage retrieval unavailable; not replaced with revised history.
- Unavailable publication-aware macro/credit series: BAMLH0A0HYM2, CPIAUCSL, DGS10, DGS2, DTWEXBGS
- Current vendor history is not a point-in-time corporate-action dataset.
- GBP targets inherit the disclosed asynchronous FX valuation convention.
- Prior audits inspected this history; a fresh prospective holdout is still needed.
- Bootstrap inference and residual intervals can fail under changing regimes.
- No trading, scenario probability or competition-eligibility claim is made.

Full per-asset/horizon/model conclusions: validation.csv. Regime diagnostics: regime_validation.csv. Calibration: calibration.csv.
Research estimates: latest_forecasts.csv. Predictions: oos_predictions.csv.
Input hashes, source provenance, design configuration and feature names: metadata.json.

## Reproduce

```bash
uv run --locked aycapital-scenarios --start 2020-10-13 --end 2026-10-15
uv run --locked aycapital-forecast --input outputs/scenarios
```

A different snapshot or changed configuration is a different experiment. After tuning on these results, the same OOS outcomes are no longer untouched.

Committed evidence includes validation.csv, regime_validation.csv and metadata.json. Full chronological predictions, samples and latest estimates remain generated artifacts in outputs/scenarios/forecasts.

## Software checks

90 automated tests passed with the dashboard extra. Ruff and whitespace checks passed.
The original demonstration and a live dashboard check of 40-session Ridge forecasts,
probability/interval plots and evidence warnings passed. Full macro-enabled empirical
validation remains unavailable without FRED credentials.
