# Fixed-rule allocation research

**Retrospective historical evidence, not forecasts or prospective validation.**

Evaluation: 2021-07-30 to 2026-10-14; source: yahoo; valuation currency GBP.

| Hold | Strategy | CAGR | Volatility | Max drawdown | Zero-hurdle Sharpe | Annual traded notional | GBP minus USD total return |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20 | equal_weight | 9.11% | 12.21% | -12.71% | 0.775 | 0.53× | 7.88% |
| 20 | inverse_volatility | 6.83% | 10.78% | -11.05% | 0.667 | 1.02× | 7.06% |
| 20 | risk_parity | 7.42% | 10.60% | -11.37% | 0.729 | 1.24× | 7.27% |
| 20 | trend_following | 7.58% | 10.28% | -14.31% | 0.762 | 3.61× | 7.32% |
| 20 | defensive | 7.37% | 10.15% | -10.72% | 0.751 | 0.42× | 7.25% |
| 20 | SPY | 14.06% | 18.72% | -22.16% | 0.796 | 0.19× | 9.92% |
| 40 | equal_weight | 9.17% | 12.21% | -12.63% | 0.780 | 0.44× | 7.90% |
| 40 | inverse_volatility | 6.95% | 10.79% | -10.98% | 0.677 | 0.82× | 7.10% |
| 40 | risk_parity | 7.44% | 10.64% | -11.25% | 0.728 | 0.98× | 7.28% |
| 40 | trend_following | 8.28% | 10.37% | -13.00% | 0.819 | 2.49× | 7.58% |
| 40 | defensive | 7.36% | 10.16% | -10.77% | 0.750 | 0.35× | 7.25% |
| 40 | SPY | 14.06% | 18.72% | -22.16% | 0.796 | 0.19× | 9.92% |

10 bps per traded side shown; 25/50 bps are in performance.csv. Turnover is buys plus sells, including initial cash entry. GBP minus USD is a total-return percentage-point difference with identical targets, not an additive FX P&L decomposition or hedged return.

## Prespecified evidence gate

| Hold | Candidate | Complete blocks | Worst Sharpe CI lower | Holm p | Historical gate |
|---:|---|---:|---:|---:|---|
| 20 | inverse_volatility | 65 | -0.367 | 1.000 | False |
| 20 | risk_parity | 65 | -0.326 | 1.000 | False |
| 20 | trend_following | 65 | -0.462 | 1.000 | False |
| 40 | inverse_volatility | 32 | -0.364 | 1.000 | False |
| 40 | risk_parity | 32 | -0.321 | 1.000 | False |
| 40 | trend_following | 32 | -0.380 | 1.000 | False |

0 of six candidate/horizon combinations pass the historical gate. Zero have prospective confirmation. The gate requires improvement over both equal-weight and defensive benchmarks at 10 AND 25 bps, under both 40- and 80-session block bootstraps, positive lower 95% intervals, Holm family correction and at least 40 full holding blocks. See uncertainty.csv for paired return-difference intervals. These approximate tests assume local stationarity and cannot establish robustness to unseen structural breaks.

## Decision

No tested dynamic allocation demonstrates a robust risk-adjusted advantage under the registered gate. Retain equal-weight with 40-session rebalancing as the simplest tested research default, chosen before results. Do not promote a point estimate or a favorable regime to predictive skill.

## Regime coverage and limitations

Regime flags use previous-close information, including conservative macro availability. Buckets overlap; counts are sessions and episodes, not independent economic cycles. Scattered regime days are not compounded as a tradable NAV.

| Regime | Sessions | Episodes | Evidence |
|---|---:|---:|---|
| equity_correction | 115 | 19 | descriptive_only |
| inflation_shock | 227 | 7 | descriptive_only |
| economic_slowdown | 0 | 0 | insufficient_evidence |
| recovery | 725 | 50 | descriptive_only |

### Strategy behavior within regimes

Mean daily excess returns below are basis points versus equal-weight, at 10 bps costs. They are descriptive associations, not forecasts or independent regime tests. Zero-session buckets cannot support conclusions.

| Hold | Regime | Strategy | Excess bps/day | Daily-risk annualized |
|---:|---|---|---:|---:|
| 20 | equity_correction | inverse_volatility | -1.81 | 15.03% |
| 20 | equity_correction | risk_parity | -2.81 | 14.91% |
| 20 | equity_correction | trend_following | -4.90 | 12.98% |
| 20 | equity_correction | defensive | -0.91 | 12.11% |
| 20 | inflation_shock | inverse_volatility | -0.48 | 10.61% |
| 20 | inflation_shock | risk_parity | 0.55 | 10.52% |
| 20 | inflation_shock | trend_following | 2.67 | 9.71% |
| 20 | inflation_shock | defensive | -0.57 | 9.67% |
| 20 | recovery | inverse_volatility | -0.55 | 9.32% |
| 20 | recovery | risk_parity | -0.28 | 9.13% |
| 20 | recovery | trend_following | -0.64 | 9.15% |
| 20 | recovery | defensive | 0.36 | 8.96% |
| 40 | equity_correction | inverse_volatility | -0.82 | 15.07% |
| 40 | equity_correction | risk_parity | -2.07 | 15.10% |
| 40 | equity_correction | trend_following | -3.99 | 13.22% |
| 40 | equity_correction | defensive | -0.81 | 12.14% |
| 40 | inflation_shock | inverse_volatility | -0.38 | 10.62% |
| 40 | inflation_shock | risk_parity | 0.47 | 10.62% |
| 40 | inflation_shock | trend_following | 2.64 | 9.91% |
| 40 | inflation_shock | defensive | -0.64 | 9.66% |
| 40 | recovery | inverse_volatility | -0.59 | 9.32% |
| 40 | recovery | risk_parity | -0.30 | 9.15% |
| 40 | recovery | trend_following | -0.42 | 9.24% |
| 40 | recovery | defensive | 0.35 | 8.97% |

Macro warnings:
- No vintage retrieval failures. See macro_features.csv for actual observation, knowledge and availability dates; unavailable states remain missing.

Data/method limitations:
- GBP valuation uses latest already-available dated FX quotes, not synchronized equity/FX closes; daily FX dates are conservatively usable the next UTC day.
- Bounded backward FX alignment can span weekends/holidays; source dates are exported.
- FX quote age exceeds four calendar days on 2 session(s); maximum observed age is 5.83 days. This may distort synchronized-currency risk estimates.
- Vendor adjusted prices approximate total returns and can be revised.
- Static proxy universe; common history excludes unavailable leading observations.
- Historical portfolio comparisons are ex-post illustrations, not OOS validation.
- UMushroom instrument availability, eligibility and rules remain unverified.
- Provider returned 1 FX date(s) outside the requested interval; raw quotes retained and explicitly excluded from conversion.
- Provider FX contains 6 missing quote dates; raw cache retains them, and matching uses only valid dated quotes.
- No fresh holdout has been evaluated; all results are retrospective.
- Cost assumptions are unverified; no measured spread, impact or capacity.
- No taxes, securities lending, custody, FX entry/exit fees or terminal liquidation.
- Zero-hurdle Sharpe is not excess performance over investable GBP cash.
- Only one short common-history sample; regimes are overlapping diagnostics.
- Monthly vintage snapshots delay availability; not exact initial release times.
- Current-vendor adjusted histories are not point-in-time corporate-action data.

The prospective holdout starts 2026-10-15 and remains unscored. Its first 252 sessions are sealed; even that provides too few independent 20/40-day blocks for strong confirmation. Do not tune on interim results. A future explicit reviewed change is required to unlock evaluation. See the frozen allocation_plan.md and configuration commit ef17a02.

Macro source: [ALFRED historical-vintage help](https://alfred.stlouisfed.org/help). This study uses delayed month-end knowledge states, not observation-date alignment or guessed initial publication lags. Inflation and unemployment changes compare observations within the same vintage. NBER hindsight recession dates are not used.

Competition eligibility is unverified. None of these proxy allocations is an executable UMushroom recommendation. SGOV retains USD/GBP exposure; equal ETF weights also overweight overlapping equity/credit exposures.
