# Allocation research 01: frozen protocol

Registered before computing allocation results. Existing v0.4 market history has
already been inspected: all data through 2026-10-14 is retrospective evidence,
not a newly untouched test set. No new forecasting or machine-learning models.

Compare equal-weight, inverse 60-session GBP volatility, equal-risk-contribution
60-session Ledoit–Wolf GBP covariance, and equal-weight trend following (GBP
price above its trailing 200-session average; inactive asset weights move to SGOV).
Use the existing defensive illustration: 40% SGOV, 25% IEF, 20% GLD, 15% SPY.
All 14 existing research proxies remain fixed; no universe selection on results.
These overlap in exposures and are not confirmed UMushroom instruments.

All strategies use the same first decision after 200 prices and next-close entry.
Rebalance every 20 or 40 sessions; no parameter search. Risk estimates use only
trailing data available at decision close. Use the existing self-financing ledger;
fully invested long-only, fractional units, no leverage, no tax, no terminal sale.
SPY buy-and-hold is an additional reference. SGOV is USD bills, not GBP cash.

Per traded-notional costs: 10, 25 and 50 bps. These are explicit execution
assumptions covering fees, spreads and slippage, not observed quotes or verified
competition fees. The whole robustness gate requires both 10 and 25 bps; 50 is a
severe descriptive stress. No hypothetical return assumptions enter performance.
GBP and native-USD comparison must use exactly the same GBP-selected targets;
USD results are currency attribution, not a hedged investable strategy.

Historical rolling tests are sequential out-of-information calculations with no
fit on future returns, but cannot remove researcher familiarity with this period.
Report rolling 252-session performance and disjoint holding-period returns;
rolling/overlapping windows are descriptive, not independent replications.

Primary risk-adjusted endpoint: annualized net return / daily volatility (zero
hurdle; not GBP cash excess Sharpe). Paired circular moving-block bootstrap
recomputes this statistic, with blocks of 40 AND 80 sessions. Candidate superiority
requires positive lower 95% CI against BOTH equal-weight and defensive benchmarks
at BOTH 10 and 25 bps, using BOTH block lengths; take the largest centered-null
bootstrap p across these comparisons. Holm correction covers the six candidate
rules × horizons (inverse-volatility, risk-parity, trend × 20/40). Require at least
40 complete disjoint holding blocks. Bootstrap stationarity is approximate;
structural breaks can invalidate inference. Report return-difference uncertainty
as well. Do not treat historical significance as prospective confirmation.

Regime diagnostics are prespecified and may overlap: inflation shock = release-
known CPI YoY >=4% and +0.5 percentage points versus three observation months ago;
slowdown = release-known unemployment +0.5 pp in three observation months;
equity correction = GBP SPY <=90% of trailing 252-session high;
recovery = positive trailing 60-session SPY return and above 200-session mean,
excluding correction and inflation/slowdown flags. No regime claims with fewer
than 60 sessions; even larger buckets are not independent economic episodes.
No NBER recession dates are used as tradable real-time information.

Macro: public ALFRED historical snapshots if accessible, or keyed FRED vintages.
Month-end vintage snapshots may only become usable the following UTC midnight.
They are deliberately delayed knowledge states, NOT exact initial release dates.
Compute CPI changes from values in the SAME available vintage, never the latest
revised history. Record observation period, vintage/known-by date and availability.
Do not fill macro retrieval failures with current/revised data; label gaps and
withhold unsupported regime conclusions. Preserve original cached snapshots.

Prospective confirmation: first 252 exchange sessions from 2026-10-15 are sealed.
Default historical commands reject input containing later dates and never score
this holdout. Future evaluation requires a separately reviewed, explicit unlocking
change after all 252 closes have occurred, using this registered protocol and
its commit. No interim inspection or tuning. 252 sessions alone still cannot
satisfy the historical 40-block gate at 20/40 days; extend confirmation before
strong claims. No current result can claim credible prospective superiority.

If evidence is inadequate, retain the simplest tested research reference:
equal-weight at 40-session intervals (lower intervention frequency, chosen now,
not selected on subsequent return). This is a research default, not a competition
allocation; eligible instruments, overlapping exposures, GBP risk tolerance and
cash needs must be resolved separately before allocation recommendations.
