# Version 0.1 methodology

Simple returns are P[t]/P[t-1]-1 and log returns are log(P[t]/P[t-1]).
Only simple returns enter cumulative wealth, statistics and covariance functions.
Wealth starts at one; drawdown is wealth divided by its running high minus one,
including the initial unit wealth. Maximum drawdown is the most negative drawdown.
Volatility scales sample standard deviation by sqrt(periods per year).
Covariance scales per-period estimates by periods per year.

GBP prices equal native prices times GBP per native quote unit at matching dates.
FX therefore affects returns as well as exposure. Constant unit conversion is not
an FX time series. Yahoo auto-adjustment is explicitly requested and still requires
independent validation for research involving dividends and corporate actions.

Optimizer inputs must have matching asset labels and a positive semidefinite matrix.
Weights sum to one and obey position, eligibility and named group limits. Numerical
feasibility is checked within 1e-7; do not independently round weights for execution.
No rounding, position sizing or execution is implemented in this milestone.

Synthetic demo artifacts illustrate software behavior only. Real-data artifacts
are historical descriptions, not evidence of predictive or out-of-sample success.
