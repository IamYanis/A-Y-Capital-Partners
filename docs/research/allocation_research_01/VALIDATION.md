# Validation results

- Full suite: **103 passed** in 31.42 seconds, with the existing NumPy/calendar
  deprecation warnings; no test failures. No market downloads in unit tests.
- Ruff: all checks passed. `git diff --check`: clean.
- Locked environment: `uv sync --locked --extra dev --extra dashboard` succeeded;
  no new dependencies or machine-learning models.
- Original demo, synthetic cross-asset research, synthetic scenario CLI and
  forecasting CLI smoke runs all exited successfully.
- New study: actual cached Yahoo adjusted prices and recorded GBP FX alignment,
  1,308 evaluated sessions. CPI and unemployment each have 73 known-by snapshots;
  all 14 provider cache batches retrieved or restored without macro warnings.
- Offline restoration: decompressed frozen input files and extracted macro cache
  into separate directories, ran the CLI without network, and verified identical
  performance, uncertainty, evidence, macro features, regime and rolling tables.
- Dashboard: Streamlit AppTest against actual saved-market outputs passed;
  changed holding period, cost stress and rolling drawdown controls successfully.
  Automated tests also reject changed source snapshots.
- Published artifact SHA-256 manifest verified before submission.

New regression tests cover diagonal risk budgets, future-price perturbations at
both holding periods, next-close entry, shared-ledger equivalence, transaction
cost validation, vintage/revision timing and same-vintage denominators, public
provider identity and caching, unknown macro regimes, sealed-period protection,
FX availability and constant-native-price FX returns, paired bootstrap identity,
small-sample evidence rejection, and saved dashboard controls.

These software checks establish implementation behavior, not predictive skill.
Zero of six candidate/horizon comparisons passed the registered historical gate.
The prospective evaluation remains sealed and entirely unscored.
