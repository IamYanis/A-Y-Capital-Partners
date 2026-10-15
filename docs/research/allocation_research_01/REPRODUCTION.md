# Reproduce allocation research 01

The protocol was committed as `ef17a02` before strategy calculations. It fixes
all rules, costs, horizons, minimum samples and the multiple-comparison family.
The study uses 1,508 actual adjusted-price/FX observations, 2020-10-13 through
2026-10-14. After the common 200-price warmup, 1,308 sessions are evaluated from
2021-07-30. This history was already inspected in v0.4; the sequential calculations
are causal, but the study is retrospective, not newly untouched validation.

The newly reserved period starts 2026-10-15. Its 252nd scheduled XNYS session is
2027-10-15. Nothing in this change reads or scores that period. The fixed plan
hash and historical boundary are enforced; changing a YAML date cannot unlock it.
A separate reviewed evaluation change after maturity is required. At least 40
20/40-day blocks would require a longer confirmation history, so one year alone
cannot establish the registered level of evidence.

## Offline reproduction

The published input files are gzip copies of the exact existing v0.4 GBP
snapshot, not another provider download. Their decompressed hashes must match
`metadata.json` → `input_sha256`. `macro_cache.zip` contains the immutable public
ALFRED cache snapshots and pointers used for this run. Its original snapshot
hashes are recorded under `macro_provenance`. No FRED key or network is needed
once these files are restored.

From the repository root:

```bash
uv sync --locked --extra dev --extra dashboard
mkdir -p outputs/allocation_frozen data/cache
python3 -c 'import gzip,pathlib; src=pathlib.Path("docs/research/allocation_research_01/input"); dst=pathlib.Path("outputs/allocation_frozen"); [(dst/p.stem).write_bytes(gzip.decompress(p.read_bytes())) for p in src.glob("*.gz")]'
python3 -m zipfile -e docs/research/allocation_research_01/macro_cache.zip data/cache
uv run --locked aycapital-allocations --input outputs/allocation_frozen --output outputs/allocation_reproduced
uv run --locked --extra dashboard streamlit run scripts/scenario_dashboard.py
```

For the existing full scenario dashboard, select the original `outputs/scenarios`
input directory and the saved allocation directory `outputs/allocation_reproduced`
in its Allocation research tab. The published minimal frozen input supports the
research CLI; it deliberately omits old dashboard scenario charts and settings.
To inspect the published allocation report inside the existing dashboard, set
its allocation research directory to `docs/research/allocation_research_01`, with
the matching original scenario input selected. Large evidence CSVs are gzip
compressed; pandas and the dashboard read them directly.

`macro_snapshots.csv.gz` is the aligned provider snapshot table; `macro_features`
retains each observation period, known-by date and conservative availability.
The final UNRATE request for 2026-10-14 returned an archived 2026-10-13 identity;
this earlier state is delayed to the requested known-by date, never advanced.
Month-end snapshots are deliberately slower than daily release-date retrieval.
ALFRED describes historical vintages and its release dating conventions in its
[official help](https://alfred.stlouisfed.org/help). Do not replace these data with
current `fredgraph.csv` and an assumed publication lag.

No macro retrieval failed. After warmup there are zero qualifying slowdown days
under the frozen +0.5 percentage-point three-month unemployment threshold. The
common sample and warmup exclude the initial COVID shock. Zero coverage is not
proof of strategy robustness during slowdowns. Correction/recovery episode counts
also reflect threshold crossings; they are not counts of independent market cycles.

## Interpretation

At 10 bps per traded side, 40-day equal-weight has 9.17% CAGR, 12.21% volatility,
-12.63% maximum drawdown and 0.780 zero-hurdle Sharpe. Trend following has 8.28%
CAGR, 10.37% volatility, -13.00% drawdown and 0.819 Sharpe, but 2.49× annual
buy-plus-sell turnover versus 0.44×. These are sample estimates, not forecasts.

The paired 40-session-block 95% interval for the trend-minus-equal-weight Sharpe
difference is [-0.322, 0.424] at 10 bps; its annual arithmetic return-difference
interval is [-4.76%, +2.82%]. At 25 bps their Sharpes are 0.783 and 0.774; at
50 bps they are 0.722 and 0.765. These cost assumptions are not optimized and
are not measured trading quotes. None of the six candidate/horizon combinations
passes the joint benchmark/cost/block-length gate or the Holm correction.

Inverse-volatility, risk parity and defensive allocations reduce sample volatility
and often drawdown, with lower growth and no established risk-adjusted dominance.
Their suitability depends on an independently chosen GBP risk budget, not a claim
of predictive advantage. SGOV is unhedged USD short bills. Identical-target USD
comparison shows material currency effects; it does not isolate hedged returns.
The original asynchronous FX and Yahoo adjustment limitations still apply.

Keep the preregistered equal-weight/40-session reference for research, not because
it won a parameter search. Before a real competition allocation, resolve eligible
instruments, duplicate proxy exposures, weight limits, GBP cash and risk tolerance.
A genuinely broader regime study needs a separately registered point-in-time
universe with longer histories; do not silently replace SGOV or extend today’s
surviving ETF list backwards to manufacture recession evidence.

## Files and checks

`performance`, `uncertainty`, `evidence`, `regimes` and `holding_returns` contain
all strategy/cost/horizon comparisons. `daily_returns`, `turnover`, `trades` and
`rolling` provide the account ledger and overlapping 252-session diagnostics.
Rolling windows and overlapping regime buckets are descriptive, not independent
tests. Bootstrap inference is approximate and assumes local stationarity.

Verify published artifacts against `sha256.json`; hashes exclude that manifest.
Run `uv run --locked --extra dashboard pytest -q` and `uv run --locked ruff check .`.
No strategies were tuned or added in response to these results; no new ML models,
no trades, no merge, and no prospective holdout evaluation.
