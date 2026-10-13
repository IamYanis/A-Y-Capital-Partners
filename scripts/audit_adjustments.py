import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from aycapital.cross_asset.config import ResearchConfig
from aycapital.cross_asset.market import load_market

config = ResearchConfig.load("config/cross_asset.yaml")
panel = load_market(config, "2020-10-13", "2026-10-13")
raw = yf.download(
    list(config.assets),
    start="2020-10-13",
    end="2026-10-13",
    auto_adjust=False,
    progress=False,
    threads=False,
    group_by="column",
)
if raw is None or raw.empty or "Adj Close" not in raw:
    raise ValueError("Unadjusted provider response unavailable; no substitution")
records = []
for ticker in config.assets:
    adjusted = raw["Adj Close"][ticker].reindex(panel.prices.index)
    close = raw["Close"][ticker].reindex(panel.prices.index)
    if adjusted.isna().any() or close.isna().any():
        raise ValueError(f"Raw comparison incomplete: {ticker}")
    difference = np.abs(adjusted.to_numpy() / panel.prices[ticker].to_numpy() - 1)
    records.append(
        {
            "ticker": ticker,
            "max_relative_adjusted_difference": float(difference.max()),
            "adjusted_differs_from_raw_close": bool(not np.allclose(adjusted, close)),
            "max_absolute_daily_return": float(panel.prices[ticker].pct_change().abs().max()),
            "missing_prices": int(panel.prices[ticker].isna().sum()),
            "observations": len(adjusted),
        }
    )
output = Path("docs/audit/version_0_2")
pd.DataFrame(records).to_csv(output / "adjustment_crosscheck.csv", index=False)
(output / "adjustment_crosscheck_metadata.json").write_text(
    json.dumps(
        {
            "provider": "yahoo",
            "comparison": "Cached auto_adjust=True Close versus new auto_adjust=False Adj Close",
            "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "limitation": (
                "Same vendor consistency check; not independent total-return "
                "or historical-vintage verification"
            ),
        },
        indent=2,
    )
)
print(pd.DataFrame(records).to_string(index=False))
