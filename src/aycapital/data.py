"""Adjusted Yahoo prices, explicit currency conversion, and auditable CSV snapshots."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from aycapital.validation import frame


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    prices = frame(prices)
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Prices require a DatetimeIndex")
    if prices.index.hasnans or (prices <= 0).any().any():
        raise ValueError("Invalid dates or nonpositive prices")
    return prices


def load_yahoo(tickers: list[str], start: str, end: str,
               cache_dir: str | Path = "data/cache", *, refresh: bool = False) -> pd.DataFrame:
    """End is exclusive. Strict common-calendar data; never forward-fill gaps.

    Yahoo adjusted prices are vendor estimates, not verified total-return indices.
    Cache preserves retrieval time and content digest; refresh can revise history.
    """
    if not tickers or len(set(tickers)) != len(tickers):
        raise ValueError("Supply unique tickers")
    if pd.Timestamp(start) >= pd.Timestamp(end):
        raise ValueError("start must precede end")
    request = {"tickers": tickers, "start": start, "end": end, "auto_adjust": True,
               "provider": "yahoo", "schema": 1}
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:20]
    directory = Path(cache_dir)
    csv = directory / f"{key}.csv"
    meta = directory / f"{key}.json"
    if csv.exists() and meta.exists() and not refresh:
        metadata = json.loads(meta.read_text())
        if (metadata["request"] != request or metadata["sha256"]
                != hashlib.sha256(csv.read_bytes()).hexdigest()):
            raise ValueError("Cache integrity check failed")
        prices = pd.read_csv(csv, index_col=0, parse_dates=True)
    else:
        downloaded = yf.download(tickers, start=start, end=end, auto_adjust=True,
                                 progress=False, threads=False, group_by="column")
        if downloaded.empty:
            raise ValueError("Yahoo returned no data")
        if isinstance(downloaded.columns, pd.MultiIndex):
            prices = downloaded["Close"]
        elif len(tickers) == 1:
            prices = downloaded[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            raise ValueError("Unexpected provider column format")
        if not set(tickers).issubset(prices.columns):
            raise ValueError("Provider omitted requested assets")
        prices = validate_prices(prices.loc[:, tickers])
        directory.mkdir(parents=True, exist_ok=True)
        prices.to_csv(csv, float_format="%.17g")
        metadata = {"request": request,
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                    "sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
                    "currency": "native quote units; user verification required"}
        meta.write_text(json.dumps(metadata, indent=2) + "\n")
    prices = validate_prices(prices)
    if list(prices.columns) != tickers:
        raise ValueError("Cache assets do not match request")
    if (prices.index < pd.Timestamp(start)).any() or (prices.index >= pd.Timestamp(end)).any():
        raise ValueError("Provider data outside requested dates")
    prices.attrs["source"] = metadata
    return prices


def to_gbp(prices: pd.DataFrame, gbp_per_quote_unit: pd.DataFrame) -> pd.DataFrame:
    """Multiply each asset by synchronous GBP per quote unit rates.

    Caller verifies currency/units (GBp requires 0.01 GBP), timestamps and FX quality.
    No FX inference, forward filling, or implicit conversion is performed.
    """
    prices = validate_prices(prices)
    rates = validate_prices(gbp_per_quote_unit)
    if not rates.index.equals(prices.index) or not rates.columns.equals(prices.columns):
        raise ValueError("FX rates must exactly match price dates and assets")
    return validate_prices(prices * rates)


def synthetic_prices(tickers: list[str], observations: int = 505, seed: int = 2026):
    """Clearly synthetic demonstration data, never a historical market record."""
    import numpy as np

    if not tickers or len(set(tickers)) != len(tickers) or observations < 3:
        raise ValueError("Unique assets and at least three observations required")
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.004, (observations - 1, 1))
    innovations = rng.normal(0.0002, 0.008, (observations - 1, len(tickers))) + common
    values = np.vstack([np.ones(len(tickers)), np.exp(innovations).cumprod(axis=0)]) * 100
    return pd.DataFrame(values, index=pd.bdate_range("2022-01-03", periods=observations),
                        columns=tickers)
