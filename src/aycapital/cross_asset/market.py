"""Provider-independent adjusted daily market panels on real exchange sessions."""

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import yfinance as yf

from aycapital.cross_asset.cache import SnapshotCache
from aycapital.cross_asset.config import ResearchConfig
from aycapital.cross_asset.logging import event
from aycapital.data import validate_prices


@dataclass(frozen=True)
class MarketPanel:
    prices: pd.DataFrame
    closes_utc: pd.Series
    metadata: dict[str, Any]


class MarketProvider(Protocol):
    name: str

    def download(self, ticker: str, start: str, end: str) -> pd.Series:
        """Adjusted close in native quote units; daily session labels."""
        ...


class YahooProvider:
    name = "yahoo"

    def download(self, ticker: str, start: str, end: str) -> pd.Series:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
            keepna=True,
            group_by="column",
        )
        if raw is None or raw.empty:
            raise ValueError("Provider returned no observations")
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close[ticker]
        return close.rename(ticker)


def session_schedule(calendar: str, start: str, end: str) -> pd.Series:
    """Daily session labels with timezone-aware actual close times; end exclusive."""
    if pd.Timestamp(start) >= pd.Timestamp(end):
        raise ValueError("start must precede end")
    cal = xcals.get_calendar(calendar, start=start, end=end)
    schedule = cal.schedule.loc[start:end]
    labels = pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()
    closes = pd.Series(
        pd.to_datetime(schedule["close"].to_numpy(), utc=True), index=labels, name="close_utc"
    )
    return closes.loc[closes.index < pd.Timestamp(end)]


def validate_panel(panel: MarketPanel) -> MarketPanel:
    prices = validate_prices(panel.prices)
    if prices.index.tz is not None or not (prices.index == prices.index.normalize()).all():
        raise ValueError("Market prices require timezone-naive midnight session labels")
    closes = panel.closes_utc
    if not closes.index.equals(prices.index) or closes.isna().any():
        raise ValueError("Close timestamps must exactly align with prices")
    if not isinstance(closes.dtype, pd.DatetimeTZDtype) or str(closes.dt.tz) != "UTC":
        raise ValueError("Actual session close times must be timezone-aware UTC")
    if not closes.is_monotonic_increasing:
        raise ValueError("Session closes must increase")
    calendar = panel.metadata.get("calendar")
    if calendar is None:
        raise ValueError("Panel must declare its exchange calendar")
    expected = session_schedule(
        calendar,
        prices.index[0].strftime("%Y-%m-%d"),
        (prices.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    if not expected.equals(closes):
        raise ValueError("Missing exchange sessions or incorrect session close timestamps")
    return MarketPanel(prices, closes, panel.metadata)


def load_market(
    config: ResearchConfig,
    start: str,
    end: str,
    provider: MarketProvider | None = None,
    *,
    refresh: bool = False,
    now: pd.Timestamp | None = None,
) -> MarketPanel:
    provider = provider or YahooProvider()
    cache = SnapshotCache(config.cache_dir / "market")
    schedule = session_schedule(config.calendar, start, end)
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    if now.tzinfo is None:
        raise ValueError("now must be timezone aware")
    schedule = schedule.loc[schedule < now]
    if len(schedule) < 201:
        raise ValueError("Need at least 201 completed sessions for 200-day trend warmup")
    series = []
    provenance = {}
    for ticker in config.assets:
        request = {
            "provider": provider.name,
            "ticker": ticker,
            "start": start,
            "end": end,
            "adjusted": True,
            "schema": 1,
        }
        try:
            cached = None if refresh else cache.read(request)
            if cached is None:
                values = provider.download(ticker, start, end)
                values.index = pd.DatetimeIndex(values.index).tz_localize(None).normalize()
                if values.index.has_duplicates or not values.index.is_monotonic_increasing:
                    raise ValueError("Duplicate or unordered provider dates")
                if values.index.hasnans:
                    raise ValueError("Invalid provider timestamps")
                if (values.index < pd.Timestamp(start)).any() or (
                    values.index >= pd.Timestamp(end)
                ).any():
                    raise ValueError("Provider observations outside requested range")
                if not values.index.isin(session_schedule(config.calendar, start, end).index).all():
                    raise ValueError("Provider returned observations outside exchange sessions")
                table = values.to_frame(ticker)
                metadata = cache.write(request, table, currency=config.assets[ticker]["currency"])
            else:
                table, metadata = cached
                table.index = pd.to_datetime(table.index)
                if list(table.columns) != [ticker] or table.index.has_duplicates:
                    raise ValueError("Invalid cached assets or dates")
                values = table[ticker]
            aligned = values.reindex(schedule.index)
            first = aligned.first_valid_index()
            if first is None:
                raise ValueError("No valid observations on completed sessions")
            available = aligned.loc[first:]
            if (
                available.isna().any()
                or not np.isfinite(available.to_numpy()).all()
                or (available <= 0).any()
            ):
                raise ValueError("Missing, nonfinite or nonpositive observations after first quote")
            if len(available) < 201:
                raise ValueError("Insufficient available history for signal warmup")
            if first > schedule.index[0]:
                event(
                    logging.WARNING,
                    "short_history",
                    ticker=ticker,
                    first_session=first,
                    message="Leading history unavailable; common sample starts later",
                )
            series.append(aligned.rename(ticker))
            provenance[ticker] = metadata
        except Exception as exc:
            event(
                logging.ERROR, "market_request_failed", ticker=ticker, error_type=type(exc).__name__
            )
            raise ValueError(f"Market retrieval/quality failure for {ticker}") from exc
    prices = pd.concat(series, axis=1)
    first_common = max(s.first_valid_index() for s in series)
    prices = prices.loc[first_common:]
    closes = schedule.loc[prices.index]
    metadata = {
        "source": provider.name,
        "calendar": config.calendar,
        "currency": config.report_currency,
        "assets": provenance,
        "requested_start": start,
        "end_exclusive": end,
        "common_start": str(first_common.date()),
        "adjustment": "Vendor adjusted close; unverified total-return approximation",
        "alignment": "Leading unavailable history trimmed; internal gaps rejected",
    }
    event(
        logging.INFO,
        "market_panel_ready",
        observations=len(prices),
        assets=len(prices.columns),
        common_start=first_common,
    )
    return validate_panel(MarketPanel(prices, closes, metadata))


def synthetic_panel(config: ResearchConfig, start: str, end: str, seed: int = 2026) -> MarketPanel:
    """Seeded illustration on actual sessions, never a fallback for failed requests."""
    closes = session_schedule(config.calendar, start, end)
    if len(closes) < 201:
        raise ValueError("Synthetic demo needs 201 sessions")
    rng = np.random.default_rng(seed)
    tickers = list(config.assets)
    common = rng.normal(0, 0.003, (len(closes) - 1, 1))
    innovations = rng.normal(0.00015, 0.007, (len(closes) - 1, len(tickers))) + common
    values = np.vstack([np.ones(len(tickers)), np.exp(innovations).cumprod(axis=0)]) * 100
    prices = pd.DataFrame(values, index=closes.index, columns=tickers)
    return validate_panel(
        MarketPanel(
            prices,
            closes,
            {
                "source": "synthetic",
                "seed": seed,
                "calendar": config.calendar,
                "currency": config.report_currency,
                "warning": "Not historical market observations",
            },
        )
    )
