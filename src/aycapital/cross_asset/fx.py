"""GBP conversion with explicit FX availability and bounded backward alignment."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from aycapital.cross_asset.cache import SnapshotCache
from aycapital.cross_asset.logging import event
from aycapital.cross_asset.market import MarketPanel, MarketProvider, YahooProvider, validate_panel
from aycapital.data import to_gbp


@dataclass(frozen=True)
class FxData:
    usd_per_gbp: pd.Series
    available_at_utc: pd.Series
    metadata: dict[str, Any]


def validate_fx(data: FxData) -> FxData:
    rates, times = data.usd_per_gbp, data.available_at_utc
    if (
        not isinstance(rates.index, pd.DatetimeIndex)
        or len(rates) < 2
        or rates.index.has_duplicates
        or rates.index.hasnans
        or not rates.index.is_monotonic_increasing
        or rates.index.tz is not None
    ):
        raise ValueError("FX needs unique sorted naive observation dates")
    if (
        not np.isfinite(rates.to_numpy(dtype=float)).all()
        or (rates <= 0).any()
        or not times.index.equals(rates.index)
        or times.isna().any()
    ):
        raise ValueError("FX observations must be positive, finite and exactly timestamped")
    if (
        not isinstance(times.dtype, pd.DatetimeTZDtype)
        or str(times.dt.tz) != "UTC"
        or not times.is_monotonic_increasing
    ):
        raise ValueError("FX availability must be increasing timezone-aware UTC")
    if (times < rates.index.tz_localize("UTC")).any():
        raise ValueError("FX cannot be available before its observation date")
    return data


def load_fx(
    start: str,
    end: str,
    cache_dir: str,
    provider: MarketProvider | None = None,
    *,
    refresh: bool = False,
) -> FxData:
    """Yahoo GBPUSD=X is USD per GBP; date-only quotes usable next UTC midnight.

    This is a conservative research convention, not a verified vendor release time.
    No exchange calendar is imposed on FX observations; no missing quote is filled.
    """
    provider = provider or YahooProvider()
    cache = SnapshotCache(cache_dir)
    request = {
        "provider": provider.name,
        "ticker": "GBPUSD=X",
        "start": start,
        "end": end,
        "schema": 1,
        "units": "USD per GBP",
    }
    cached = None if refresh else cache.read(request)
    if cached is None:
        rates = provider.download("GBPUSD=X", start, end).astype(float)
        rates.index = pd.DatetimeIndex(rates.index).tz_localize(None).normalize()
        times = pd.Series(
            rates.index.tz_localize("UTC") + pd.Timedelta(days=1),
            index=rates.index,
            name="available_at_utc",
        )
        if (
            rates.index.has_duplicates
            or rates.index.hasnans
            or not rates.index.is_monotonic_increasing
        ):
            raise ValueError("Invalid raw FX observation dates")
        in_scope = (rates.index >= pd.Timestamp(start)) & (rates.index < pd.Timestamp(end))
        usable = rates.notna() & in_scope
        validate_fx(FxData(rates.loc[usable], times.loc[usable], {}))
        table = pd.DataFrame({"usd_per_gbp": rates, "available_at_utc": times})
        metadata = cache.write(
            request,
            table,
            availability="Observation date + 1 UTC day",
            missing_observation_dates=[str(d.date()) for d in rates.index[~usable]],
        )
    else:
        table, metadata = cached
        table.index = pd.to_datetime(table.index)
        rates = table.usd_per_gbp.astype(float)
        times = pd.to_datetime(table.available_at_utc, utc=True)
    if rates.index.has_duplicates or rates.index.hasnans or not rates.index.is_monotonic_increasing:
        raise ValueError("Invalid raw FX observation dates")
    in_scope = (rates.index >= pd.Timestamp(start)) & (rates.index < pd.Timestamp(end))
    ignored = rates.index[~in_scope]
    if len(ignored):
        event(
            logging.WARNING,
            "out_of_range_fx_quotes",
            count=len(ignored),
            message="Raw quotes preserved; excluded from the requested interval",
        )
    metadata = dict(metadata, excluded_out_of_range_dates=[str(d.date()) for d in ignored])
    usable = rates.notna() & in_scope
    if rates.isna().any():
        event(
            logging.WARNING,
            "missing_fx_quotes",
            count=int(rates.isna().sum()),
            message="Missing raw quotes retained in cache; bounded backward alignment only",
        )
    return validate_fx(FxData(rates.loc[usable], times.loc[usable], metadata))


def gbp_panel(
    panel: MarketPanel, fx: FxData, max_age_days: int = 6
) -> tuple[MarketPanel, pd.DataFrame]:
    panel = validate_panel(panel)
    fx = validate_fx(fx)
    if panel.metadata.get("currency") != "USD":
        raise ValueError("This FX adapter requires USD-denominated asset prices")
    if isinstance(max_age_days, bool) or not isinstance(max_age_days, int) or max_age_days < 1:
        raise ValueError("FX maximum age must be a positive number of calendar days")
    decisions = pd.DataFrame(
        {"session": panel.prices.index, "asset_close_utc": panel.closes_utc.to_numpy()}
    )
    quotes = pd.DataFrame(
        {
            "fx_observation_date": fx.usd_per_gbp.index,
            "usd_per_gbp": fx.usd_per_gbp.to_numpy(),
            "fx_available_at_utc": fx.available_at_utc.to_numpy(),
        }
    )
    alignment = pd.merge_asof(
        decisions,
        quotes,
        left_on="asset_close_utc",
        right_on="fx_available_at_utc",
        direction="backward",
    )
    if alignment.usd_per_gbp.isna().any():
        raise ValueError("No already-available FX quote for one or more asset sessions")
    age = alignment.asset_close_utc - alignment.fx_observation_date.dt.tz_localize("UTC")
    if (age > pd.Timedelta(days=max_age_days)).any():
        raise ValueError(
            "FX quote exceeds the explicit maximum age; refresh or inspect missing data"
        )
    alignment["fx_age_calendar_days"] = age.dt.total_seconds() / 86400
    alignment = alignment.set_index("session")
    factors = pd.DataFrame({ticker: 1 / alignment.usd_per_gbp for ticker in panel.prices})
    prices = to_gbp(panel.prices, factors)
    warnings = [
        "GBP valuation uses latest already-available dated FX quotes, not synchronized "
        "equity/FX closes; daily FX dates are conservatively usable the next UTC day.",
        "Bounded backward FX alignment can span weekends/holidays; source dates are exported.",
    ]
    extended = alignment.fx_age_calendar_days > 4
    if extended.any():
        warnings.append(
            f"FX quote age exceeds four calendar days on {int(extended.sum())} "
            f"session(s); maximum observed age is {alignment.fx_age_calendar_days.max():.2f} "
            "days. This may distort synchronized-currency risk estimates."
        )
    metadata = dict(
        panel.metadata,
        currency="GBP",
        native_currency="USD",
        fx=fx.metadata,
        fx_policy={
            "max_age_calendar_days": max_age_days,
            "availability": "Next UTC day for Yahoo daily FX",
        },
        data_quality_warnings=warnings,
    )
    return validate_panel(MarketPanel(prices, panel.closes_utc, metadata)), alignment
