"""FRED retrieval and vintage-aware availability. Macro is chart-only in Phase 1."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import requests

from aycapital.cross_asset.cache import SnapshotCache
from aycapital.cross_asset.logging import event


@dataclass(frozen=True)
class MacroData:
    observations: pd.DataFrame
    metadata: dict[str, Any]


class MacroProvider(Protocol):
    def load(
        self,
        series: str,
        start: str,
        end: str,
        as_of: str,
        *,
        mode: str = "latest",
        refresh: bool = False,
    ) -> MacroData: ...


class FredProvider:
    def __init__(
        self,
        cache_dir: str | Path,
        api_key: str | None = None,
        session: requests.Session | None = None,
    ):
        self.cache = SnapshotCache(Path(cache_dir) / "macro")
        self.api_key = api_key if api_key is not None else os.environ.get("FRED_API_KEY")
        self.session = session or requests.Session()

    def load(
        self,
        series: str,
        start: str,
        end: str,
        as_of: str,
        *,
        mode: str = "latest",
        refresh: bool = False,
    ) -> MacroData:
        if mode not in {"latest", "vintages"}:
            raise ValueError("Unknown macro retrieval mode")
        if pd.Timestamp(start) > pd.Timestamp(end) or pd.Timestamp(end) > pd.Timestamp(as_of):
            raise ValueError("Observation range must end no later than the as-of date")
        request = {
            "provider": "fred",
            "series": series,
            "start": start,
            "end": end,
            "as_of": as_of,
            "mode": mode,
            "schema": 1,
        }
        cached = None if refresh else self.cache.read(request)
        if cached is not None:
            data, metadata = cached
            data = _validate_observations(data, mode)
            return MacroData(data, metadata)
        if not self.api_key:
            raise ValueError("FRED_API_KEY is required for uncached macro retrieval")
        params = {
            "series_id": series,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end,
            "realtime_start": as_of if mode == "latest" else "1776-07-04",
            "realtime_end": as_of,
            "output_type": 1,
            "limit": 100000,
            "offset": 0,
            "sort_order": "asc",
        }
        rows = []
        try:
            while True:
                response = self.session.get(
                    "https://api.stlouisfed.org/fred/series/observations", params=params, timeout=30
                )
                response.raise_for_status()
                payload = response.json()
                page = payload["observations"]
                rows.extend(page)
                count = int(payload["count"])
                params["offset"] += len(page)
                if params["offset"] >= count:
                    break
                if not page:
                    raise ValueError("Incomplete paginated FRED response")
            if not rows:
                raise ValueError("FRED returned no observations")
            raw = pd.DataFrame(rows).rename(
                columns={"date": "observation_date", "realtime_start": "vintage_date"}
            )
            raw["value"] = raw["value"].replace(".", np.nan).astype(float)
            data = _validate_observations(raw, mode)
            metadata = self.cache.write(
                request,
                data,
                historical_use=(mode == "vintages"),
                availability="Conservative next UTC day after vintage date",
                warning="Latest snapshot cannot enter earlier backtests",
            )
            event(logging.INFO, "macro_retrieved", series=series, rows=len(data), mode=mode)
            return MacroData(data, metadata)
        except Exception as exc:
            # HTTP error strings can include the key in the URL; do not propagate them.
            event(
                logging.ERROR, "macro_request_failed", series=series, error_type=type(exc).__name__
            )
            raise ValueError(f"FRED retrieval/quality failure for {series}") from None


def _validate_observations(data: pd.DataFrame, mode: str) -> pd.DataFrame:
    required = {"observation_date", "vintage_date", "realtime_end", "value"}
    if data.empty or not required.issubset(data):
        raise ValueError("Invalid macro observation schema")
    data = data.loc[:, sorted(required)].copy()
    data["observation_date"] = pd.to_datetime(data.observation_date)
    data["vintage_date"] = pd.to_datetime(data.vintage_date)
    data["value"] = pd.to_numeric(data.value).astype(float)
    if (
        data.observation_date.isna().any()
        or data.vintage_date.isna().any()
        or np.isinf(data.value).any()
    ):
        raise ValueError("Invalid macro values or dates")
    if data.duplicated(["observation_date", "vintage_date"]).any():
        raise ValueError("Duplicate macro vintages")
    if mode == "latest" and data.observation_date.duplicated().any():
        raise ValueError("Latest snapshot contains multiple vintages")
    if mode == "vintages" and (data.vintage_date < data.observation_date).any():
        raise ValueError("Vintage precedes its observation period")
    return data.sort_values(["observation_date", "vintage_date"]).reset_index(drop=True)


def available_as_of(data: MacroData, timestamps: pd.DatetimeIndex) -> pd.Series:
    """Latest observation known at each timestamp, with revisions handled causally.

    FRED vintage dates have no intraday release timestamp. Delay to next UTC
    midnight; this is conservative, not an assertion of the actual release time.
    Macro publication frequency is retained; no observation-date forward fill.
    """
    if not data.metadata.get("historical_use"):
        raise ValueError("Latest/revised macro snapshots cannot be used historically")
    if timestamps.tz is None:
        raise ValueError("As-of timestamps must be timezone-aware")
    observations = _validate_observations(data.observations, "vintages")
    available = observations.vintage_date.dt.tz_localize("UTC") + pd.Timedelta(days=1)
    result = []
    for timestamp in timestamps:
        known = observations.loc[available <= timestamp]
        if known.empty:
            result.append(np.nan)
            continue
        revised = known.sort_values("vintage_date").drop_duplicates("observation_date", keep="last")
        # A published missing value stays missing rather than resurrecting an old value.
        result.append(revised.sort_values("observation_date").value.iloc[-1])
    return pd.Series(result, index=timestamps, name="macro_value")


def display_series(data: MacroData) -> pd.Series:
    """Most recent retrieved value per period, for descriptive charts only."""
    observations = data.observations.sort_values("vintage_date").drop_duplicates(
        "observation_date", keep="last"
    )
    return observations.set_index("observation_date").value.sort_index()


def macro_display_frame(series: dict[str, MacroData]) -> pd.DataFrame:
    frame = pd.concat({name: display_series(data) for name, data in series.items()}, axis=1)
    if {"DGS2", "DGS10"}.issubset(frame):
        frame["10Y_minus_2Y"] = frame.DGS10 - frame.DGS2
    # Calendar year comparison for monthly inflation indices, no missing-period filling.
    for index in ("CPIAUCSL", "PCEPI"):
        if index in frame:
            monthly = frame[index].dropna()
            monthly.index = monthly.index.to_period("M")
            if monthly.index.has_duplicates:
                raise ValueError("Inflation index must have one observation per month")
            previous = monthly.reindex(monthly.index - 12).to_numpy()
            yoy = pd.Series(
                (monthly.to_numpy() / previous - 1) * 100, index=monthly.index.to_timestamp()
            )
            frame[f"{index}_yoy_percent"] = yoy.reindex(frame.index)
    return frame
