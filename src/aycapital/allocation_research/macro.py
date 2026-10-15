"""Public ALFRED knowledge snapshots, never revised current values backdated."""

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from aycapital.cross_asset.cache import SnapshotCache


class PublicVintageProvider:
    """Key-free ALFRED graph snapshots at explicit historical vintage dates.

    Availability is next UTC day after the requested snapshot, deliberately later
    than initial release. No assertion of exact publication times. Downloads and
    revision states are immutable, checksummed snapshots in the existing cache.
    """

    def __init__(self, cache_dir: str | Path, session=None):
        self.cache = SnapshotCache(Path(cache_dir) / "public_vintages")
        self.session = session or requests.Session()

    def load(self, series: str, start: str, end: str) -> tuple[pd.DataFrame, list[dict]]:
        vintages = pd.date_range(start, end, freq="ME")
        # Include the last frozen knowledge state, never today's revised snapshot.
        vintages = vintages.union(pd.DatetimeIndex([pd.Timestamp(end)]))
        records, provenance = [], []
        for left in range(0, len(vintages), 12):
            dates = vintages[left : left + 12].strftime("%Y-%m-%d").tolist()
            request = {
                "provider": "alfred_graph",
                "series": series,
                "vintage_dates": dates,
                "start": start,
                "schema": 1,
            }
            cached = self.cache.read(request)
            if cached is None:
                response = self.session.get(
                    "https://alfred.stlouisfed.org/graph/alfredgraph.csv",
                    params={"id": ",".join([series] * len(dates)), "vintage_date": ",".join(dates)},
                    timeout=60,
                )
                response.raise_for_status()
                raw = pd.read_csv(StringIO(response.text), index_col=0, na_values=".")
                expected = [f"{series}_{day.replace('-', '')}" for day in dates]
                returned = raw.columns.tolist()
                if len(returned) != len(dates):
                    raise ValueError("ALFRED vintage identities have incorrect count")
                for column, day in zip(returned, dates, strict=True):
                    prefix = f"{series}_"
                    suffix = column.removeprefix(prefix)
                    if (
                        not column.startswith(prefix)
                        or len(suffix) != 8
                        or not suffix.isdigit()
                        or pd.to_datetime(suffix, format="%Y%m%d") > pd.Timestamp(day)
                    ):
                        raise ValueError("ALFRED vintage identities are missing or forward-looking")
                # ALFRED can cap the final column to its latest archived day.
                # Earlier returned states are safe with our LATER requested known-by date.
                raw.columns = expected
                raw.index = pd.to_datetime(raw.index)
                raw = raw.loc[raw.index >= pd.Timestamp(start) - pd.DateOffset(years=2)]
                metadata = self.cache.write(
                    request,
                    raw,
                    historical_use=True,
                    source_url="https://alfred.stlouisfed.org/graph/alfredgraph.csv",
                    availability="Next UTC day after known-by snapshot; not initial release",
                    returned_vintage_columns=returned,
                )
            else:
                raw, metadata = cached
                raw.index = pd.to_datetime(raw.index)
            provenance.append(metadata)
            for day in dates:
                col = f"{series}_{day.replace('-', '')}"
                values = pd.to_numeric(raw[col], errors="raise").dropna()
                if values.empty or not np.isfinite(values).all():
                    raise ValueError("Empty/nonfinite ALFRED historical snapshot")
                if (values.index > pd.Timestamp(day)).any() or values.index.has_duplicates:
                    raise ValueError("Invalid observation periods in ALFRED snapshot")
                for observation, value in values.items():
                    records.append(
                        {
                            "series": series,
                            "observation_date": observation,
                            "known_by_date": pd.Timestamp(day),
                            "value": value,
                        }
                    )
        return pd.DataFrame(records), provenance


def monthly_features(snapshots: pd.DataFrame, closes: pd.Series) -> pd.DataFrame:
    """Align whole revision states; changes use same-vintage calendar-month lags.

    Do not use the latest revised historical value for a lagged comparison.
    A snapshot older than 45 calendar days is unavailable, not indefinitely filled.
    """
    required = {"series", "observation_date", "known_by_date", "value"}
    if not required.issubset(snapshots) or closes.dt.tz is None:
        raise ValueError("Need vintage snapshots and timezone-aware decision closes")
    data = snapshots.copy()
    for column in ["known_by_date", "observation_date"]:
        data[column] = pd.to_datetime(data[column])
    if (data.observation_date > data.known_by_date).any():
        raise ValueError("Macro observation period exceeds knowledge date")
    if data.duplicated(["series", "observation_date", "known_by_date"]).any():
        raise ValueError("Duplicate macro vintage observations")
    if not np.isfinite(data.value).all():
        raise ValueError("Invalid macro values")
    states = {}
    for (series, vintage), group in data.groupby(["series", "known_by_date"]):
        values = group.set_index("observation_date").value
        values.index = values.index.to_period("M")
        if values.index.has_duplicates:
            raise ValueError("Expected monthly macro observations")
        values = values.sort_index()
        latest = values.index[-1]
        value = float(values.iloc[-1])
        # Require actual observations rather than filling missing calendar months.
        yoy = (value / values.get(latest - 12, np.nan) - 1) * 100
        old_yoy = (values.get(latest - 3, np.nan) / values.get(latest - 15, np.nan) - 1) * 100
        states.setdefault(series, []).append(
            {
                "available_at_utc": vintage.tz_localize("UTC") + pd.Timedelta(days=1),
                "known_by_date": vintage,
                "observation_date": latest.to_timestamp(),
                "value": value,
                "yoy": yoy,
                "yoy_change_3m": yoy - old_yoy,
                "change_3m": value - values.get(latest - 3, np.nan),
            }
        )
    output = pd.DataFrame(index=closes.index)
    for series, rows in states.items():
        right = pd.DataFrame(rows).sort_values("available_at_utc")
        left = pd.DataFrame({"close": closes.to_numpy()})
        joined = pd.merge_asof(
            left,
            right,
            left_on="close",
            right_on="available_at_utc",
            direction="backward",
            tolerance=pd.Timedelta(days=45),
        )
        stale = (joined.close.dt.tz_localize(None) - joined.observation_date).dt.days > 90
        joined.loc[stale, ["value", "yoy", "yoy_change_3m", "change_3m"]] = np.nan
        joined.index = closes.index
        output = output.join(joined.drop(columns="close").add_prefix(f"{series}_"))
    return output
