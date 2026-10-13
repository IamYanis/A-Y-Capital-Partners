import numpy as np
import pandas as pd
import pytest
import requests

from aycapital.cross_asset.macro import (
    FredProvider,
    MacroData,
    available_as_of,
    macro_display_frame,
)


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class Session:
    def __init__(self):
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append(dict(params))
        row = {
            "date": "2024-01-01",
            "realtime_start": "2024-02-10",
            "realtime_end": "2024-02-29",
            "value": "100",
        }
        if params["offset"]:
            row = row | {
                "realtime_start": "2024-03-01",
                "realtime_end": "9999-12-31",
                "value": "110",
            }
        return Response({"count": 2, "observations": [row]})


def test_fred_pagination_cache_and_release_guard(tmp_path):
    session = Session()
    provider = FredProvider(tmp_path, api_key="test-secret", session=session)
    data = provider.load("CPIAUCSL", "2024-01-01", "2024-03-10", "2024-03-10", mode="vintages")
    assert [call["offset"] for call in session.calls] == [0, 1]
    assert session.calls[0]["realtime_start"] == "1776-07-04"
    cached = provider.load("CPIAUCSL", "2024-01-01", "2024-03-10", "2024-03-10", mode="vintages")
    pd.testing.assert_frame_equal(data.observations, cached.observations)
    assert len(session.calls) == 2
    assert all("test-secret" not in path.read_text() for path in tmp_path.rglob("*.json"))
    times = pd.DatetimeIndex(
        [
            "2024-02-01 21:00Z",
            "2024-02-10 21:00Z",
            "2024-02-11 00:00Z",
            "2024-03-01 21:00Z",
            "2024-03-02 00:00Z",
        ]
    )
    values = available_as_of(data, times)
    assert np.isnan(values.iloc[0]) and np.isnan(values.iloc[1])
    np.testing.assert_allclose(values.iloc[2:], [100, 100, 110])
    with pytest.raises(ValueError, match="snapshots"):
        available_as_of(MacroData(data.observations, {"historical_use": False}), times)


def test_macro_release_uses_latest_known_period_not_future_revision():
    observations = pd.DataFrame(
        {
            "observation_date": ["2024-01-01", "2024-01-01", "2024-02-01"],
            "vintage_date": ["2024-02-10", "2024-03-10", "2024-03-15"],
            "realtime_end": ["2024-03-09", "9999-12-31", "9999-12-31"],
            "value": [100.0, 105.0, np.nan],
        }
    )
    data = MacroData(observations, {"historical_use": True})
    times = pd.DatetimeIndex(["2024-03-01 21:00Z", "2024-03-11 21:00Z", "2024-03-16 21:00Z"])
    values = available_as_of(data, times)
    np.testing.assert_allclose(values.iloc[:2], [100, 105])
    assert np.isnan(values.iloc[2])


def test_macro_error_does_not_leak_credentials(tmp_path, caplog):
    class Broken:
        def get(self, *args, **kwargs):
            raise requests.HTTPError("https://fred.example?api_key=test-secret")

    provider = FredProvider(tmp_path, api_key="test-secret", session=Broken())
    with pytest.raises(ValueError) as failure:
        provider.load("DGS2", "2024-01-01", "2024-02-01", "2024-02-01")
    assert "test-secret" not in str(failure.value)
    assert "test-secret" not in caplog.text
    assert "macro_request_failed" in caplog.text


def test_missing_fred_key(tmp_path, monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(ValueError, match="FRED_API_KEY"):
        FredProvider(tmp_path).load("DGS2", "2024-01-01", "2024-02-01", "2024-02-01")


def test_yield_spread_and_inflation_calendar_alignment():
    def data(dates, values):
        return MacroData(
            pd.DataFrame(
                {
                    "observation_date": pd.to_datetime(dates),
                    "vintage_date": pd.to_datetime(dates),
                    "value": values,
                }
            ),
            {},
        )

    dates = pd.date_range("2023-01-01", periods=14, freq="MS")
    cpi = data(dates, [100.0] * 12 + [105.0, 106.0])
    frame = macro_display_frame(
        {"DGS2": data(dates[:2], [4.0, 4.1]), "DGS10": data(dates[:2], [3.0, 3.2]), "CPIAUCSL": cpi}
    )
    assert frame.loc[dates[0], "10Y_minus_2Y"] == pytest.approx(-1)
    assert frame.loc[dates[12], "CPIAUCSL_yoy_percent"] == pytest.approx(5)
    assert frame.loc[dates[13], "CPIAUCSL_yoy_percent"] == pytest.approx(6)
    assert (
        frame.loc[dates[11], "CPIAUCSL_yoy_percent"] != frame.loc[dates[11], "CPIAUCSL_yoy_percent"]
    )  # Undefined without a prior calendar year.


def test_expired_macro_vintage_is_not_resurrected():
    data = MacroData(
        pd.DataFrame(
            {
                "observation_date": ["2024-01-01"],
                "vintage_date": ["2024-02-10"],
                "realtime_end": ["2024-02-12"],
                "value": [100.0],
            }
        ),
        {"historical_use": True},
    )
    times = pd.DatetimeIndex(["2024-02-13 21:00Z", "2024-02-14 00:00Z"])
    values = available_as_of(data, times)
    assert values.iloc[0] == 100  # Conservative one-day delayed information clock.
    assert np.isnan(values.iloc[1])  # No replacement or valid record remains.


def test_macro_snapshot_does_not_claim_coverage_after_retrieval_horizon():
    data = MacroData(
        pd.DataFrame(
            {
                "observation_date": ["2024-01-01"],
                "vintage_date": ["2024-02-10"],
                "realtime_end": ["9999-12-31"],
                "value": [100.0],
            }
        ),
        {"historical_use": True, "request": {"as_of": "2024-02-12"}},
    )
    values = available_as_of(data, pd.DatetimeIndex(["2024-02-15 21:00Z"]))
    assert np.isnan(values.iloc[0])


@pytest.mark.parametrize("ends", [["2024-02-09", "9999-12-31"], ["2024-03-11", "9999-12-31"]])
def test_invalid_macro_validity_intervals_rejected(ends):
    data = MacroData(
        pd.DataFrame(
            {
                "observation_date": ["2024-01-01", "2024-01-01"],
                "vintage_date": ["2024-02-10", "2024-03-10"],
                "realtime_end": ends,
                "value": [100.0, 101.0],
            }
        ),
        {"historical_use": True},
    )
    with pytest.raises(ValueError):
        available_as_of(data, pd.DatetimeIndex(["2024-03-15 21:00Z"]))


def test_fred_rejects_future_vintage_from_provider(tmp_path):
    class FutureSession:
        def get(self, *args, **kwargs):
            return Response(
                {
                    "count": 1,
                    "observations": [
                        {
                            "date": "2024-01-01",
                            "realtime_start": "2024-04-01",
                            "realtime_end": "9999-12-31",
                            "value": "110",
                        }
                    ],
                }
            )

    provider = FredProvider(tmp_path, api_key="test-secret", session=FutureSession())
    with pytest.raises(ValueError, match="quality"):
        provider.load("CPIAUCSL", "2024-01-01", "2024-03-01", "2024-03-01", mode="vintages")
    assert not list(tmp_path.rglob("snapshot-*.csv"))
