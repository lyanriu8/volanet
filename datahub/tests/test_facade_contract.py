# datahub/tests/test_facade_contract.py

from __future__ import annotations
        
from datetime import datetime, timezone

import pandas as pd
import pytest

from datahub.schemas import REQUIRED_OHLCV_COLUMNS, Interval, OHLCVQuery, OHLCVWindowRequest
import datahub.facade as facade


def _raw_yahoo_like_df_unsorted() -> pd.DataFrame:
    """
    Simulate raw-ish yahoo/yfinance output BEFORE normalize_ohlcv:
    - timestamp col as 'Date'
    - capitalized OHLCV
    - unsorted timestamps
    """
    t1 = datetime(2025, 1, 2)
    t2 = datetime(2025, 1, 3)
    t3 = datetime(2025, 1, 6)

    return pd.DataFrame(
        {
            "Date": [t2, t1, t3],  # unsorted on purpose
            "Open": [182.6, 181.2, 184.0],
            "High": [184.4, 183.1, 186.2],
            "Low": [181.7, 180.9, 183.4],
            "Close": [183.9, 182.5, 185.7],
            "Volume": [49_811_200, 53_422_100, 61_299_300],
        }
    )


def _patch_client_fetch(monkeypatch: pytest.MonkeyPatch, raw_df: pd.DataFrame) -> None:
    """
    Patch the exact symbol used by your facade:
    facade.get_ohlcv(...)
    """
    monkeypatch.setattr(facade, "get_ohlcv", lambda req: raw_df)


def test_get_history_returns_canonical_datahubframe(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _raw_yahoo_like_df_unsorted()
    _patch_client_fetch(monkeypatch, raw)

    q = OHLCVQuery(
        ticker=" aapl ",
        interval=Interval.d1,
        start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end=datetime(2025, 1, 10, tzinfo=timezone.utc),
        adjusted=False,
    )

    frame = facade.get_history(q)

    # wrapper/spec sanity
    assert frame.spec.interval == Interval.d1
    assert frame.spec.adjusted is False

    df = frame.df

    # required columns exist
    assert set(REQUIRED_OHLCV_COLUMNS).issubset(df.columns)

    # timestamp dtype + ordering
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert df["timestamp"].is_monotonic_increasing

    # numeric dtypes
    for c in ("open", "high", "low", "close", "volume"):
        assert pd.api.types.is_numeric_dtype(df[c])

    # contract validator should pass
    frame.validate_frame()


def test_get_window_tails_to_requested_bars(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _raw_yahoo_like_df_unsorted()
    _patch_client_fetch(monkeypatch, raw)

    req = OHLCVWindowRequest(
        ticker="AAPL",
        interval=Interval.d1,
        bars=2,
        as_of=datetime(2025, 1, 10, tzinfo=timezone.utc),
        adjusted=False,
    )

    frame = facade.get_window(req)
    assert len(frame.df) == 2
    frame.validate_frame()


def test_get_window_uses_as_of_default_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    We don't assert exact timestamps because your _get_start overfetching is approximate.
    This just checks the call doesn't crash and the output is canonical.
    """
    raw = _raw_yahoo_like_df_unsorted()
    _patch_client_fetch(monkeypatch, raw)

    req = OHLCVWindowRequest(
        ticker="AAPL",
        interval=Interval.d1,
        bars=3,
        as_of=None,  # should default to now() inside get_window
        adjusted=False,
    )

    frame = facade.get_window(req)
    assert set(REQUIRED_OHLCV_COLUMNS).issubset(frame.df.columns)
    frame.validate_frame()
