from __future__ import annotations

import pandas as pd

from datahub.schemas import PriceField, REQUIRED_OHLCV_COLUMNS


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert vendor output into canonical OHLCV frame:
    timestamp, open, high, low, close, volume
    - timestamp is UTC datetime64
    - sorted ascending
    - duplicates removed
    - numeric columns enforced
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=list(REQUIRED_OHLCV_COLUMNS))

    df = df.rename(
        columns={
            "Datetime": "timestamp",
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    if PriceField.timestamp.value not in df.columns and df.index is not None:
        if isinstance(df.index, (pd.DatetimeIndex,)):
            df = df.reset_index().rename(columns={"index": "timestamp"})

    df[PriceField.timestamp.value] = pd.to_datetime(df[PriceField.timestamp.value], utc=True)

    df = df.loc[:, [c for c in REQUIRED_OHLCV_COLUMNS if c in df.columns]]

    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="raise")

    df = df.sort_values(PriceField.timestamp.value)
    df = df.drop_duplicates(subset=[PriceField.timestamp.value], keep="last")

    return df.reset_index(drop=True)
