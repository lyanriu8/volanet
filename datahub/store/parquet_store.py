from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class ParquetStoreConfig:
    root_dir: Path = Path("data/ohlcv_parquet")  # change if you want
    partition_by: Optional[str] = None          # keep None for simplicity


def _key_dir(cfg: ParquetStoreConfig, *, adjusted: bool) -> Path:
    # separate adjusted/unadjusted so you never mix them by accident
    return cfg.root_dir / f"adjusted={str(adjusted).lower()}"


def parquet_path(cfg: ParquetStoreConfig, *, ticker: str, interval: str, adjusted: bool) -> Path:
    # 1 file per (ticker, interval)
    base = _key_dir(cfg, adjusted=adjusted)
    return base / ticker.upper() / f"{interval}.parquet"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_ohlcv(
    cfg: ParquetStoreConfig,
    *,
    ticker: str,
    interval: str,
    adjusted: bool,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> Optional[pd.DataFrame]:
    """
    Loads canonical OHLCV from parquet, optionally time-sliced.
    Assumes dataframe has a 'timestamp' column.
    """
    path = parquet_path(cfg, ticker=ticker, interval=interval, adjusted=adjusted)
    if not path.exists():
        return None

    df = pd.read_parquet(path)

    if start is not None:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["timestamp"] <= pd.Timestamp(end)]

    # always return sorted / reset
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def upsert_ohlcv(
    cfg: ParquetStoreConfig,
    *,
    ticker: str,
    interval: str,
    adjusted: bool,
    df_new: pd.DataFrame,
) -> Path:
    """
    Upserts rows by 'timestamp' into parquet.
    - concatenates old + new
    - drops duplicate timestamps (keeps last)
    - sorts by timestamp
    """
    if "timestamp" not in df_new.columns:
        raise ValueError("df_new must contain a 'timestamp' column")

    path = parquet_path(cfg, ticker=ticker, interval=interval, adjusted=adjusted)
    ensure_parent(path)

    if path.exists():
        df_old = pd.read_parquet(path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new.copy()

    # canonicalize timestamp dtype
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")

    # dedupe and sort
    df = (
        df.sort_values("timestamp")
          .drop_duplicates(subset=["timestamp"], keep="last")
          .reset_index(drop=True)
    )

    df.to_parquet(path, index=False)
    return path
