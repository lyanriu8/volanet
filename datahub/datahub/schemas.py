from __future__ import annotations

from enum import Enum 
from datetime import datetime, timezone
from typing import Literal, Optional
from dataclasses import dataclass
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

INTERVAL = Literal["1m", "5m", "15m", "1d", "1h"] # Can only be these strings

class Interval(str, Enum):                        # has both type string and type enum
    """Canonical bar interval"""
    m1 = "1m"
    m5 = "5m"
    m15 = "15m"
    d1 = "1d"
    h1 = "1h"

class PriceField(str, Enum):                     # has both type string and type enum
    """Canonical price field expected in OHLCV frames"""
    timestamp = "timestamp"
    open = "open"
    high = "high"
    low = "low"
    close = "close"
    volume = "volume"
    

REQUIRED_OHLCV_COLUMNS = (
    PriceField.timestamp.value,
    PriceField.open.value,
    PriceField.high.value,
    PriceField.low.value,
    PriceField.close.value,
    PriceField.volume.value,
)

class OHLCVSpec(BaseModel):
    """Invariants of an OHLCV frame that datahub returns -> contract volanet can rely on"""
    interval: Interval
    tz: str = Field(default="UTC", description="Timezone name used for timestamps")
    adjusted: bool = Field(default=False, description="True if OHLCV is adjusted for corporate actions, else false")
    source: str = Field(default="unknown", description="Source identifier")
    columns: tuple[str, ...] = Field(default=REQUIRED_OHLCV_COLUMNS)
    
    @field_validator("tz")
    @classmethod #class methods are methods native to class that exist before instance -> field validators are ran before the model is created
    def tz_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("tz must be non-empty")
        return v;

class OHLCVQuery(BaseModel):
    """Parameters for requesting OHLCV data"""
    ticker: str = Field(..., min_length=1, description="Company ticker")
    interval: Interval = Interval.d1
    start: Optional[datetime] = Field(default=None, description="Inclusive start datetime")
    end: Optional[datetime] = Field(default=None, description="Exclusive end datetime")
    limit: Optional[int] = Field(default=None, ge=1, description="Max number of bars to return")
    adjusted: bool = False
    
    @model_validator(mode="after") #means validator ran after instantiation -> if mode="before" then this would have to be a class method
    def check_range(self) -> "OHLCVQuery":
        if self.start and self.end and self.start >= self.end:
            raise ValueError("start must be less than end")
        return self
    
    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        return v.strip().upper()


class OHLCVWindowRequest(BaseModel):
    """Request for retrieving given amount of bars up to a certain date"""
    ticker: str
    interval: Interval = Interval.d1
    bars: int = Field(default=..., description="Number of bars needed for feature windows")
    as_of: Optional[datetime] = Field(default=None, description="Exclusive end time, defaults to now()")
    adjusted: bool = False
    
    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        return v.strip().upper()
    
    @field_validator("as_of")
    @classmethod
    def default_as_of_utc(cls, v: Optional[datetime]) -> datetime:
        if v is None: 
            return datetime.now(timezone.utc)
        if v.tzinfo is None: 
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

@dataclass
class DatahubFrame():
    """Wrapper around OHLCV pandas DataFrame its spec 
    -> avoids pydantic-serializing a DataFrame across HTTP 
    -> boundary between datahub and callers"""
    
    spec: OHLCVSpec
    df: pd.DataFrame
    
    def validate_frame(self) -> None:
        validate_ohlcv_df(self.df, self.spec)


def validate_ohlcv_df(df: pd.DataFrame, spec: Optional[OHLCVSpec] = None) -> None:
    """
    Enforce the canonical OHLCV contract.
    Call this inside datahub before returning, and inside volanet before using.
    """

    if df is None or len(df) == 0:
        raise ValueError("OHLCV DataFrame is empty")

    missing = [c for c in REQUIRED_OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV DataFrame missing required columns: {missing}")

    # timestamps must be monotonic increasing
    ts = df[PriceField.timestamp.value]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        raise ValueError("timestamp column must be datetime64 dtype")

    if not ts.is_monotonic_increasing:
        raise ValueError("timestamp must be sorted ascending")

    # basic numeric sanity
    for c in (PriceField.open.value, PriceField.high.value, PriceField.low.value, PriceField.close.value, PriceField.volume.value):
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"{c} must be numeric")

    # optional: OHLC consistency
    if (df[PriceField.high.value] < df[PriceField.low.value]).any():
        raise ValueError("Found high < low in OHLCV frame")

    if spec is not None:
        # ensure caller gets what was promised
        for c in spec.columns:
            if c not in df.columns:
                raise ValueError(f"Frame violates spec: missing column {c}")

    