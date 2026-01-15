
# datahub/datahub/facade.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from datahub.schemas import DatahubFrame, OHLCVQuery, OHLCVSpec, OHLCVWindowRequest, Interval, validate_ohlcv_df


# ----- Public API ------

def get_window(r: OHLCVWindowRequest) -> DatahubFrame:
    """window based fetch -> for inference"""
    
    ticker = r.ticker
    interval = r.interval
    bars = r.bars
    end = r.as_of or datetime.now(timezone.utc)
    adjusted = r.adjusted
    start = _get_start(end=end, interval=interval, bars=bars)
    
    df = _get_df(ticker, interval, start, end, None, adjusted)
    
    df = _normalize(df) #change this later
    
    # Keep only the last N bars (after normalization)
    df = df.tail(bars).reset_index(drop=True)
    
    spec = OHLCVSpec(interval=interval, adjusted=adjusted, source="not yet :)")
    
    validate_ohlcv_df(df, spec)
    
    return DatahubFrame(spec=spec, df=df)
    

def get_history(q: OHLCVQuery) -> DatahubFrame:
    """range based fetch -> for training"""
    ticker = q.ticker
    interval = q.interval
    end = q.end
    adjusted = q.adjusted
    start = q.start
    
    df = _get_df(ticker=ticker, interval=interval, start=start, end=end, limit=None, adjusted=adjusted)
    
    df = _normalize(df) #change this later
    
    spec = OHLCVSpec(interval=interval, adjusted=adjusted, source="not yet :)")
    
    validate_ohlcv_df(df, spec)
    
    return DatahubFrame(spec=spec, df=df);
    


# ------ Helpers -------

def _get_df(ticker: str, interval: Interval, start: Optional[datetime], end: Optional[datetime], limit: Optional[int], adjusted: bool,) -> pd.DataFrame:
    """
    Single internal entry point for fetching.
    For now: STUB. Later: call cache/store/vendor in this order.
    Must return a DataFrame with at least timestamp/open/high/low/close/volume
    (column names can be vendor-ish; normalization will fix).
    """
    
    raise NotImplementedError("Wire to first client")

def _get_start(*, end: datetime, interval: Interval, bars: int) -> datetime:
    """
    Conservative approximation.
    For daily bars, assume 1 bar ~ 1 day (overfetch a bit for weekends/holidays).
    For minute/hour, overfetch similarly.
    Replace this later with smarter method later
    """
    
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    if interval == Interval.d1:
        # overfetch 2x to cover weekends/holidays
        return end - timedelta(days=bars * 2)
    if interval == Interval.h1:
        return end - timedelta(hours=bars * 2)
    if interval == Interval.m15:
        return end - timedelta(minutes=bars * 15 * 2)
    if interval == Interval.m5:
        return end - timedelta(minutes=bars * 5 * 2)
    if interval == Interval.m1:
        return end - timedelta(minutes=bars * 2)

    # fallback
    return end - timedelta(days=bars * 2)

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Stub. Actual logic should be in pipline"""
    return df
