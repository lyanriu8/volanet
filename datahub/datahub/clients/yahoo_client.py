import yfinance as yf
import pandas as pd
from dataclasses import dataclass
from ..schemas import Interval, PriceField
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")

@dataclass
class YahooRequest:
    """Client single ticker request to yFinance"""
    ticker: str
    interval: Interval
    start: datetime #date is in utc -> must conver to NY
    end: datetime
    
@dataclass
class YahooBulkRequest:
    """Client bulk tickers request to yFinance"""
    tickers: list[str]
    interval: Interval
    start: datetime #date must be in yyyy-mm-dd
    end: datetime


def get_ohlcv(r: YahooRequest) -> pd.DataFrame:
    """
    Returns raw ohlcv from yfinance api
    -> intraday data is unreliable :(
    """
    
    limit_yahoo(r.interval)
    r.start = to_yahoo_time(r.start)
    r.end = to_yahoo_time(r.end)
    
    df = yf.Ticker(r.ticker).history(start=r.start, end=r.end, interval=r.interval, auto_adjust=False, actions=False)
    
    if df is None:
        return pd.DataFrame()
    
    df = df.reset_index()
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
    
    return df

def get_bulk_ohlcv(r: YahooBulkRequest) -> pd.DataFrame:
    """
    Returns raw ohlcv from yfinance api
    -> intraday data is unreliable :(
    """
    
    limit_yahoo(r.interval)
    
    df = yf.download(tickers=r.tickers, interval=r.interval, start=r.start, end=r.end, group_by='ticker')
    
    if df is None:
        return pd.DataFrame()
    
    df = df.reset_index()
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
    
    return df

def limit_yahoo(interval: Interval):
    if interval in [Interval.m1, Interval.m5, Interval.m15, Interval.h1,]:
        raise ValueError("Yahoo client can only take 1day bar data")

def to_yahoo_time(dt: datetime):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(NY)

# Test
if __name__ == "__main__":
    # tickers = ["AAPL"]

    # df = get_bulk_ohlcv(r=YahooBulkRequest(tickers, Interval.d1, datetime(2023, 12, 1), datetime(2024, 1, 1)))
    # print(df.info())
    # print(df.head())
    
    df = get_ohlcv(r=YahooRequest("AAPL", Interval.d1, datetime(2023, 12, 1, tzinfo=timezone.utc), datetime(2024, 1, 1, tzinfo=timezone.utc)))
    
    print(df.info())