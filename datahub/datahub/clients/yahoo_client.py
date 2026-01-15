import yfinance as yf
import pandas as pd
from dataclasses import dataclass
from ..schemas import Interval

@dataclass
class YahooRequest:
    tickers: list[str]
    interval: Interval
    start: str #date must be in yyyy-mm-dd
    end: str

def get_raw_ohlcv(r: YahooRequest) -> pd.DataFrame:
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

if __name__ == "__main__":
    tickers = ["AAPL", "AMZN"]

    df = get_raw_ohlcv(r=YahooRequest(tickers, Interval.d1, "2024-11-01", "2024-12-01"))
    print(df.info())
    print(df.head())