from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from datahub.schemas import OHLCVQuery, Interval
from datahub.facade import get_history  # <-- wherever your get_history lives
from datahub.store.parquet_store import ParquetStoreConfig, upsert_ohlcv


TICKERS = [
    # start with 50-ish liquid names; replace with your own universe loader later
    "AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","JPM","V","MA",
    "UNH","XOM","COST","HD","LLY","ABBV","KO","PEP","AVGO","ORCL",
    "WMT","PG","BAC","NFLX","CRM","ADBE","CSCO","INTC","TMO","ABT",
    "NKE","MCD","QCOM","DHR","LIN","AMD","AMAT","DIS","TXN","UPS",
    "PM","LOW","IBM","GS","CAT","GE","NOW","SPGI","ISRG","BKNG",
]


def main() -> None:
    cfg = ParquetStoreConfig(root_dir=Path("data/ohlcv_parquet"))

    # choose your history span
    start = datetime(2015, 1, 1, tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)

    interval = Interval.d1
    adjusted = True  # pick one; keep it consistent

    for t in TICKERS:
        try:
            dh = get_history(
                OHLCVQuery(
                    ticker=t,
                    interval=interval,
                    start=start,
                    end=end,
                    adjusted=adjusted,
                )
            )
            # dh.df is normalized already in your get_history
            path = upsert_ohlcv(
                cfg,
                ticker=t,
                interval=str(interval),  # Interval.d1 -> use your enum's string form
                adjusted=adjusted,
                df_new=dh.df,
            )
            print(f"[OK] {t} -> {path} rows={len(dh.df)}")

        except Exception as e:
            print(f"[FAIL] {t}: {e}")


if __name__ == "__main__":
    main()
