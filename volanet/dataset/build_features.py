import numpy as np
import pandas as pd

from volanet.schemas import FeatureConfig, FeatureFrame


def build_features(df: pd.DataFrame, cfg: FeatureConfig):
    """
    input: canonical ohlcv dataframe 
    -> [timestamp, open, high, low, close, volume]
    
    output: feature engineered dataframe
    -> [log_return, rv_5, rv_10, rv_20, hl_range, oc_return, log_volume, vol_change]
    """
    df = df.copy()
    feature_cols: list[str] = []
    
    if cfg.use_log_returns:
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    else:
        raise ValueError("unimplemented must set log returns to true")
    
    feature_cols.append("log_return")

    # rolling realized volatility
    for w in cfg.windows:
        w = int(w)
        col = f"rv{w}"
        df[col] = df["log_return"].rolling(w).std()
        feature_cols.append(col)
    
    if cfg.include_gaps:
        df["gaps"] = df["open"] / df["close"].shift(1) - 1.0
        feature_cols.append("gaps")
           
    # range based features
    if cfg.include_ranges:
        df["hl_range"] = (df["high"] - df["low"]) / df["close"]
        df["oc_return"] = (df["close"] - df["open"]) / df["open"]
        feature_cols.append("hl_range")
        feature_cols.append("oc_return")
        
    # volume dynamics
    if cfg.include_volume:
        df["log_volume"] = np.log(df["volume"] + 1.0)
        df["vol_change"] = df["log_volume"].diff()
        feature_cols.append("log_volume")
        feature_cols.append("vol_change")
    
    if cfg.drop_na_row:
        df = df.dropna(subset=feature_cols)
    
    return FeatureFrame(df=df, feature_cols=feature_cols)
