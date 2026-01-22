# volanet/dataset/build_labels.py
from __future__ import annotations

import numpy as np
import pandas as pd

from volanet.schemas import LabelConfig


def build_labels(df: pd.DataFrame, cfg: LabelConfig) -> pd.Series:
    """
    Build supervised labels from a feature-engineered dataframe.

    Input:
        df: feature dataframe containing at least:
            - timestamp
            - log_return

    Output:
        pd.Series:
            index aligned to df.index
            name = cfg.target
            value at time t = realized volatility over (t+1 ... t+horizon)

    IMPORTANT:
        - Uses *future* returns (shift(-1))
        - Does NOT drop NaNs
        - Dataset builder is responsible for alignment + dropping rows
    """

    if "log_return" not in df.columns:
        raise ValueError("build_labels requires 'log_return' column in dataframe.")
    
    # realized volatility over future window
    y = (
        df["log_return"]
        .shift(-1)
        .rolling(window=cfg.horizon)
        .apply(lambda x: float(np.sqrt(np.sum(np.square(x)))), raw=True)
    )

    y = y.rename(cfg.target)

    # optional annualization (kept explicit, not mixed with target)
    if cfg.annualize:
        ann = np.sqrt(cfg.trading_days / cfg.horizon)
        y = y * ann

    return y
