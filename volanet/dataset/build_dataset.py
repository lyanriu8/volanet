# volanet/dataset/build_dataset.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from volanet.schemas import (
    BuildConfig,
    DatasetConfig,
    DatasetManifest,
    FeatureFrame,
    SupervisedFeatureFrame,
    InferenceRequest,
    SequenceDataset,
    SplitDatasets
)

from volanet.dataset.build_features import build_features
from volanet.dataset.build_labels import build_labels

# -----------------------------
# Helpers
# -----------------------------

def _ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise ValueError("Expected a 'timestamp' column in canonical/feature dataframe.")
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=False, errors="coerce")
    if out["timestamp"].isna().any():
        bad = out[out["timestamp"].isna()]
        raise ValueError(f"Some 'timestamp' values could not be parsed. Example rows:\n{bad.head(3)}")
    return out


def _apply_as_of(df: pd.DataFrame, as_of: Optional[datetime]) -> pd.DataFrame:
    """Cut the dataframe at as_of (inclusive)."""
    if as_of is None:
        return df
    ts = pd.to_datetime(as_of)
    return df[df["timestamp"] <= ts].copy()


def _time_split(
    ds: SequenceDataset,
    split: Tuple[float, float, float],
) -> tuple[SequenceDataset, SequenceDataset, SequenceDataset]:
    a, b, c = split
    if not np.isclose(a + b + c, 1.0, atol=1e-6):
        raise ValueError(f"split must sum to 1.0, got {split}")

    n = ds.X.shape[0]
    if n == 0:
        raise ValueError("No sequences were created (n=0). Check lookback/stride and NA dropping.")

    n_train = int(n * a)
    n_val = int(n * b)

    if n_train == 0 and n >= 1:
        n_train = 1
    if n_val == 0 and n - n_train >= 2:
        n_val = 1

    def _slice(i0: int, i1: int) -> SequenceDataset:
        X = ds.X[i0:i1]
        y = None if ds.y is None else ds.y[i0:i1]
        t = ds.end_timestamps[i0:i1]
        return SequenceDataset(X=X, y=y, end_timestamps=t)

    train = _slice(0, n_train)
    val = _slice(n_train, n_train + n_val)
    test = _slice(n_train + n_val, n)
    return train, val, test


def _build_supervised_frame(
    canonical_ohlcv: pd.DataFrame,
    cfg: BuildConfig,
) -> SupervisedFeatureFrame:
    base = _ensure_timestamp(canonical_ohlcv)
    base = _apply_as_of(base, cfg.as_of)
    base = base.sort_values("timestamp").reset_index(drop=True)

    ff: FeatureFrame = build_features(base, cfg.features)

    # build_labels returns Series
    y = build_labels(ff.df, cfg.label)
    label_col = cfg.label.target
    if y.name != label_col:
        y = y.rename(label_col)

    out = ff.df.copy()
    out[label_col] = y

    keep_cols = ["timestamp"] + list(ff.feature_cols) + [label_col]
    missing = [c for c in keep_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Missing expected columns after feature/label build: {missing}")

    out = out.dropna(subset=list(ff.feature_cols) + [label_col]).copy()
    out = out.sort_values("timestamp").reset_index(drop=True)

    return SupervisedFeatureFrame(df=out, feature_cols=ff.feature_cols, label_cols=label_col)


def _to_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: Optional[str],
    ds_cfg: DatasetConfig,
) -> SequenceDataset:
    lookback = int(ds_cfg.lookback)
    stride = int(ds_cfg.stride)

    if lookback <= 1:
        raise ValueError("lookback must be > 1")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if "timestamp" not in df.columns:
        raise ValueError("df must contain a 'timestamp' column.")
    if len(df) < lookback:
        raise ValueError(f"Not enough rows ({len(df)}) to build lookback={lookback} sequences.")

    Xv = df[feature_cols].to_numpy(dtype=np.float32)
    ts = pd.to_datetime(df["timestamp"]).to_numpy()

    yv = None
    if label_col is not None:
        if label_col not in df.columns:
            raise ValueError(f"label_col '{label_col}' not found in df.")
        yv = df[label_col].to_numpy(dtype=np.float32)

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    t_list: list[datetime] = []

    for i in range(lookback - 1, len(df), stride):
        X_list.append(Xv[i - lookback + 1 : i + 1])
        t_list.append(ts[i])
        if yv is not None:
            y_list.append(float(yv[i]))

    if not X_list:
        raise ValueError("No sequences created. Check lookback/stride and NA dropping.")

    X = np.stack(X_list, axis=0)
    end_ts = np.array(t_list)
    y = None if yv is None else np.array(y_list, dtype=np.float32)
    return SequenceDataset(X=X, y=y, end_timestamps=end_ts)


# -----------------------------
# Public API
# -----------------------------

def build_training_splits(
    canonical_ohlcv: pd.DataFrame,
    cfg: BuildConfig,
) -> SplitDatasets:
    """
    _build_supervised_frame -> creates a single dataframe that contains feature cols and label cols
    
    """
    sf = _build_supervised_frame(canonical_ohlcv, cfg)
    label_col = sf.label_cols

    seq_ds = _to_sequences(sf.df, sf.feature_cols, label_col, cfg.dataset)
    train, val, test = _time_split(seq_ds, cfg.dataset.split)

    manifest = DatasetManifest(
        dataset_version="v1",
        ticker=cfg.ticker,
        interval=cfg.interval,
        adjusted=cfg.adjusted,
        feature_config=cfg.features,
        label_config=cfg.label,
        feature_cols=sf.feature_cols,
        label_col=label_col,
    )

    return SplitDatasets(
        train=train,
        val=val,
        test=test,
        manifest=manifest,
        label_col=label_col,
    )


def build_inference_dataset(req: InferenceRequest) -> SequenceDataset:
    """
    Build a single inference sample (X only), without labels.

    Uses req.as_of to ensure we only use data available up to that timestamp.
    Input df must be canonical OHLCV: [timestamp, open, high, low, close, volume].
    """
    base = _ensure_timestamp(req.ohlcv_df)
    base = _apply_as_of(base, req.as_of)
    base = base.sort_values("timestamp").reset_index(drop=True)

    ff: FeatureFrame = build_features(base, req.features)

    df_feat = _ensure_timestamp(ff.df).sort_values("timestamp").reset_index(drop=True)

    lookback = int(req.dataset.lookback)
    if len(df_feat) < lookback:
        raise ValueError(f"Not enough feature rows ({len(df_feat)}) for lookback={lookback}.")

    df_last = df_feat.iloc[-lookback:].copy().reset_index(drop=True)

    Xv = df_last[ff.feature_cols].to_numpy(dtype=np.float32)
    X = Xv[None, :, :]  # (1, lookback, n_features)

    end_ts = np.array([pd.to_datetime(df_last["timestamp"].iloc[-1]).to_numpy()])

    return SequenceDataset(X=X, y=None, end_timestamps=end_ts)

    