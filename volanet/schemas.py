from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal ,Optional , Sequence

import pandas as pd 
import numpy as np
from pydantic import BaseModel, Field, model_validator, ConfigDict

Interval = Literal["1d", "1h", "30m", "15m", "5m", "1m"]




# ------ Feature Engineering Specs --------

class FeatureConfig(BaseModel):
    """Determines features what to build from canonical OHLCV from datahub"""
    
    feature_set: str = Field(default="basic_v1")
    windows: Sequence[int] = Field(default=(5, 20, 60))
    use_log_returns: bool = True # use log to comptute returns
    include_gaps: bool = True # compares how current open comapres to day befores close
    include_volume: bool = True
    include_ranges: bool = True
    drop_na_row: bool = True # drop rows until all rolling windows are all valid
    
    
class LabelConfig(BaseModel):
    """Determines the supervised target construction 
    -> the correct answer to what the model is trying to predict
    -> realized_vol = sqrt(sum future returns^2 over horizon)"""
    
    target: Literal["realized_vol"] = "realized_vol"
    horizon: int = 5
    annualize: bool = False
    trading_days: int = 252
    
    @model_validator(mode="after")
    def _validate(self) -> "LabelConfig":
        if self.horizon <= 0:
            raise ValueError("horion must be > 0")
        if self.annualize and self.trading_days <= 0:
            raise ValueError("trading_days mys be > 0")
        return self

class DatasetConfig(BaseModel):
    """Determines how to turn feature + label to LSTM samples"""
    lookback: int = 60
    stride: int = 1
    split: tuple[float, float, float] = (0.7, 0.15, 0.15) #fraction of data split into train/val/test
    
    @model_validator(mode="after")
    def _validate(self) -> "DatasetConfig":
        if self.lookback <= 1:
            raise ValueError("lookback must be > 1")
        if self.stride <= 0:
            raise ValueError("split must sum to 1.0")
        return self

class BuildConfig(BaseModel):
    """Full spec to build dataset from canonical OHLCV"""
    ticker: str
    interval: Interval = "1d"
    adjusted: bool = True
    
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    label: LabelConfig = Field(default_factory=LabelConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    
    as_of: Optional[datetime] = None

@dataclass(frozen=True)
class FeatureFrame:
    """"A feature dataframe aligned with timestamps so that model knows where cols with 
    features are
    -> Output of feature engineering"""
    
    df: pd.DataFrame
    feature_cols: list[str]
    
@dataclass(frozen=True)
class SupervisedFeatureFrame:
    """A feature + label dataframe aligned with timestamps so that the model knows where features + labels are
    -> Output of labeling"""
    
    df: pd.DataFrame
    feature_cols: list[str]
    label_cols: str

@dataclass(frozen=True)
class SequenceDataset:
    """A sequence dataset for LSTM-style models."""
    X: np.ndarray                 # (N, lookback, n_features)
    y: Optional[np.ndarray]       # (N,) for training; None for inference
    end_timestamps: np.ndarray    # (N,) timestamps aligned to each sample (sequence end)


@dataclass(frozen=True)
class SplitDatasets:
    train: SequenceDataset
    val: SequenceDataset
    test: SequenceDataset
    manifest: DatasetManifest
    label_col: str


# --------- Inference Request ----------

class InferenceRequest(BaseModel):
    """Inference request made to the model from API or CLI side"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    ticker: str
    interval: Interval = "1d"
    adjusted: bool = True
    
    ohlcv_df: pd.DataFrame
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    
    model_version: str = "v1"
    as_of: Optional[datetime] = None
    



# ---------- Model Output -----------

class VolatilityPrediction(BaseModel):
    """Volatility prediction model"""
    
    ticker: str
    interval: Interval
    model_version: str
    
    as_of: datetime      # datetime of last bar used
    horizon: int         # predicted horizon (from label config)
    
    y_hat: float         # predicted volatility :))))
    y_hat_annulaized: Optional[float] = None
    
    # diagnostics
    lookback: int
    num_feautures: int
    feature_set: str

class PredictionResponse(BaseModel):
    """Response Model 
    -> can add more prediction values in the future"""
    
    prediction: VolatilityPrediction
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)




# --------- Artifact/Manifest Schemas ---------

class DatasetManifest(BaseModel):
    """Dataset manifest
    -> how to recover previously used dataset"""
    
    dataset_version: str = "v1"
    ticker: str
    interval: Interval
    adjusted: bool
    feature_config: FeatureConfig
    label_config: LabelConfig
    
    feature_cols: list[str]
    label_col: str
    
class ModelMannifest(BaseModel):
    """LSTM mannifest
    -> how to recover model settings"""
    
    model_version: str = "v1"
    created_at: datetime
    
    dataset_mannifest: DatasetManifest
    
    model_type: Literal["lstm"] = "lstm"
    hidden_size: int
    num_layers: int
    dropout: float
    
    scaler_type: Literal["standard"] = "standard"
    scalar_path: str
    
    weights_path: str
