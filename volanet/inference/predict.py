from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch

from volanet.model.lstm_vol import LSTMVolatility
from volanet.dataset.build_dataset import build_inference_dataset
from volanet.schemas import InferenceRequest, BuildConfig, PredictionResponse, VolatilityPrediction

def load_model_bundle(bundle_path: str | Path, device: torch.device) -> tuple[LSTMVolatility, dict]:
    """
    Loads a bundled checkpoint that contains: 
        - model_state_dict
        - model_kwargs
        - build_config
    """
    bundle_path = Path(bundle_path)
    ckpt = torch.load(bundle_path, map_location=device)
    
    model_kwargs = ckpt.get("model_kwargs")
    if model_kwargs is None:
        raise ValueError("Bundle missing 'model_kwargs'.")
    
    model = LSTMVolatility(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    return model, ckpt    
    

@torch.no_grad()
def predict(
    ohlcv_df: pd.DataFrame,
    ticker: str,
    bundle_path: str | Path = "artifacts/vol_lstm_bundle.pt",
    as_of: Optional[datetime] = None,
    device: Optional[torch.device] = None,
) -> PredictionResponse:
    """
    Minimal inference:
      DF -> features -> last lookback window -> model -> PredictionResponse
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) load model + config snapshot
    model, ckpt = load_model_bundle(bundle_path, device)

    # build_config should be inside bundle (recommended)
    build_cfg_dict = ckpt.get("build_config")
    if build_cfg_dict is None:
        raise ValueError("Bundle missing 'build_config'. Include cfg.model_dump() when saving the bundle.")
    build_cfg = BuildConfig(**build_cfg_dict)

    # 2) construct inference request (reuses your existing dataset code)
    req = InferenceRequest(
        ticker=ticker,
        interval=build_cfg.interval,
        adjusted=build_cfg.adjusted,
        ohlcv_df=ohlcv_df,    
        features=build_cfg.features,
        dataset=build_cfg.dataset,
        model_version=ckpt.get("model_version", "v1"),
        as_of=as_of,
    )

    # 3) build X with shape (1, lookback, F)
    seq_ds = build_inference_dataset(req)
    X = torch.from_numpy(seq_ds.X).float().to(device)  # (1, T, F)

    # 4) forward pass -> scalar
    y_hat = float(model(X).squeeze(0).detach().cpu().item())

    # 5) pack response
    pred = VolatilityPrediction(
        ticker=ticker,
        interval=req.interval,
        model_version=req.model_version,
        as_of=pd.to_datetime(seq_ds.end_timestamps[0]).to_pydatetime(),
        horizon=build_cfg.label.horizon,
        y_hat=y_hat,
        y_hat_annulaized=None,  # you can add later in postprocess
        lookback=build_cfg.dataset.lookback,
        num_feautures=X.shape[-1],
        feature_set=build_cfg.features.feature_set,
    )

    return PredictionResponse(prediction=pred, warnings=[], metadata={})


def test():
    import pandas as pd
    from volanet.inference.predict import predict

    df = pd.read_parquet("data/ohlcv_parquet/adjusted=true/AAPL/Interval.d1.parquet")

    resp = predict(df, ticker="AAPL", bundle_path="artifacts/vol_lstm_bundle.pt")
    print(resp.prediction.y_hat)
    print(resp.prediction.as_of)

if __name__ == "__main__":
    test()
