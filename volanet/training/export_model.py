import torch
from pathlib import Path

from volanet.model.lstm_vol import LSTMVolatility
from volanet.schemas import BuildConfig, FeatureConfig, LabelConfig, DatasetConfig

def main():
    ckpt_path = Path("artifacts/lstm_best.pt")
    bundle_path = Path("artifacts/vol_lstm_bundle.pt")
    device = torch.device("cpu")
    
    ckpt = torch.load(ckpt_path, map_location=device)
    
    model_kwargs = {
        "n_features": 9,         
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.1}
    
    model = LSTMVolatility(**model_kwargs)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    f_cfg = FeatureConfig()
    l_cfg = LabelConfig(target="realized_vol", horizon=5)
    d_cfg = DatasetConfig(lookback=60)
    cfg = BuildConfig(ticker="S&P500", 
                      interval="1d", 
                      adjusted=True, 
                      as_of=None, 
                      features=f_cfg,
                      label=l_cfg,
                      dataset=d_cfg)
    
    save_dict = {
        "model_state_dict": model.state_dict(),
        "model_kwargs": model_kwargs,
        "build_config": cfg.model_dump(),
        "best_val": ckpt["best_val"],
        "best_epoc": ckpt["best_epoch"]
    }
    torch.save(save_dict, bundle_path)


if __name__ == "__main__":
    main()