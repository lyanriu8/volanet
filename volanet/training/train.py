# volanet/training/train_lstm.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

import pandas as pd

from volanet.dataset.build_dataset import build_training_splits
from volanet.model.lstm_vol import LSTMVolatility
from volanet.training.torch_dataset import TorchSeqDataset
from volanet.schemas import BuildConfig, FeatureConfig, LabelConfig, DatasetConfig

def _step(
    model: nn.Module,
    batch,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """
    Runs one train OR eval step.
    Returns scalar loss (float).
    """
    xb, yb = batch
    xb = xb.to(device)
    yb = yb.to(device)

    yhat = model(xb)               # (B,)
    loss = criterion(yhat, yb)     # scalar

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return float(loss.detach().cpu().item())


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.eval()
    losses = []
    for batch in loader:
        loss = _step(model, batch, device, criterion, optimizer=None)
        losses.append(loss)
    return sum(losses) / max(1, len(losses))


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float = 1e-3,
    epochs: int = 20,
    weight_decay: float = 0.0,
    grad_clip: float | None = 1.0,
    save_path: str | Path | None = None,
) -> Dict[str, float]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.to(device)
    best_val = float("inf")
    best_epoch = -1

    for epoch in range(1, epochs + 1):          # how many times the model has seem the same data
        model.train()
        train_losses = []

        for batch in train_loader:
            xb, yb = batch
            xb = xb.to(device)
            yb = yb.to(device)

            yhat = model(xb)
            loss = criterion(yhat, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        train_loss = sum(train_losses) / max(1, len(train_losses))  # avg loss
        val_loss = evaluate(model, val_loader, device, criterion)   # evaluate

        print(f"epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            if save_path is not None:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "best_val": best_val,
                        "best_epoch": best_epoch,
                    },
                    save_path,
                )

    return {"best_val": best_val, "best_epoch": float(best_epoch)}


def main():
    
    base = Path("data/ohlcv_parquet/adjusted=true")
    train_sets = []
    val_sets = []
    test_sets = []
    
    for ticker_dir in base.iterdir():
        if not ticker_dir.is_dir():
            continue
        
        parquet_path = ticker_dir / "Interval.d1.parquet"
        if not parquet_path.exists():
            continue
        
        ticker = ticker_dir.name
        print(f"loading {ticker}")
        df = pd.read_parquet(parquet_path)
        
        f_cfg = FeatureConfig()
        l_cfg = LabelConfig(target="realized_vol", horizon=5)
        d_cfg = DatasetConfig(lookback=60)
        cfg = BuildConfig(ticker=ticker, 
                          interval="1d", 
                          adjusted=True, 
                          as_of=None, 
                          features=f_cfg,
                          label=l_cfg,
                          dataset=d_cfg)
        splits = build_training_splits(df, cfg)
        train_sets.append(TorchSeqDataset(splits.train))
        val_sets.append(TorchSeqDataset(splits.val))
        test_sets.append(TorchSeqDataset(splits.test))
    
    if not train_sets:
        raise RuntimeError("No datasets built — check path/filename/dir logic.")

    train_ds = ConcatDataset(train_sets)
    val_ds = ConcatDataset(val_sets)
    test_ds = ConcatDataset(test_sets)
    
    print("train samples:", len(train_ds), "val:", len(val_ds), "test:", len(test_ds))

    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    
    first_X, _ = train_sets[0][0]     # train_sets is a list of TorchSeqDataset
    n_features = first_X.shape[-1]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMVolatility(n_features=n_features)
    
    save_path = "artifacts/lstm_best.pt"

    train(model=model, 
          train_loader=train_loader, 
          val_loader=val_loader,
          device=device, 
          epochs=20,
          save_path=save_path)
    
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    test_loss = evaluate(model, test_loader, device, nn.MSELoss())
    print("test loss:", test_loss)


if __name__ == "__main__":
    main()
