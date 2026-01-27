# models/lstm_vol.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMVolatility(nn.Module):
    """
    Predicts a single scalar volatility value from a sequence of features.

    Input:  x shape (B, T, F)
    Output: y_hat shape (B,)  (positive)
    """
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        out_activation: str = "softplus"
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_activation = out_activation

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        lstm_out_dim = hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Linear(lstm_out_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, (h_n, c_n) = self.lstm(x)

        # Take last layer’s hidden state
        h_last = h_n[-1]

        y = self.head(h_last).squeeze(-1)  # (B,)

        # Enforce positive volatility if desired
        if self.out_activation == "softplus":
            y = F.softplus(y) + 1e-8
        elif self.out_activation == "exp":
            y = torch.exp(y).clamp_min(1e-8)
        elif self.out_activation == "none":
            pass
        else:
            raise ValueError(f"Unknown out_activation={self.out_activation}")

        return y
