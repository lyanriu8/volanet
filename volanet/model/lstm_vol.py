import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMVolatility(nn.Module):
    """Input: (B, T, F) → Output: (B,)"""
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        out_activation: str = "softplus",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.out_activation = out_activation
        self.eps = eps

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def _activate(self, y: torch.Tensor) -> torch.Tensor:
        if self.out_activation == "softplus":
            return F.softplus(y) + self.eps
        if self.out_activation == "exp":
            return torch.exp(y).clamp_min(self.eps)
        if self.out_activation == "none":
            return y
        raise ValueError(f"Unknown out_activation={self.out_activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)           # h_n: (num_layers, B, hidden)
        h_last = h_n[-1]                      # (B, hidden)
        y = self.fc(self.norm(h_last)).squeeze(-1)  # (B,)
        return self._activate(y)
