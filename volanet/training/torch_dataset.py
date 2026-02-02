import torch
from torch.utils.data import Dataset

class TorchSeqDataset(Dataset):
    def __init__(self, seq_ds):
        self.X = torch.from_numpy(seq_ds.X).float()
        self.y = None if seq_ds.y is None else torch.from_numpy(seq_ds.y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]
