"""
============================================================
  model.py
  ────────
  • AudioDataset  – PyTorch Dataset that lazily extracts
                    features on-the-fly (with augmentation
                    on the training split).
  • StressNet     – CNN + bi-LSTM classifier.

  Architecture overview (default hypers)
  ──────────────────────────────────────
  Input          →  (B, 1, 203, T)   1-channel "image"
  Conv block ×3  →  each: Conv2d → BN → ReLU → MaxPool
                     channels: 1 → 64 → 128 → 256
  AdaptiveAvg    →  (B, 256, 1, 8)   collapse freq axis
  Reshape        →  (B, 8, 256)      time × features
  BiLSTM         →  (B, 8, 128*2)    captures temporal order
  Global-avg     →  (B, 256)
  FC head        →  (B, 1)           sigmoid → P(stressed)
  ────────────────────────────────────────────────────────
  ~1.2 M parameters — light enough for a laptop GPU.
============================================================
"""

import numpy  as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from feature_extractor import extract_features


# ─── dataset ─────────────────────────────────────────────
class AudioDataset(Dataset):
    """
    Wraps the metadata DataFrame from data_loader.

    Parameters
    ----------
    df      : pd.DataFrame   – must have columns 'filepath' and 'stress'
    augment : bool           – True only for the training split
    """

    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df      = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row     = self.df.iloc[idx]
        features = extract_features(row["filepath"], augment=self.augment)
        # add channel dim  →  (1, C, T)
        x = torch.from_numpy(features).unsqueeze(0)
        y = torch.tensor(row["stress"], dtype=torch.float32)
        return x, y


# ─── model ───────────────────────────────────────────────
class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, pool: tuple = (2, 2)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        )

    def forward(self, x):
        return self.block(x)


class StressNet(nn.Module):
    """
    CNN feature extractor  →  Bi-LSTM temporal encoder  →  FC classifier.

    Parameters
    ----------
    n_features : int   – height of the input spectrogram  (C dimension)
    lstm_hidden: int   – hidden size of the bi-LSTM (default 128)
    dropout    : float – dropout probability (default 0.3)
    """

    def __init__(self, n_features: int = 203, lstm_hidden: int = 128, dropout: float = 0.3):
        super().__init__()

        # ── convolutional backbone ───────────────────────
        self.conv1 = ConvBlock(1,   64,  kernel=3, pool=(2, 2))
        self.conv2 = ConvBlock(64,  128, kernel=3, pool=(2, 2))
        self.conv3 = ConvBlock(128, 256, kernel=3, pool=(2, 1))   # keep time axis

        # after 3 blocks the freq axis is  n_features // 8  (due to two (2,2) + one (2,1) pools)
        freq_out = n_features // 8                                # works cleanly for 141 → 17

        # ── adaptive pooling → collapse freq to 1 ───────
        self.adapt_pool = nn.AdaptiveAvgPool2d((1, None))         # (B, 256, 1, T')

        # ── bi-LSTM ──────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size   = 256,
            hidden_size  = lstm_hidden,
            num_layers   = 2,
            batch_first  = True,
            bidirectional= True,
            dropout      = dropout,
        )

        # ── classifier head ──────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    # ── forward ──────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x  :  (B, 1, C, T)
        out:  (B, 1)   – raw logits  (apply sigmoid for probability)
        """
        # CNN
        x = self.conv1(x)                    # (B, 64,  C/2,  T/2)
        x = self.conv2(x)                    # (B, 128, C/4,  T/4)
        x = self.conv3(x)                    # (B, 256, C/8,  T/4)

        # collapse frequency axis
        x = self.adapt_pool(x)               # (B, 256, 1,    T')
        x = x.squeeze(2)                     # (B, 256, T')

        # LSTM expects  (B, T, features)
        x = x.permute(0, 2, 1)              # (B, T', 256)

        # Bi-LSTM
        x, _ = self.lstm(x)                  # (B, T', 256)

        # global average over time
        x = x.mean(dim=1)                    # (B, 256)

        # classifier
        x = self.head(x)                     # (B, 1)
        return x


# ── smoke-test ───────────────────────────────────────────
if __name__ == "__main__":
    model = StressNet(n_features=203)
    dummy = torch.randn(4, 1, 203, 300)      # batch=4
    out   = model(dummy)
    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")      # (4, 1)
    print(f"Parameters   : {sum(p.numel() for p in model.parameters()):,}")
