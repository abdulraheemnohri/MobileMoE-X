"""Simple text-file dataset for next-byte prediction."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class ByteTextDataset(Dataset):
    def __init__(self, path: str | Path, context_length: int):
        raw = Path(path).read_bytes()
        if len(raw) < context_length + 1:
            raise ValueError(
                f"training text needs at least {context_length + 1} bytes; got {len(raw)}"
            )
        self.tokens = torch.tensor(list(raw), dtype=torch.long)
        self.context_length = context_length

    def __len__(self) -> int:
        return self.tokens.numel() - self.context_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[index:index + self.context_length + 1]
        return chunk[:-1], chunk[1:]
