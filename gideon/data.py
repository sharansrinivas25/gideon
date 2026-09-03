"""Dataset handling. Corpus fits in memory; batches are random windows."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import torch

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


def download_tinyshakespeare(path: str | Path = "data/tinyshakespeare.txt") -> Path:
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(TINY_SHAKESPEARE_URL, timeout=60) as r:
        path.write_bytes(r.read())
    return path


class CharDataset:
    """Holds the encoded corpus and serves random batches."""

    def __init__(self, text: str, tokenizer, block_size: int, split_ratio: float = 0.9):
        self.tokenizer = tokenizer
        self.block_size = block_size
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        n = int(len(data) * split_ratio)
        # split by position, not randomly, or val windows leak into train
        self.train_data = data[:n]
        self.val_data = data[n:]

    def __repr__(self) -> str:
        return (
            f"CharDataset(train={len(self.train_data):,} tokens, "
            f"val={len(self.val_data):,} tokens, vocab={self.tokenizer.vocab_size})"
        )

    def get_batch(
        self,
        split: str,
        batch_size: int,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(
            len(data) - self.block_size - 1, (batch_size,), generator=generator
        )
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        # y is x shifted by one
        y = torch.stack([data[i + 1 : i + 1 + self.block_size] for i in ix])
        return x.to(device), y.to(device)
