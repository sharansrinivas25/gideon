"""Checkpoints. Config and tokeniser are stored with the weights."""

from __future__ import annotations

from pathlib import Path

import torch

from .config import GPTConfig
from .model import GPT
from .tokenizer import BPETokenizer, CharTokenizer


def save_checkpoint(
    path: str | Path,
    model: GPT,
    tokenizer,
    iter_num: int = 0,
    val_loss: float | None = None,
    history: list | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(tokenizer, CharTokenizer):
        tok = {"type": "char", "chars": tokenizer.chars}
    elif isinstance(tokenizer, BPETokenizer):
        tok = {"type": "bpe", "merges": [[a, b, n] for (a, b), n in tokenizer.merges.items()]}
    else:
        raise TypeError(f"unsupported tokenizer: {type(tokenizer)}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": model.config.to_dict(),
            "tokenizer": tok,
            "iter_num": iter_num,
            "val_loss": val_loss,
            "history": history or [],
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str = "cpu", use_flash: bool = False):
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)

    tok = ckpt["tokenizer"]
    if tok["type"] == "char":
        tokenizer = CharTokenizer(tok["chars"])
    else:
        tokenizer = BPETokenizer({(a, b): n for a, b, n in tok["merges"]})

    config = GPTConfig(**ckpt["config"])
    model = GPT(config, use_flash=use_flash)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, tokenizer
