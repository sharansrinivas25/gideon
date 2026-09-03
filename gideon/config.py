"""Model and training configuration.

Split so that inference only needs GPTConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 256      # maximum context length in tokens
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384          # must be divisible by n_head
    dropout: float = 0.2
    bias: bool = False         # LLaMA-style: no bias in Linear / LayerNorm

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    # optimisation
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # schedule
    max_iters: int = 3000
    warmup_iters: int = 100
    min_lr_ratio: float = 0.1  # cosine decays to learning_rate * this

    # batching
    batch_size: int = 32
    grad_accum_steps: int = 1

    # logging / eval
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 50

    seed: int = 1337

    def to_dict(self) -> dict:
        return asdict(self)
