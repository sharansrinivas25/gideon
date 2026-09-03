"""The GPT model: embeddings, a stack of pre-norm blocks, and an LM head."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention
from .config import GPTConfig
from .kvcache import KVCache


class MLP(nn.Module):
    """Feed-forward: 4x expansion, GELU, back down (GPT-2 convention)."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """One pre-norm transformer block: x + f(norm(x))."""

    def __init__(self, config: GPTConfig, use_flash: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, use_flash=use_flash)
        self.ln2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, cache=None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cache=cache)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Decoder-only transformer."""

    def __init__(self, config: GPTConfig, use_flash: bool = False):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config, use_flash) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying: input embedding and output projection share a matrix
        self.tok_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # scale residual projections by 1/sqrt(2 * n_layer) so variance doesn't
        # grow with depth (GPT-2 paper, 2.3)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------ #
    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()  # tok_emb is tied to lm_head
        return n

    def forward(
        self,
        idx: torch.Tensor,                    # (B, T) token ids
        targets: torch.Tensor | None = None,  # (B, T) next-token labels
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        past_len = cache.length if cache is not None else 0
        if past_len + T > self.config.block_size:
            raise ValueError(
                f"sequence length {past_len + T} exceeds block_size {self.config.block_size}"
            )

        # positions continue from wherever the cache left off
        pos = torch.arange(past_len, past_len + T, device=idx.device)

        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for i, block in enumerate(self.blocks):
            x = block(x, cache=cache[i] if cache is not None else None)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
            return logits, loss

        # only the last position is needed to sample, so skip the rest
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    # ------------------------------------------------------------------ #
    def new_cache(self, batch_size: int = 1, device=None, dtype=torch.float32) -> KVCache:
        device = device or next(self.parameters()).device
        return KVCache(
            n_layer=self.config.n_layer,
            batch_size=batch_size,
            n_head=self.config.n_head,
            max_seq_len=self.config.block_size,
            head_dim=self.config.head_dim,
            device=device,
            dtype=dtype,
        )

    def configure_optimizer(self, weight_decay: float, lr: float, betas: tuple[float, float]):
        """AdamW, weight decay on 2-D params only (not biases or LayerNorm)."""
        decay, no_decay = [], []
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)
