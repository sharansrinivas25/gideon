"""Multi-head causal self-attention.

Shapes:  B batch, T query length, S key/value length (past + T),
C embedding dim, nh heads, hd head dim.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig
from .kvcache import LayerKVCache


class CausalSelfAttention(nn.Module):
    """Masked multi-head self-attention with optional KV caching.

    use_flash switches to torch's fused kernel; the manual path is the
    reference the tests check it against.
    """

    def __init__(self, config: GPTConfig, use_flash: bool = False):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        self.dropout_p = config.dropout

        # one fused GEMM for Q, K, V; split afterwards
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.use_flash = use_flash and hasattr(F, "scaled_dot_product_attention")

        # (1, 1, block, block) so it broadcasts over batch and heads
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size),
                             persistent=False)

    def _attend_manual(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, past_len: int
    ) -> torch.Tensor:
        """softmax(QK^T / sqrt(d) + mask) @ V."""
        T, S = q.size(2), k.size(2)

        # (B, nh, T, hd) @ (B, nh, hd, S) -> (B, nh, T, S)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # query i sits at absolute position past_len + i, so the triangle
        # needs slicing with that row offset
        mask = self.causal_mask[:, :, past_len : past_len + T, :S]
        att = att.masked_fill(mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        return att @ v  # (B, nh, T, hd)

    def _attend_flash(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, past_len: int
    ) -> torch.Tensor:
        """Fused kernel path."""
        T, S = q.size(2), k.size(2)
        if past_len == 0 and T == S:
            return F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        # is_causal aligns the triangle to the wrong corner when cached
        mask = self.causal_mask[:, :, past_len : past_len + T, :S].bool()
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,                       # (B, T, C)
        cache: LayerKVCache | None = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x)                                   # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)             # 3 x (B, T, C)

        # (B, T, C) -> (B, nh, T, hd), so the matmuls batch per head
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        past_len = 0
        if cache is not None:
            past_len = cache.length
            k, v = cache.update(k, v)   # k, v now cover 0..past_len+T

        if self.use_flash:
            y = self._attend_flash(q, k, v, past_len)
        else:
            y = self._attend_manual(q, k, v, past_len)

        # back to (B, T, C); contiguous because view needs it after transpose
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))
