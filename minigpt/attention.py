"""Multi-head causal self-attention, written out by hand.

The whole point of this file is that nothing here is ``nn.MultiheadAttention``
or ``nn.Transformer``. The QKV projection, the head split, the scaled dot
product, the causal mask and the output projection are all explicit, so the
shapes can be traced end to end.

Shape legend used throughout:
    B  batch
    T  query length for this call (T = full sequence when prefilling,
       T = 1 for each step of cached decoding)
    S  key/value length actually attended over (S = past + T)
    C  embedding dim (n_embd)
    nh number of heads
    hd head dim (C // nh)
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

    Parameters
    ----------
    config:
        Model shape.
    use_flash:
        If True and PyTorch exposes ``scaled_dot_product_attention``, use the
        fused kernel. The hand-written path is kept as the reference
        implementation and is what the tests check the fused path against.
    """

    def __init__(self, config: GPTConfig, use_flash: bool = False):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        self.dropout_p = config.dropout

        # One fused projection producing Q, K and V. Cheaper than three
        # separate matmuls because it is a single GEMM with better arithmetic
        # intensity; we split the result afterwards.
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.use_flash = use_flash and hasattr(F, "scaled_dot_product_attention")

        # Lower-triangular causal mask, registered as a buffer so it moves with
        # .to(device) and is saved/restored with the module but is not a
        # parameter. Shape (1, 1, block, block) broadcasts over B and nh.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size),
                             persistent=False)

    # ------------------------------------------------------------------ #
    # the actual attention maths
    # ------------------------------------------------------------------ #
    def _attend_manual(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, past_len: int
    ) -> torch.Tensor:
        """Reference implementation: softmax(QK^T / sqrt(d) + mask) V."""
        T, S = q.size(2), k.size(2)

        # (B, nh, T, hd) @ (B, nh, hd, S) -> (B, nh, T, S)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # Query i sits at absolute position past_len + i and may attend to any
        # key j <= past_len + i. Slicing the precomputed triangle with that
        # row offset gives exactly that, and costs nothing at decode time
        # (T = 1 means a single row of all-ones).
        mask = self.causal_mask[:, :, past_len : past_len + T, :S]
        att = att.masked_fill(mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        return att @ v  # (B, nh, T, hd)

    def _attend_flash(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, past_len: int
    ) -> torch.Tensor:
        """Fused kernel path (memory-efficient / flash attention)."""
        T, S = q.size(2), k.size(2)
        if past_len == 0 and T == S:
            # Plain causal case: let the kernel build the mask itself.
            return F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        # Cached case: is_causal would align the triangle to the wrong corner,
        # so pass the offset mask explicitly.
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

        # (B, T, C) -> (B, T, nh, hd) -> (B, nh, T, hd). The transpose puts the
        # head axis next to batch so the matmuls below are batched per head.
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

        # (B, nh, T, hd) -> (B, T, nh, hd) -> (B, T, C). contiguous() is needed
        # because transpose only changes strides and view demands a contiguous
        # buffer.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))
