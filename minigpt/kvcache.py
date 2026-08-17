"""Key/value cache for incremental decoding.

Why this exists
---------------
A decoder-only transformer is causal: token *t* attends only to tokens
``<= t``. So when we generate token by token, the keys and values computed for
tokens ``0..t-1`` are **exactly the same** on every subsequent step. Recomputing
them turns generation into O(n^2) work per token and O(n^3) for the whole
sequence. Caching them makes each step O(n) and the sequence O(n^2).

Implementation note
-------------------
The cache is **pre-allocated** to ``max_seq_len`` and filled in place, rather
than ``torch.cat``-ing a new tensor each step. Concatenation reallocates and
copies the whole cache on every token, which quietly reintroduces the quadratic
memory traffic we were trying to remove. Pre-allocation is what real serving
stacks (vLLM, TensorRT-LLM) do, modulo paging.
"""

from __future__ import annotations

import torch


class LayerKVCache:
    """Cache of keys and values for a single attention layer."""

    def __init__(
        self,
        batch_size: int,
        n_head: int,
        max_seq_len: int,
        head_dim: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        shape = (batch_size, n_head, max_seq_len, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.max_seq_len = max_seq_len
        self.length = 0  # number of valid positions currently stored

    def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append ``k``/``v`` (B, nh, T_new, hd) and return the full valid cache.

        Returns views of length ``self.length`` so the caller attends over
        everything seen so far, not over the zero-padded tail.
        """
        t_new = k.size(2)
        if self.length + t_new > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: {self.length} + {t_new} > {self.max_seq_len}. "
                "Increase block_size or truncate the context."
            )
        start, end = self.length, self.length + t_new
        self.k[:, :, start:end] = k
        self.v[:, :, start:end] = v
        self.length = end
        return self.k[:, :, :end], self.v[:, :, :end]

    def reset(self) -> None:
        self.length = 0

    def memory_bytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2


class KVCache:
    """One :class:`LayerKVCache` per transformer layer."""

    def __init__(
        self,
        n_layer: int,
        batch_size: int,
        n_head: int,
        max_seq_len: int,
        head_dim: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.layers = [
            LayerKVCache(batch_size, n_head, max_seq_len, head_dim, device, dtype)
            for _ in range(n_layer)
        ]

    def __getitem__(self, i: int) -> LayerKVCache:
        return self.layers[i]

    def __len__(self) -> int:
        return len(self.layers)

    @property
    def length(self) -> int:
        return self.layers[0].length

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()

    def memory_bytes(self) -> int:
        return sum(layer.memory_bytes() for layer in self.layers)
