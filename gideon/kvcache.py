"""Key/value cache for incremental decoding.

Pre-allocated to max_seq_len and written in place. Growing it with torch.cat
would reallocate and copy the whole cache every token.
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
        """Append k/v (B, nh, T_new, hd) and return the valid slice of the cache."""
        t_new = k.size(2)
        if self.length + t_new > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: {self.length} + {t_new} > {self.max_seq_len}"
            )
        start, end = self.length, self.length + t_new
        self.k[:, :, start:end] = k
        self.v[:, :, start:end] = v
        self.length = end
        return self.k[:, :, :end], self.v[:, :, :end]

    def evict_oldest(self, n: int) -> None:
        """Drop the n oldest positions, sliding the remainder to the front."""
        if n <= 0:
            return
        n = min(n, self.length)
        keep = self.length - n
        if keep > 0:
            # clone: overlapping copy on the same storage is undefined in torch
            self.k[:, :, :keep] = self.k[:, :, n : self.length].clone()
            self.v[:, :, :keep] = self.v[:, :, n : self.length].clone()
        self.length = keep

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

    def evict_oldest(self, n: int) -> None:
        for layer in self.layers:
            layer.evict_oldest(n)

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()

    def memory_bytes(self) -> int:
        return sum(layer.memory_bytes() for layer in self.layers)
