"""Sliding-window eviction: the mechanism that keeps long generations cheap.

Equivalence with the naive decoder only holds *within* ``block_size``; past it
the rolling cache is a documented approximation (surviving entries are
re-indexed to the front of the window). These tests pin down what must still be
true: the eviction moves the right data, the cache never overflows, and long
generations stay finite and in-vocabulary.
"""

import time

import pytest
import torch

from minigpt.config import GPTConfig
from minigpt.generate import generate, generate_naive
from minigpt.kvcache import LayerKVCache
from minigpt.model import GPT


@pytest.fixture
def config():
    return GPTConfig(vocab_size=32, block_size=32, n_layer=2, n_head=4, n_embd=32, dropout=0.0)


def test_eviction_keeps_the_newest_entries():
    cache = LayerKVCache(batch_size=1, n_head=2, max_seq_len=8, head_dim=4)
    k = torch.randn(1, 2, 6, 4)
    cache.update(k, torch.randn(1, 2, 6, 4))

    cache.evict_oldest(2)
    assert cache.length == 4
    # what remains must be the last four of the original six, in order
    torch.testing.assert_close(cache.k[:, :, :4], k[:, :, 2:6])


def test_eviction_frees_room_to_append():
    cache = LayerKVCache(batch_size=1, n_head=2, max_seq_len=4, head_dim=4)
    cache.update(torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4))
    cache.evict_oldest(2)
    new_k = torch.randn(1, 2, 1, 4)
    k, _ = cache.update(new_k, torch.randn(1, 2, 1, 4))
    assert cache.length == 3
    torch.testing.assert_close(k[:, :, 2:], new_k)


def test_evicting_everything_is_a_reset():
    cache = LayerKVCache(batch_size=1, n_head=2, max_seq_len=8, head_dim=4)
    cache.update(torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4))
    cache.evict_oldest(99)
    assert cache.length == 0


def test_evict_zero_is_a_noop():
    cache = LayerKVCache(batch_size=1, n_head=2, max_seq_len=8, head_dim=4)
    k = torch.randn(1, 2, 3, 4)
    cache.update(k, torch.randn(1, 2, 3, 4))
    cache.evict_oldest(0)
    assert cache.length == 3
    torch.testing.assert_close(cache.k[:, :, :3], k)


def test_long_generation_stays_valid(config):
    """Four times the context window must not overflow, NaN, or leave the vocab."""
    torch.manual_seed(0)
    model = GPT(config).eval()
    prompt = torch.zeros((1, 1), dtype=torch.long)
    n = config.block_size * 4
    out = generate(model, prompt, n, temperature=0.8, top_k=10)
    assert out.shape == (1, n + 1)
    assert out.min() >= 0 and out.max() < config.vocab_size


def test_long_generation_is_faster_than_naive():
    """The whole point: past the window the cache must still beat recomputation.

    Before sliding-window eviction this regressed - the cache was rebuilt from
    scratch every step once full, costing a full forward pass per token.

    Uses a larger model than the other tests on purpose: at toy sizes per-step
    Python overhead dominates the matmuls and the comparison measures the
    interpreter rather than the algorithm.
    """
    torch.manual_seed(0)
    big = GPTConfig(vocab_size=32, block_size=64, n_layer=4, n_head=4, n_embd=128, dropout=0.0)
    model = GPT(big).eval()
    prompt = torch.zeros((1, 1), dtype=torch.long)
    n = big.block_size * 3

    t0 = time.perf_counter()
    generate_naive(model, prompt, n, temperature=0.0)
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    generate(model, prompt, n, temperature=0.0)
    t_cached = time.perf_counter() - t0

    assert t_cached < t_naive, f"cached {t_cached:.3f}s vs naive {t_naive:.3f}s"


def test_equivalence_still_holds_within_the_window(config):
    """Eviction must not disturb the exact-match guarantee below block_size."""
    torch.manual_seed(0)
    model = GPT(config).eval()
    prompt = torch.zeros((1, 2), dtype=torch.long)
    n = config.block_size - 5   # stays inside the window, no eviction triggered
    assert torch.equal(
        generate_naive(model, prompt.clone(), n, temperature=0.0),
        generate(model, prompt.clone(), n, temperature=0.0),
    )
