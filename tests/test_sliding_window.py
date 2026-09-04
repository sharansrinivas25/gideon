"""Sliding-window eviction.

Equivalence with the naive decoder only holds within block_size. Past it these
check that eviction moves the right data and long generations stay valid.
"""

import pytest
import torch

from gideon.config import GPTConfig
from gideon.generate import generate, generate_naive
from gideon.kvcache import LayerKVCache
from gideon.model import GPT


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
    """4x the context window: no overflow, no NaN, tokens stay in vocab."""
    torch.manual_seed(0)
    model = GPT(config).eval()
    prompt = torch.zeros((1, 1), dtype=torch.long)
    n = config.block_size * 4
    out = generate(model, prompt, n, temperature=0.8, top_k=10)
    assert out.shape == (1, n + 1)
    assert out.min() >= 0 and out.max() < config.vocab_size


def _count_positions(model):
    """Count token positions pushed through the model, via a forward pre-hook."""
    seen = {"positions": 0}

    def hook(_module, args, kwargs):
        idx = args[0] if args else kwargs["idx"]
        seen["positions"] += idx.numel()

    return seen, model.register_forward_pre_hook(hook, with_kwargs=True)


def test_long_generation_does_far_less_work_than_naive():
    """Past the window the cache should still avoid recomputation.

    Counts positions fed through the model rather than timing it: wall clock
    on a shared CI runner measures the machine's mood, not the algorithm.
    """
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=32, block_size=64, n_layer=2, n_head=4, n_embd=64, dropout=0.0)
    model = GPT(cfg).eval()
    prompt = torch.zeros((1, 1), dtype=torch.long)
    n = cfg.block_size * 3

    naive, h = _count_positions(model)
    generate_naive(model, prompt, n, temperature=0.0)
    h.remove()

    cached, h = _count_positions(model)
    generate(model, prompt, n, temperature=0.0)
    h.remove()

    assert cached["positions"] * 5 < naive["positions"], (
        f'cached processed {cached["positions"]} positions, naive {naive["positions"]}'
    )


def test_equivalence_still_holds_within_the_window(config):
    """Below block_size nothing has been evicted, so output must match exactly."""
    torch.manual_seed(0)
    model = GPT(config).eval()
    prompt = torch.zeros((1, 2), dtype=torch.long)
    n = config.block_size - 5   # stays inside the window, no eviction triggered
    assert torch.equal(
        generate_naive(model, prompt.clone(), n, temperature=0.0),
        generate(model, prompt.clone(), n, temperature=0.0),
    )
