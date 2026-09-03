"""KV cache correctness: mechanics, logits, and full sampled sequences."""

import pytest
import torch

from gideon.config import GPTConfig
from gideon.generate import generate, generate_naive
from gideon.kvcache import KVCache, LayerKVCache
from gideon.model import GPT


@pytest.fixture
def config():
    return GPTConfig(vocab_size=32, block_size=32, n_layer=3, n_head=4, n_embd=32, dropout=0.0)


@pytest.fixture
def model(config):
    torch.manual_seed(0)
    return GPT(config).eval()


# --------------------------------------------------------------------- #
# cache mechanics
# --------------------------------------------------------------------- #
def test_cache_accumulates_in_order():
    cache = LayerKVCache(batch_size=1, n_head=2, max_seq_len=8, head_dim=4)
    assert cache.length == 0

    a_k, a_v = torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)
    k, v = cache.update(a_k, a_v)
    assert cache.length == 3 and k.shape == (1, 2, 3, 4)
    torch.testing.assert_close(k, a_k)

    b_k, b_v = torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4)
    k, v = cache.update(b_k, b_v)
    assert cache.length == 4 and k.shape == (1, 2, 4, 4)
    torch.testing.assert_close(k[:, :, :3], a_k)   # old entries untouched
    torch.testing.assert_close(k[:, :, 3:], b_k)   # new entry appended


def test_cache_overflow_raises():
    cache = LayerKVCache(batch_size=1, n_head=2, max_seq_len=4, head_dim=4)
    cache.update(torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4))
    with pytest.raises(ValueError, match="overflow"):
        cache.update(torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4))


def test_cache_reset(config):
    cache = KVCache(config.n_layer, 1, config.n_head, config.block_size, config.head_dim)
    cache[0].update(torch.randn(1, config.n_head, 5, config.head_dim),
                    torch.randn(1, config.n_head, 5, config.head_dim))
    assert cache[0].length == 5
    cache.reset()
    assert cache.length == 0


# --------------------------------------------------------------------- #
# equivalence: cached vs uncached
# --------------------------------------------------------------------- #
def test_incremental_logits_match_full_forward(model, config):
    """One token at a time through the cache == one full pass."""
    torch.manual_seed(1)
    idx = torch.randint(0, config.vocab_size, (1, 10))

    with torch.no_grad():
        full_logits, _ = model(idx)                # logits for the last position

        cache = model.new_cache(batch_size=1)
        step_logits = None
        for t in range(idx.size(1)):
            step_logits, _ = model(idx[:, t : t + 1], cache=cache)

    assert cache.length == 10
    torch.testing.assert_close(full_logits, step_logits, rtol=1e-4, atol=1e-5)


def test_prefill_then_decode_matches_full_forward(model, config):
    """Prefill the prompt, then step one token at a time."""
    torch.manual_seed(2)
    idx = torch.randint(0, config.vocab_size, (2, 12))

    with torch.no_grad():
        cache = model.new_cache(batch_size=2)
        model(idx[:, :8], cache=cache)             # prefill
        last = None
        for t in range(8, 12):
            last, _ = model(idx[:, t : t + 1], cache=cache)
        full, _ = model(idx)

    torch.testing.assert_close(full, last, rtol=1e-4, atol=1e-5)


def test_generated_sequences_are_identical(model, config):
    """Same seed: cached and naive decoding give the same tokens."""
    prompt = torch.randint(0, config.vocab_size, (1, 4))

    g1 = torch.Generator().manual_seed(42)
    naive = generate_naive(model, prompt.clone(), max_new_tokens=15, temperature=1.0, top_k=10, generator=g1)

    g2 = torch.Generator().manual_seed(42)
    cached = generate(model, prompt.clone(), max_new_tokens=15, temperature=1.0, top_k=10, generator=g2)

    assert torch.equal(naive, cached), (
        f"cache changed the output:\n naive={naive.tolist()}\ncached={cached.tolist()}"
    )


def test_greedy_decoding_identical(model, config):
    """Greedy decoding, no sampling noise."""
    prompt = torch.randint(0, config.vocab_size, (1, 3))
    naive = generate_naive(model, prompt.clone(), 20, temperature=0.0)
    cached = generate(model, prompt.clone(), 20, temperature=0.0)
    assert torch.equal(naive, cached)


def test_batched_generation_identical(model, config):
    prompt = torch.randint(0, config.vocab_size, (4, 5))
    naive = generate_naive(model, prompt.clone(), 12, temperature=0.0)
    cached = generate(model, prompt.clone(), 12, temperature=0.0)
    assert torch.equal(naive, cached)


def test_generation_past_context_window(model, config):
    """Generating beyond block_size shouldn't crash."""
    prompt = torch.randint(0, config.vocab_size, (1, 4))
    out = generate(model, prompt, max_new_tokens=config.block_size + 10, temperature=0.0)
    assert out.shape == (1, 4 + config.block_size + 10)
    assert out.max() < config.vocab_size and out.min() >= 0
