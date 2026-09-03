"""Sampling behaviour: temperature, top-k, top-p, determinism."""

import pytest
import torch

from gideon.config import GPTConfig
from gideon.generate import _sample_from_logits, generate
from gideon.model import GPT


@pytest.fixture
def config():
    return GPTConfig(vocab_size=16, block_size=16, n_layer=2, n_head=2, n_embd=16, dropout=0.0)


def test_greedy_picks_argmax():
    logits = torch.tensor([[0.1, 5.0, 0.2, 0.3]])
    assert _sample_from_logits(logits, temperature=0.0).item() == 1


def test_top_k_never_samples_outside_the_k():
    logits = torch.tensor([[3.0, 2.9, 0.1, 0.0, -1.0]])
    g = torch.Generator().manual_seed(0)
    samples = {_sample_from_logits(logits, 1.0, top_k=2, generator=g).item() for _ in range(200)}
    assert samples <= {0, 1}


def test_top_p_keeps_the_nucleus():
    # softmax puts ~0.88 on index 0; top_p=0.5 should leave only that token.
    logits = torch.tensor([[5.0, 2.9, 2.8, 2.7]])
    g = torch.Generator().manual_seed(0)
    samples = {_sample_from_logits(logits, 1.0, top_p=0.5, generator=g).item() for _ in range(100)}
    assert samples == {0}


def test_low_temperature_concentrates_the_distribution():
    logits = torch.tensor([[2.0, 1.9, 1.8]])
    g = torch.Generator().manual_seed(0)
    cold = [_sample_from_logits(logits, 0.1, generator=g).item() for _ in range(200)]
    g = torch.Generator().manual_seed(0)
    hot = [_sample_from_logits(logits, 5.0, generator=g).item() for _ in range(200)]
    assert cold.count(0) > hot.count(0)


def test_same_seed_same_output(config):
    torch.manual_seed(0)
    model = GPT(config).eval()
    prompt = torch.zeros((1, 1), dtype=torch.long)
    a = generate(model, prompt, 20, temperature=1.0, top_k=5,
                 generator=torch.Generator().manual_seed(7))
    b = generate(model, prompt, 20, temperature=1.0, top_k=5,
                 generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_output_length_and_range(config):
    torch.manual_seed(0)
    model = GPT(config).eval()
    prompt = torch.zeros((2, 3), dtype=torch.long)
    out = generate(model, prompt, 10, temperature=0.8, top_k=5)
    assert out.shape == (2, 13)
    assert out.min() >= 0 and out.max() < config.vocab_size


def test_prompt_is_preserved(config):
    torch.manual_seed(0)
    model = GPT(config).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    out = generate(model, prompt.clone(), 8, temperature=0.8)
    assert torch.equal(out[:, :5], prompt)
