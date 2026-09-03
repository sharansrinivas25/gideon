"""Attention correctness.

The hand-written attention is checked against two independent references:
PyTorch's fused ``scaled_dot_product_attention`` and a deliberately naive
loop-based implementation. If all three agree, the maths is right.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from minigpt.attention import CausalSelfAttention
from minigpt.config import GPTConfig


@pytest.fixture
def config():
    return GPTConfig(vocab_size=32, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.0)


def test_output_shape(config):
    attn = CausalSelfAttention(config).eval()
    x = torch.randn(2, 8, config.n_embd)
    assert attn(x).shape == (2, 8, config.n_embd)


def test_manual_matches_pytorch_fused(config):
    """Our softmax(QK^T/sqrt(d))V must equal the fused kernel."""
    torch.manual_seed(0)
    attn = CausalSelfAttention(config, use_flash=False).eval()
    flash = CausalSelfAttention(config, use_flash=True).eval()
    flash.load_state_dict(attn.state_dict())

    x = torch.randn(3, 12, config.n_embd)
    with torch.no_grad():
        torch.testing.assert_close(attn(x), flash(x), rtol=1e-5, atol=1e-6)


def test_manual_matches_naive_loop(config):
    """Check the vectorised implementation against an unmistakably correct loop."""
    torch.manual_seed(0)
    attn = CausalSelfAttention(config, use_flash=False).eval()
    B, T, C = 1, 6, config.n_embd
    x = torch.randn(B, T, C)

    with torch.no_grad():
        qkv = attn.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        nh, hd = config.n_head, config.head_dim
        q = q.view(B, T, nh, hd).transpose(1, 2)
        k = k.view(B, T, nh, hd).transpose(1, 2)
        v = v.view(B, T, nh, hd).transpose(1, 2)

        # Explicit per-head, per-query-position loop.
        out = torch.zeros(B, nh, T, hd)
        for h in range(nh):
            for i in range(T):
                scores = torch.tensor(
                    [float(q[0, h, i] @ k[0, h, j]) / math.sqrt(hd) for j in range(i + 1)]
                )
                w = F.softmax(scores, dim=0)
                out[0, h, i] = sum(w[j] * v[0, h, j] for j in range(i + 1))

        expected = attn.proj(out.transpose(1, 2).contiguous().view(B, T, C))
        torch.testing.assert_close(attn(x), expected, rtol=1e-4, atol=1e-5)


def test_causality_future_tokens_cannot_leak(config):
    """Changing token t must not change the output at any position < t.

    This is the property the whole architecture rests on: if it fails, the
    model is cheating during training and will fall apart at generation time.
    """
    torch.manual_seed(0)
    attn = CausalSelfAttention(config).eval()
    x = torch.randn(1, 10, config.n_embd)

    with torch.no_grad():
        y1 = attn(x)
        x2 = x.clone()
        x2[:, 7:] = torch.randn_like(x2[:, 7:])  # scramble the future
        y2 = attn(x2)

    torch.testing.assert_close(y1[:, :7], y2[:, :7], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(y1[:, 7:], y2[:, 7:]), "the future should have changed"


def test_attention_weights_are_a_distribution(config):
    """Rows of the attention matrix sum to 1 and are zero above the diagonal."""
    torch.manual_seed(0)
    attn = CausalSelfAttention(config).eval()
    B, T, C = 1, 8, config.n_embd
    x = torch.randn(B, T, C)

    with torch.no_grad():
        q, k, _ = attn.qkv(x).split(C, dim=2)
        nh, hd = config.n_head, config.head_dim
        q = q.view(B, T, nh, hd).transpose(1, 2)
        k = k.view(B, T, nh, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        att = att.masked_fill(attn.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

    torch.testing.assert_close(att.sum(-1), torch.ones(B, nh, T), rtol=1e-5, atol=1e-6)
    upper = torch.triu(torch.ones(T, T), diagonal=1).bool()
    assert att.masked_select(upper.expand(B, nh, T, T)).abs().max() == 0


def test_rejects_bad_head_division():
    with pytest.raises(ValueError, match="divisible"):
        GPTConfig(n_embd=30, n_head=4)
