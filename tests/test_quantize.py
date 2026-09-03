"""Quantisation correctness and error bounds."""

import pytest
import torch
import torch.nn as nn

from gideon.config import GPTConfig
from gideon.generate import generate
from gideon.model import GPT
from gideon.quantize import (
    QuantizedLinear,
    model_size_bytes,
    quantization_error,
    quantize_model,
    quantize_tensor_per_channel,
)


@pytest.fixture
def config():
    return GPTConfig(vocab_size=32, block_size=32, n_layer=3, n_head=4, n_embd=64, dropout=0.0)


def test_quantized_values_fit_int8():
    w = torch.randn(16, 32) * 5
    q, s = quantize_tensor_per_channel(w)
    assert q.dtype == torch.int8
    assert q.abs().max() <= 127
    assert s.shape == (16, 1)


def test_per_channel_scales_track_row_magnitude():
    """A row with a huge outlier must not degrade the other rows' resolution."""
    w = torch.randn(4, 32) * 0.01
    w[0] *= 1000.0                       # one wildly-scaled row
    q, s = quantize_tensor_per_channel(w)
    assert s[0] > s[1] * 100             # that row gets its own big scale
    # every row still uses close to the full int8 range
    assert q.abs().amax(dim=1).min() >= 120


def test_roundtrip_error_is_small():
    torch.manual_seed(0)
    w = torch.randn(128, 256) * 0.02     # realistic init scale
    err = quantization_error(w)
    # symmetric int8 over a Gaussian: relative RMSE lands around 0.2-0.5%
    assert err["relative_rmse"] < 0.01, err


def test_zero_row_does_not_produce_nan():
    w = torch.randn(4, 8)
    w[2] = 0.0
    q, s = quantize_tensor_per_channel(w)
    assert torch.isfinite(s).all()
    assert (q[2] == 0).all()


def test_quantized_linear_approximates_fp32():
    torch.manual_seed(0)
    lin = nn.Linear(64, 32, bias=True)
    qlin = QuantizedLinear(lin.weight.data, lin.bias.data)
    x = torch.randn(8, 64)
    torch.testing.assert_close(lin(x), qlin(x), rtol=0.02, atol=0.02)


def test_quantize_model_replaces_linears(config):
    model = GPT(config).eval()
    qmodel = quantize_model(model)
    n_q = sum(1 for m in qmodel.modules() if isinstance(m, QuantizedLinear))
    # per block: attn.qkv, attn.proj, mlp.fc, mlp.proj = 4; lm_head is skipped
    assert n_q == 4 * config.n_layer


def test_quantized_model_is_smaller(config):
    model = GPT(config).eval()
    qmodel = quantize_model(model)
    ratio = model_size_bytes(model) / model_size_bytes(qmodel)
    assert ratio > 1.4, f"expected meaningful shrinkage, got {ratio:.2f}x"


def test_quantized_model_still_produces_sane_logits(config):
    torch.manual_seed(0)
    model = GPT(config).eval()
    qmodel = quantize_model(model)
    idx = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        a, _ = model(idx)
        b, _ = qmodel(idx)
    assert torch.isfinite(b).all()
    # logits should be close, and crucially the ranking should barely move
    assert (a.argmax(-1) == b.argmax(-1)).float().mean() >= 0.5
    torch.testing.assert_close(a, b, rtol=0.1, atol=0.5)


def test_quantized_model_generates(config):
    torch.manual_seed(0)
    qmodel = quantize_model(GPT(config).eval())
    out = generate(qmodel, torch.zeros((1, 1), dtype=torch.long), 20, temperature=0.8, top_k=10)
    assert out.shape == (1, 21)
    assert out.max() < config.vocab_size
