"""Model-level behaviour: shapes, loss, gradients, weight tying."""

import math

import pytest
import torch

from minigpt.config import GPTConfig
from minigpt.model import GPT


@pytest.fixture
def config():
    return GPTConfig(vocab_size=40, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.0)


def test_forward_shapes(config):
    model = GPT(config).eval()
    idx = torch.randint(0, config.vocab_size, (2, 10))

    logits, loss = model(idx)                     # inference: last position only
    assert logits.shape == (2, 1, config.vocab_size) and loss is None

    logits, loss = model(idx, targets=idx)        # training: every position
    assert logits.shape == (2, 10, config.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_initial_loss_is_near_uniform(config):
    """An untrained model should be about as confused as a uniform guess.

    Expected cross-entropy = ln(vocab_size). A much higher value means the
    initialisation is broken; a much lower one means something is leaking.
    """
    torch.manual_seed(0)
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (8, 16))
    _, loss = model(idx, targets=idx)
    assert abs(loss.item() - math.log(config.vocab_size)) < 0.4, loss.item()


def test_weights_are_tied(config):
    model = GPT(config)
    assert model.tok_emb.weight.data_ptr() == model.lm_head.weight.data_ptr()


def test_all_parameters_receive_gradients(config):
    """A parameter with no gradient is a parameter that is silently dead."""
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (4, 12))
    _, loss = model(idx, targets=idx)
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient for: {missing}"


def test_model_can_overfit_a_single_batch(config):
    """The sanity check that catches almost every training bug.

    If a model cannot drive the loss to near-zero on one fixed batch, the
    problem is the model or the optimiser, not the data or the hyperparameters.
    """
    torch.manual_seed(0)
    model = GPT(config)
    model.train()
    idx = torch.randint(0, config.vocab_size, (4, 16))
    targets = torch.randint(0, config.vocab_size, (4, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    for _ in range(300):
        opt.zero_grad()
        _, loss = model(idx, targets=targets)
        loss.backward()
        opt.step()

    assert loss.item() < 0.1, f"failed to overfit: final loss {loss.item():.4f}"


def test_rejects_sequence_longer_than_block_size(config):
    model = GPT(config).eval()
    idx = torch.randint(0, config.vocab_size, (1, config.block_size + 1))
    with pytest.raises(ValueError, match="exceeds block_size"):
        model(idx)


def test_param_count_matches_hand_calculation():
    """Guards against an accidental extra/missing layer."""
    cfg = GPTConfig(vocab_size=100, block_size=32, n_layer=2, n_head=2, n_embd=64, bias=False)
    model = GPT(cfg)
    C, L = cfg.n_embd, cfg.n_layer
    per_block = (3 * C * C + C * C) + (4 * C * C + 4 * C * C) + (2 * C)  # attn + mlp + 2 LayerNorms
    expected = (
        cfg.vocab_size * C          # tied embedding / lm_head
        + cfg.block_size * C        # positional embedding
        + L * per_block
        + C                         # final LayerNorm
    )
    assert model.num_params(non_embedding=False) == expected
