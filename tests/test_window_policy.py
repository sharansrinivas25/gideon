"""Long-context cache policies.

``reprefill`` recomputes K/V for the recent window so cached entries carry the
position embedding matching their slot. ``evict`` slides the window without
recomputing, which is cheaper but leaves stale entries encoded for slots they no
longer occupy.
"""

import pytest
import torch

from gideon.config import GPTConfig
from gideon.generate import WINDOW_POLICIES, generate, generate_naive
from gideon.model import GPT


@pytest.fixture
def config():
    return GPTConfig(vocab_size=32, block_size=32, n_layer=2, n_head=4, n_embd=64, dropout=0.0)


@pytest.fixture
def model(config):
    torch.manual_seed(0)
    return GPT(config).eval()


def test_unknown_policy_rejected(model):
    with pytest.raises(ValueError, match="unknown window_policy"):
        generate(model, torch.zeros((1, 1), dtype=torch.long), 5, window_policy="nonsense")


@pytest.mark.parametrize("policy", WINDOW_POLICIES)
def test_both_policies_produce_valid_output(model, config, policy):
    n = config.block_size * 3
    out = generate(model, torch.zeros((1, 1), dtype=torch.long), n,
                   temperature=0.8, top_k=10, window_policy=policy)
    assert out.shape == (1, n + 1)
    assert out.min() >= 0 and out.max() < config.vocab_size


@pytest.mark.parametrize("policy", WINDOW_POLICIES)
def test_policies_agree_inside_the_window(model, config, policy):
    """Below block_size neither policy has fired, so both must match exactly."""
    prompt = torch.zeros((1, 2), dtype=torch.long)
    n = config.block_size - 5
    assert torch.equal(
        generate_naive(model, prompt.clone(), n, temperature=0.0),
        generate(model, prompt.clone(), n, temperature=0.0, window_policy=policy),
    )


def test_reprefill_tracks_the_uncached_decoder_past_the_window(model, config):
    """Past the window, reprefill must stay close to full recomputation.

    Not bit-identical - reprefill attends over a slightly shorter window between
    refreshes - but the predicted distribution should barely move. This is the
    test that would have caught the original eviction bug, where output
    collapsed into noise the moment the context filled up.
    """
    torch.manual_seed(3)
    block = config.block_size
    ids = torch.randint(0, config.vocab_size, (1, block * 2))
    chunk = max(1, block // 4)

    with torch.no_grad():
        cache = model.new_cache(batch_size=1)
        cached_logits = None
        for t in range(ids.size(1)):
            if cache.length >= block:
                cache.reset()
                cached_logits, _ = model(ids[:, t + 1 - (block - chunk) : t + 1], cache=cache)
            else:
                cached_logits, _ = model(ids[:, t : t + 1], cache=cache)

        ref_logits, _ = model(ids[:, -block:])

    a = cached_logits[0, -1].softmax(-1)
    b = ref_logits[0, -1].softmax(-1)
    # total variation distance between the two next-token distributions
    tvd = 0.5 * (a - b).abs().sum().item()
    assert tvd < 0.25, f"reprefill drifted too far from the reference: TVD {tvd:.3f}"
