"""Sampling and decoding.

generate_naive re-runs the model over the whole prefix each step; generate
prefills into a KV cache then feeds one token at a time. Same seed and
settings give identical output (asserted in tests/test_kvcache.py).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import GPT

# What to do when the KV cache reaches block_size.
#   reprefill: recompute K/V for the recent window, so each entry's position
#              embedding matches its slot. ~10% overhead. Default.
#   evict:     drop the oldest entries. Cheaper, but wrong for learned absolute
#              positions (see README); correct under RoPE.
WINDOW_POLICIES = ("reprefill", "evict")


def _sample_from_logits(
    logits: torch.Tensor,          # (B, vocab)
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Turn logits into one sampled token id per batch row. (B, 1)"""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)   # greedy

    logits = logits / temperature

    if top_k is not None:
        k = min(top_k, logits.size(-1))
        kth = logits.topk(k, dim=-1).values[:, [-1]]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p is not None:
        # keep the smallest set of tokens whose cumulative prob reaches top_p
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cum - sorted_logits.softmax(dim=-1) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


@torch.no_grad()
def generate_naive(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Baseline: full forward pass every step, no cache."""
    model.eval()
    block = model.config.block_size
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block:]
        logits, _ = model(idx_cond)          # (B, 1, vocab)
        nxt = _sample_from_logits(
            logits[:, -1, :], temperature, top_k, top_p, generator
        )
        idx = torch.cat((idx, nxt), dim=1)
    return idx


@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
    cache=None,
    window_policy: str = "reprefill",
) -> torch.Tensor:
    """Cached decoding: prefill, then one token per step.

    window_policy controls what happens once the window fills up; see
    WINDOW_POLICIES.
    """
    if window_policy not in WINDOW_POLICIES:
        raise ValueError(f"unknown window_policy {window_policy!r}; expected one of {list(WINDOW_POLICIES)}")
    model.eval()
    B, T = idx.shape
    block = model.config.block_size

    if cache is None:
        cache = model.new_cache(batch_size=B, device=idx.device)
    else:
        cache.reset()

    # prefill the whole prompt in one pass
    prompt = idx[:, -block:]
    logits, _ = model(prompt, cache=cache)
    nxt = _sample_from_logits(logits[:, -1, :], temperature, top_k, top_p, generator)
    idx = torch.cat((idx, nxt), dim=1)

    # chunked so the O(window) cost is paid every block/4 steps, not every step
    chunk = max(1, block // 4)

    for _ in range(max_new_tokens - 1):
        if cache.length >= block:
            if window_policy == "reprefill":
                cache.reset()
                logits, _ = model(idx[:, -(block - chunk):], cache=cache)
            else:  # evict
                cache.evict_oldest(chunk)
                logits, _ = model(idx[:, -1:], cache=cache)
        else:
            logits, _ = model(idx[:, -1:], cache=cache)
        nxt = _sample_from_logits(logits[:, -1, :], temperature, top_k, top_p, generator)
        idx = torch.cat((idx, nxt), dim=1)

    return idx


def main() -> None:
    import argparse
    from pathlib import Path

    from .checkpoint import load_checkpoint

    ap = argparse.ArgumentParser(description="Generate text from a gideon checkpoint")
    ap.add_argument("--ckpt", default="results/ckpt.pt")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--tokens", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-cache", action="store_true", help="use the naive decoder")
    ap.add_argument("--int8", action="store_true", help="quantise weights to int8 first")
    args = ap.parse_args()

    model, tokenizer = load_checkpoint(Path(args.ckpt))
    if args.int8:
        from .quantize import quantize_model
        model = quantize_model(model)

    g = torch.Generator().manual_seed(args.seed)
    ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)
    if ids.numel() == 0:
        ids = torch.zeros((1, 1), dtype=torch.long)

    fn = generate_naive if args.no_cache else generate
    out = fn(model, ids, args.tokens, args.temperature, args.top_k, generator=g)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
