"""Sampling / decoding.

Two generation functions with deliberately identical semantics:

* :func:`generate_naive` - no cache. Every step re-runs the model over the whole
  prefix. This is the baseline the speedup is measured against.
* :func:`generate`       - prefill once into a KV cache, then feed one token per
  step.

Given the same seed and sampling settings the two produce **identical token
sequences**; ``tests/test_kvcache.py`` asserts exactly that. If they ever
diverge, the cache is wrong.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import GPT

#: What to do once the KV cache reaches ``block_size``.
#:
#: ``"reprefill"``  recompute K/V for the recent window so every cached entry
#:                  carries the position embedding matching its slot. Costs one
#:                  prefill per ``block_size/4`` steps (~10% overhead) and keeps
#:                  output quality intact. Default.
#: ``"evict"``      slide the window by dropping the oldest entries, recomputing
#:                  nothing. Faster, but with *learned absolute* position
#:                  embeddings the surviving entries now sit at slots they were
#:                  not encoded for, and generation visibly degenerates. This is
#:                  the correct policy for RoPE, where position is applied at
#:                  attention time and can be re-indexed - it is kept here to
#:                  make that trade-off measurable rather than theoretical.
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
        # Greedy: temperature -> 0 is the argmax limit of the softmax.
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None:
        k = min(top_k, logits.size(-1))
        kth = logits.topk(k, dim=-1).values[:, [-1]]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p is not None:
        # Nucleus sampling: keep the smallest set of tokens whose cumulative
        # probability reaches top_p, drop the tail.
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
    """Baseline decoding: recompute the full forward pass every step."""
    model.eval()
    block = model.config.block_size
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block:]           # crop to context window
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
    """Cached decoding: one prefill pass, then one token at a time.

    Cost per step drops from O(prefix_len) full-model work to O(1) model work
    plus an O(prefix_len) attention read.

    ``window_policy`` controls what happens once the context window fills up.
    See :data:`WINDOW_POLICIES`. The default preserves output quality; the
    alternative is faster and measurably worse, and the README shows both.
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

    # --- prefill: process the whole prompt in one batched pass -------------
    prompt = idx[:, -block:]
    logits, _ = model(prompt, cache=cache)
    nxt = _sample_from_logits(logits[:, -1, :], temperature, top_k, top_p, generator)
    idx = torch.cat((idx, nxt), dim=1)

    # Work in chunks rather than one position at a time: both policies pay an
    # O(window) cost when the window fills, so doing it every step would undo
    # the caching. Handling a quarter-window at a time amortises that cost over
    # the next block/4 steps.
    chunk = max(1, block // 4)

    # --- decode: feed back one token per step ------------------------------
    for _ in range(max_new_tokens - 1):
        if cache.length >= block:
            if window_policy == "reprefill":
                # Recompute K/V for the recent window from scratch, so every
                # entry carries the position embedding matching the slot it now
                # occupies. Costs one prefill every `chunk` steps.
                cache.reset()
                logits, _ = model(idx[:, -(block - chunk):], cache=cache)
            else:  # "evict"
                # Drop the oldest entries and slide the rest forward. Cheaper -
                # no recompute at all - but stale K/V keep the position
                # embedding from the slot they *used* to occupy, and with
                # learned absolute positions that mismatch degrades output
                # badly. It is the right policy for RoPE, not for this model.
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

    ap = argparse.ArgumentParser(description="Generate text from a mini-gpt checkpoint")
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
