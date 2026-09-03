"""Benchmark harness.

Cache speedup, prefill vs decode latency, long-context cache policy, and
quantisation. Everything is median-of-N after warmup.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from .config import GPTConfig
from .generate import generate, generate_naive
from .model import GPT
from .quantize import dynamic_quantize_torch, model_size_bytes, quantize_model


def _time_median(fn, repeats: int = 3, warmup: int = 1) -> float:
    """Median wall-clock seconds over `repeats` runs, after warmup."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


# --------------------------------------------------------------------- #
def bench_kv_cache(
    model: GPT, lengths: list[int], device: str = "cpu", repeats: int = 3
) -> list[dict]:
    """Cached vs naive decoding across generation lengths."""
    rows = []
    prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
    cache = model.new_cache(batch_size=1, device=device)

    for n in lengths:
        t_naive = _time_median(
            lambda: generate_naive(model, prompt, n, temperature=0.8, top_k=40), repeats
        )
        t_cached = _time_median(
            lambda: generate(model, prompt, n, temperature=0.8, top_k=40, cache=cache), repeats
        )
        rows.append(
            {
                "tokens": n,
                "naive_s": round(t_naive, 4),
                "cached_s": round(t_cached, 4),
                "naive_tok_per_s": round(n / t_naive, 1),
                "cached_tok_per_s": round(n / t_cached, 1),
                "speedup": round(t_naive / t_cached, 2),
            }
        )
        print(
            f"  {n:5d} tokens | naive {t_naive:7.3f}s ({n/t_naive:6.1f} tok/s) | "
            f"cached {t_cached:7.3f}s ({n/t_cached:6.1f} tok/s) | {t_naive/t_cached:5.2f}x"
        )
    return rows


def bench_prefill_vs_decode(model: GPT, prompt_lens: list[int], device: str = "cpu") -> list[dict]:
    """Time to first token (prefill) vs per-token decode time."""
    rows = []
    for p in prompt_lens:
        p = min(p, model.config.block_size - 1)
        idx = torch.randint(0, model.config.vocab_size, (1, p), device=device)

        def prefill():
            cache = model.new_cache(batch_size=1, device=device)
            with torch.no_grad():
                model(idx, cache=cache)

        cache = model.new_cache(batch_size=1, device=device)
        with torch.no_grad():
            model(idx, cache=cache)
        one = torch.zeros((1, 1), dtype=torch.long, device=device)

        def decode_step():
            with torch.no_grad():
                model(one, cache=cache)
            # roll back one slot so every trial measures the same context length
            for layer in cache.layers:
                layer.length -= 1

        ttft = _time_median(prefill, repeats=5)
        step = _time_median(decode_step, repeats=20, warmup=3)
        rows.append(
            {
                "prompt_tokens": p,
                "ttft_ms": round(ttft * 1000, 2),
                "decode_ms_per_token": round(step * 1000, 3),
            }
        )
        print(f"  prompt {p:4d} | TTFT {ttft*1000:8.2f} ms | decode {step*1000:6.3f} ms/token")
    return rows


def bench_quantization(model: GPT, n_tokens: int = 200, device: str = "cpu") -> dict:
    """Size and latency: fp32 vs from-scratch int8 vs torch dynamic int8."""
    prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
    variants = {
        "fp32": model,
        "int8_weight_only (from scratch)": quantize_model(model),
        "int8_dynamic (torch/fbgemm)": dynamic_quantize_torch(model),
    }

    out = {}
    fp32_bytes = model_size_bytes(model)
    for name, m in variants.items():
        size = model_size_bytes(m)
        t = _time_median(
            lambda: generate(m, prompt, n_tokens, temperature=0.8, top_k=40), repeats=3
        )
        out[name] = {
            "size_mb": round(size / 1e6, 3),
            "size_ratio_vs_fp32": round(fp32_bytes / size, 2),
            "gen_s": round(t, 4),
            "tok_per_s": round(n_tokens / t, 1),
        }
        print(
            f"  {name:34s} | {size/1e6:6.2f} MB ({fp32_bytes/size:4.2f}x smaller) | "
            f"{n_tokens/t:6.1f} tok/s"
        )
    return out


@torch.no_grad()
def bench_window_policy(
    model: GPT, tokenizer, text: str, seq_len: int = 512, device: str = "cpu"
) -> dict:
    """What each long-context cache policy costs in accuracy.

    Teacher-forced on held-out text, scoring only positions past block_size.
    Reference is the uncached decoder.
    """
    block = model.config.block_size
    ids = torch.tensor([tokenizer.encode(text)[: seq_len + 1]], dtype=torch.long, device=device)
    n = ids.size(1) - 1
    if n <= block:
        raise ValueError("need a sequence longer than block_size to compare policies")

    def score(step_logits: list[torch.Tensor]) -> float:
        """Mean cross-entropy past the window."""
        logits = torch.cat(step_logits, dim=0)                # (n, vocab)
        targets = ids[0, 1 : n + 1]
        keep = slice(block, n)
        return torch.nn.functional.cross_entropy(
            logits[keep], targets[keep]
        ).item()

    results = {}

    # reference: no cache
    outs = []
    for t in range(n):
        ctx = ids[:, max(0, t + 1 - block) : t + 1]
        lg, _ = model(ctx)
        outs.append(lg[:, -1, :])
    results["no cache (reference)"] = round(score(outs), 4)

    chunk = max(1, block // 4)
    for policy in ("reprefill", "evict"):
        cache = model.new_cache(batch_size=1, device=device)
        outs = []
        for t in range(n):
            if cache.length >= block:
                if policy == "reprefill":
                    cache.reset()
                    lg, _ = model(ids[:, max(0, t + 1 - (block - chunk)) : t + 1], cache=cache)
                else:
                    cache.evict_oldest(chunk)
                    lg, _ = model(ids[:, t : t + 1], cache=cache)
            else:
                lg, _ = model(ids[:, t : t + 1], cache=cache)
            outs.append(lg[:, -1, :])
        results[policy] = round(score(outs), 4)

    ref = results["no cache (reference)"]
    out = {
        k: {"val_loss_past_window": v, "delta_vs_reference": round(v - ref, 4)}
        for k, v in results.items()
    }
    for k, v in out.items():
        print(f"  {k:22s} | loss {v['val_loss_past_window']:.4f} "
              f"| delta {v['delta_vs_reference']:+.4f}")
    return out


def bench_kv_memory(config: GPTConfig) -> list[dict]:
    """KV cache footprint: 2 * n_layer * n_embd * seq_len * batch * 4 bytes."""
    rows = []
    for seq in [128, 512, 2048, 8192]:
        for batch in [1, 8, 32]:
            b = 2 * config.n_layer * config.n_embd * seq * batch * 4
            rows.append({"seq_len": seq, "batch": batch, "kv_cache_mb": round(b / 1e6, 2)})
    return rows


# --------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark gideon inference")
    ap.add_argument("--ckpt", default="results/ckpt.pt")
    ap.add_argument("--out", default="results/benchmarks.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    from .checkpoint import load_checkpoint
    from .data import download_tinyshakespeare

    model, tokenizer = load_checkpoint(args.ckpt, device=args.device)
    print(f"loaded {model.num_params():,} params from {args.ckpt}\n")

    results = {
        "meta": {
            "device": args.device,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "threads": torch.get_num_threads(),
            "config": model.config.to_dict(),
            "params": model.num_params(),
        }
    }

    print("KV cache: cached vs naive decoding")
    results["kv_cache"] = bench_kv_cache(
        model, [16, 32, 64, 128, 256, 512], args.device, args.repeats
    )

    print("\nPrefill vs decode")
    results["prefill_decode"] = bench_prefill_vs_decode(model, [8, 32, 64, 127], args.device)

    print("\nLong-context cache policy (held-out text, positions past the window)")
    text = download_tinyshakespeare().read_text(encoding="utf-8")
    val_text = text[int(len(text) * 0.9):]      # same split training held out
    results["window_policy"] = bench_window_policy(model, tokenizer, val_text, 512, args.device)

    print("\nQuantisation")
    results["quantization"] = bench_quantization(model, 200, args.device)

    print("\nKV cache memory scaling (projected)")
    results["kv_memory"] = bench_kv_memory(model.config)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nresults -> {args.out}")


if __name__ == "__main__":
    main()
