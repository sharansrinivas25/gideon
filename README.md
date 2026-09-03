# Gideon

A small GPT-style language model and inference engine, written from scratch in PyTorch.
No `nn.Transformer` or `nn.MultiheadAttention`: the attention, KV cache, tokeniser and
int8 quantisation are all implemented here.

I built it to understand how LLM inference actually works, so most of the effort went into
the serving side (caching, quantisation, benchmarking) rather than into training a good
model.

[![tests](https://github.com/sharansrinivas25/gideon/actions/workflows/tests.yml/badge.svg)](https://github.com/sharansrinivas25/gideon/actions/workflows/tests.yml)

## Results

Trained on TinyShakespeare, CPU only (2 cores). Raw numbers in `results/benchmarks.json`.

| | |
|---|---|
| Model | 2.67M params, 6 layers, 6 heads, `n_embd` 192, `block_size` 128 |
| Training | 3000 iters in 46 min, val loss 4.21 → 1.63 |
| KV cache | 3.6x faster at 512 tokens |
| Latency | 7.8 ms to first token (127-token prompt), 1.72 ms per token after |
| int8 | 3.75x smaller, but slower on CPU (see below) |
| Tests | 61 |

![loss curve](docs/loss_curve.png)

## Setup

```bash
pip install -e ".[dev]"

make test      # 61 tests, ~5s
make train     # ~46 min on CPU, a few minutes on a GPU
make bench
make figures
```

Generating text:

```bash
python -m gideon.generate --ckpt results/ckpt.pt --prompt "ROMEO:" --tokens 500
python -m gideon.generate --ckpt results/ckpt.pt --tokens 500 --no-cache   # uncached
python -m gideon.generate --ckpt results/ckpt.pt --tokens 500 --int8
```

Sample output after training:

```
ROMEO:
God and be suchord, let seal was the mour
Reath of thy forst me born to meet, and which
he plays the peace of lady:
Which her they ta'en were and good life such preson these
resk windars lame me; we never the royal of the
subjection, and sclett, for at the childress for here.

CORIOLANUS:
For thou wast him towards, to light them own and traitor.
```

Not coherent English, which is about what you'd expect from 2.7M params on 1M characters.
It gets the structure (speaker labels, line breaks, punctuation, plausible-looking words)
and nothing else.

## Layout

```
gideon/
  config.py       model + training config
  attention.py    multi-head causal self-attention
  model.py        transformer blocks, GPT
  kvcache.py      pre-allocated KV cache
  tokenizer.py    char-level and byte-level BPE
  generate.py     sampling, cached and uncached decoding
  quantize.py     int8 weight quantisation
  train.py        training loop
  benchmark.py    latency, throughput, cache benchmarks
tests/            61 tests
scripts/          figure generation
```

## KV cache

The model is causal, so keys and values for earlier tokens don't change as you generate.
Caching them makes each step O(1) model work instead of O(n).

The cache is pre-allocated to `block_size` and written in place. Growing it with
`torch.cat` reallocates and copies the whole thing every token, which cancels out most of
the benefit.

![kv cache](docs/kv_cache_speedup.png)

| tokens | no cache | KV cache | speedup |
|---:|---:|---:|---:|
| 64 | 234 tok/s | 428 tok/s | 1.8x |
| 128 | 186 tok/s | 503 tok/s | 2.7x |
| 256 | 149 tok/s | 445 tok/s | 3.0x |
| 512 | 139 tok/s | 502 tok/s | 3.6x |

Cached throughput stays flat at roughly 500 tok/s whatever the length. Without the cache
it decays as the prefix grows. The tests assert both decoders produce identical tokens
for the same seed.

Memory cost is `2 * n_layer * n_embd * seq_len * batch * 4` bytes, plotted in
`docs/kv_memory.png`. This is why long context is expensive to serve and why GQA exists.

## The context window bug

My first attempt at handling a full cache just dropped the oldest entries and slid the
window along. Fast, but the output fell apart at exactly 128 characters:

```
Which her the thate ffore atrege ownthe sthe ursstn therive mard; warvild y
mifthe th, tithon caly oonde thur aly t
Talaly cle t,
```

The model uses learned absolute position embeddings, added before K and V are computed.
Sliding a cached entry from slot 40 to slot 8 leaves it carrying the position-40
embedding, so the model reads a sequence whose positions contradict each other.

The fix (`reprefill`, now the default) recomputes K/V for the recent window when it fills
up. That costs one prefill every `block_size/4` steps, about 10% overhead, and it still
holds 3.6x over no cache.

![window policy](docs/window_policy.png)

| policy | loss past the window |
|---|---:|
| no cache (reference) | 1.484 |
| reprefill | 1.480 |
| evict | 3.173 |

Both policies are still selectable, since eviction is the right approach under RoPE where
position is applied at attention time rather than baked into the cached values.

## Quantisation

Weights go from fp32 to int8 with one scale per output channel. Per-tensor scaling gets
ruined by a single outlier row.

![quantisation](docs/quantization.png)

| | size | throughput |
|---|---:|---:|
| fp32 | 10.79 MB | 495 tok/s |
| int8 weight-only | 2.88 MB | 304 tok/s |
| int8 dynamic (torch) | 2.86 MB | 351 tok/s |

It's weight-only, so the matmul still runs in fp32 after dequantising: 3.75x smaller but
slower. int8 pays off when you're memory-bandwidth bound (large models, batch size 1, GPU
serving), not when you're compute bound on something this small. torch's dynamic
quantisation also quantises activations and uses real int8 kernels, which closes some of
the gap but still doesn't beat fp32 here.

One thing that caught me out: summing `parameters()` and `buffers()` misses torch's packed
quantised weights and reported a 20x saving that wasn't real. `model_size_bytes`
serialises the state dict instead.

## Limitations

- Character-level tokenisation on 1.1M characters. Samples are Shakespeare-shaped, not
  coherent.
- Learned absolute positions rather than RoPE. RoPE would be the obvious next change, and
  would also make cache eviction work properly.
- Standard MHA, not GQA or MQA, so the KV cache is as large as it can be.
- Benchmarks are CPU-only. int8 would look far better on a GPU.
- `reprefill` is simpler than what a real serving stack does (attention sinks, paged
  attention).
- The checkpoint isn't committed (10.8 MB). `make train` reproduces it.

## Branches

Topic branches merged into `main` with `--no-ff`, one per feature or fix:
`feat/tokenizers`, `feat/transformer`, `feat/kv-cache`,
`feat/quantisation-and-training`, `feat/benchmarks`, `fix/context-window-collapse`,
`fix/quantisation-size-measurement`, `refactor/rename-to-gideon`, `docs/*`.

## Licence

MIT
