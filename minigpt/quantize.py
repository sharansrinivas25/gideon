"""Weight-only int8 quantisation, implemented from scratch.

What it does
------------
Every ``nn.Linear`` weight (fp32, 4 bytes/element) is replaced by int8 values
(1 byte) plus one fp32 scale **per output channel**:

    scale_o = max_i |W[o, i]| / 127
    Wq[o, i] = round(W[o, i] / scale_o)          -> int8, range [-127, 127]
    W ~= Wq * scale_o

Per-channel (rather than one scale for the whole tensor) matters: a single
outlier row would otherwise force a huge scale and crush the resolution of
every other row. This is symmetric quantisation - no zero point - which is the
standard choice for weights, whose distributions are roughly zero-centred.

What it does *not* do
---------------------
This is **weight-only** quantisation, and the matmul is still run in fp32 after
dequantising. So it buys a ~4x reduction in weight memory but **no arithmetic
speedup on CPU** - in fact it costs a little time for the dequantisation. That
is the honest result and it is the interesting one: int8 pays off when you are
*memory-bandwidth bound* (large models, batch size 1, GPU serving), not when
you are compute bound on a small model.

For a genuine CPU speedup, :func:`dynamic_quantize_torch` wraps PyTorch's
fbgemm-backed dynamic quantisation, which also quantises activations at runtime
and dispatches to real int8 kernels. Benchmarks report both.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class QuantizedLinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` holding int8 weights."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        super().__init__()
        qweight, scales = quantize_tensor_per_channel(weight)
        # Buffers, not parameters: these are not trained.
        self.register_buffer("qweight", qweight)          # (out, in) int8
        self.register_buffer("scales", scales)            # (out, 1) fp32
        self.register_buffer("bias", bias.clone() if bias is not None else None)
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]

    def dequantized_weight(self) -> torch.Tensor:
        return self.qweight.to(torch.float32) * self.scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.dequantized_weight(), self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, dtype=int8"


def quantize_tensor_per_channel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel int8 quantisation of a 2-D weight."""
    assert w.dim() == 2, "expected a 2-D weight matrix"
    # amax over the input dimension -> one scale per output row.
    amax = w.abs().amax(dim=1, keepdim=True)
    # clamp_min avoids a divide-by-zero on an all-zero row.
    scales = (amax / 127.0).clamp_min(1e-8)
    qweight = torch.round(w / scales).clamp(-127, 127).to(torch.int8)
    return qweight, scales


def quantization_error(w: torch.Tensor) -> dict[str, float]:
    """Round-trip error for a weight tensor - useful for the report."""
    q, s = quantize_tensor_per_channel(w)
    w_hat = q.to(torch.float32) * s
    err = (w - w_hat)
    return {
        "max_abs_error": err.abs().max().item(),
        "rmse": err.pow(2).mean().sqrt().item(),
        "relative_rmse": (err.pow(2).mean().sqrt() / w.pow(2).mean().sqrt()).item(),
    }


def quantize_model(model: nn.Module, skip: tuple[str, ...] = ("lm_head",)) -> nn.Module:
    """Return a copy of ``model`` with every ``nn.Linear`` swapped for int8.

    ``lm_head`` is skipped by default: it is tied to the token embedding, and
    quantising the output head is the single change most likely to visibly
    degrade sample quality for the least memory saved.
    """
    model = copy.deepcopy(model).eval()

    def _replace(module: nn.Module, prefix: str = "") -> None:
        for name, child in list(module.named_children()):
            full = f"{prefix}{name}"
            if any(full.endswith(s) for s in skip):
                continue
            if isinstance(child, nn.Linear):
                setattr(module, name, QuantizedLinear(child.weight.data, child.bias.data if child.bias is not None else None))
            else:
                _replace(child, full + ".")

    _replace(model)
    return model


def dynamic_quantize_torch(model: nn.Module) -> nn.Module:
    """PyTorch dynamic quantisation - real int8 kernels, real CPU speedup.

    Included as the "what you would actually ship" comparison point against the
    from-scratch implementation above.
    """
    return torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(model).eval(), {nn.Linear}, dtype=torch.qint8
    )


def model_size_bytes(model: nn.Module) -> int:
    """Bytes of parameters + buffers, counting int8 buffers as 1 byte each."""
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        if b is not None:
            total += b.numel() * b.element_size()
    return total
