"""Weight-only int8 quantisation.

Symmetric, one scale per output channel:

    scale_o  = max_i |W[o, i]| / 127
    Wq[o, i] = round(W[o, i] / scale_o)     -> int8
    W       ~= Wq * scale_o

Weight-only, so the matmul still runs in fp32 after dequantising: smaller but
not faster on CPU. dynamic_quantize_torch wraps torch's fbgemm path for
comparison.
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
    amax = w.abs().amax(dim=1, keepdim=True)
    scales = (amax / 127.0).clamp_min(1e-8)   # clamp: all-zero rows
    qweight = torch.round(w / scales).clamp(-127, 127).to(torch.int8)
    return qweight, scales


def quantization_error(w: torch.Tensor) -> dict[str, float]:
    """Round-trip error for a weight tensor."""
    q, s = quantize_tensor_per_channel(w)
    w_hat = q.to(torch.float32) * s
    err = (w - w_hat)
    return {
        "max_abs_error": err.abs().max().item(),
        "rmse": err.pow(2).mean().sqrt().item(),
        "relative_rmse": (err.pow(2).mean().sqrt() / w.pow(2).mean().sqrt()).item(),
    }


def quantize_model(model: nn.Module, skip: tuple[str, ...] = ("lm_head",)) -> nn.Module:
    """Copy of the model with every nn.Linear swapped for int8.

    lm_head is skipped: it is tied to the token embedding, and quantising it
    hurts sample quality for very little memory saved.
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
    """torch dynamic quantisation: also quantises activations, real int8 kernels."""
    return torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(model).eval(), {nn.Linear}, dtype=torch.qint8
    )


def model_size_bytes(model: nn.Module) -> int:
    """Serialised size in bytes.

    Serialise rather than sum parameters() + buffers(): torch's quantised
    Linear keeps its weights in a packed object that appears in neither, which
    made it look ~20x smaller than it is.
    """
    import io

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes
