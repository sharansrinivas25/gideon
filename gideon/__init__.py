"""Gideon: a GPT-style transformer and inference engine built from scratch."""

from .config import GPTConfig, TrainConfig
from .model import GPT, Block, MLP
from .attention import CausalSelfAttention
from .kvcache import KVCache, LayerKVCache
from .tokenizer import CharTokenizer, BPETokenizer, load_tokenizer
from .generate import generate, generate_naive
from .quantize import quantize_model, QuantizedLinear
from .checkpoint import save_checkpoint, load_checkpoint

__version__ = "0.1.0"

__all__ = [
    "GPTConfig", "TrainConfig", "GPT", "Block", "MLP",
    "CausalSelfAttention", "KVCache", "LayerKVCache",
    "CharTokenizer", "BPETokenizer", "load_tokenizer",
    "generate", "generate_naive",
    "quantize_model", "QuantizedLinear",
    "save_checkpoint", "load_checkpoint",
]
