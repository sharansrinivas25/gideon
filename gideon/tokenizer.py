"""Tokenisers.

CharTokenizer is one token per character and is what the checkpoint uses.
BPETokenizer is byte-level BPE trained from scratch (GPT-2's algorithm without
the regex pre-tokeniser). Both expose encode / decode / vocab_size.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable


class CharTokenizer:
    """Character-level tokeniser. Vocabulary is the sorted set of characters."""

    def __init__(self, chars: Iterable[str]):
        self.chars = sorted(set(chars))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(text)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        # drop unknown characters rather than crashing on a stray prompt char
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"type": "char", "chars": self.chars}), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        assert obj["type"] == "char", f"not a char tokenizer: {obj['type']}"
        return cls(obj["chars"])


class BPETokenizer:
    """Byte-level BPE.

    Start from the 256 bytes, repeatedly merge the most frequent adjacent pair.
    Encoding replays the merges in the order they were learned.
    """

    def __init__(self, merges: dict[tuple[int, int], int] | None = None):
        self.merges: dict[tuple[int, int], int] = merges or {}
        self._build_vocab()

    def _build_vocab(self) -> None:
        # token id -> the bytes it expands to
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in sorted(self.merges.items(), key=lambda kv: kv[1]):
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    @staticmethod
    def _pair_counts(ids: list[int]) -> Counter:
        return Counter(zip(ids, ids[1:]))

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> "BPETokenizer":
        if vocab_size <= 256:
            raise ValueError("vocab_size must exceed 256 (the byte alphabet)")
        ids = list(text.encode("utf-8"))
        self.merges = {}
        for i in range(vocab_size - 256):
            counts = self._pair_counts(ids)
            if not counts:
                break
            pair, freq = counts.most_common(1)[0]
            if freq < 2:
                break
            new_id = 256 + i
            ids = self._merge(ids, pair, new_id)
            self.merges[pair] = new_id
            if verbose and i % 100 == 0:
                print(f"  merge {i:5d}: {pair} -> {new_id} (freq {freq})")
        self._build_vocab()
        return self

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        # lowest-numbered applicable merge each round == training order
        while len(ids) >= 2:
            counts = self._pair_counts(ids)
            candidates = [p for p in counts if p in self.merges]
            if not candidates:
                break
            pair = min(candidates, key=lambda p: self.merges[p])
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        raw = b"".join(self.vocab[int(i)] for i in ids)
        return raw.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "type": "bpe",
                    "merges": [[a, b, nid] for (a, b), nid in self.merges.items()],
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        assert obj["type"] == "bpe", f"not a bpe tokenizer: {obj['type']}"
        return cls({(a, b): nid for a, b, nid in obj["merges"]})


def load_tokenizer(path: str | Path):
    """Load whichever tokeniser was saved at ``path``."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"char": CharTokenizer, "bpe": BPETokenizer}[obj["type"]].load(path)
