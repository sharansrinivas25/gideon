"""Tokeniser round-trip and BPE merge behaviour."""

import pytest

from minigpt.tokenizer import BPETokenizer, CharTokenizer, load_tokenizer

TEXT = "To be, or not to be, that is the question."


def test_char_roundtrip():
    tok = CharTokenizer.from_text(TEXT)
    assert tok.decode(tok.encode(TEXT)) == TEXT


def test_char_vocab_is_sorted_unique():
    tok = CharTokenizer.from_text("banana")
    assert tok.chars == ["a", "b", "n"]
    assert tok.vocab_size == 3


def test_char_drops_unknown_characters():
    tok = CharTokenizer.from_text("abc")
    assert tok.decode(tok.encode("abcXYZ")) == "abc"


def test_char_save_load(tmp_path):
    tok = CharTokenizer.from_text(TEXT)
    p = tmp_path / "tok.json"
    tok.save(p)
    assert load_tokenizer(p).encode(TEXT) == tok.encode(TEXT)


# --------------------------------------------------------------------- #
def test_bpe_roundtrip():
    corpus = TEXT * 20
    tok = BPETokenizer().train(corpus, vocab_size=300)
    assert tok.decode(tok.encode(TEXT)) == TEXT


def test_bpe_compresses():
    """Merges should make the encoded sequence shorter than raw bytes."""
    corpus = TEXT * 50
    tok = BPETokenizer().train(corpus, vocab_size=350)
    assert len(tok.encode(corpus)) < len(corpus.encode("utf-8")) * 0.7


def test_bpe_handles_unicode():
    corpus = "héllo wörld " * 30
    tok = BPETokenizer().train(corpus, vocab_size=300)
    assert tok.decode(tok.encode("héllo wörld")) == "héllo wörld"


def test_bpe_untrained_is_identity_over_bytes():
    tok = BPETokenizer()
    assert tok.vocab_size == 256
    assert tok.encode("abc") == [97, 98, 99]
    assert tok.decode([97, 98, 99]) == "abc"


def test_bpe_rejects_tiny_vocab():
    with pytest.raises(ValueError, match="exceed 256"):
        BPETokenizer().train(TEXT, vocab_size=100)


def test_bpe_save_load(tmp_path):
    tok = BPETokenizer().train(TEXT * 20, vocab_size=300)
    p = tmp_path / "bpe.json"
    tok.save(p)
    assert load_tokenizer(p).encode(TEXT) == tok.encode(TEXT)
