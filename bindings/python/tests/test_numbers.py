"""Numeral normalization (g2p2.expand_numbers + phonemize auto-expansion).

Run locally after building the wheel:

    pip install -e ".[numbers]"        # or: maturin develop + pip install num2words2
    G2P2_MODELS=/path/to/models pytest bindings/python/tests/test_numbers.py

Tests skip cleanly when the optional `num2words2` extra or a model dir is absent,
so they never break the lean install.
"""

import os

import pytest

import g2p2

n2w = pytest.importorskip("num2words2", reason="install g2p2[numbers]")


def _norm(s):
    # expand_numbers capitalizes sentence-initially; compare case-insensitively.
    return s.lower()


def test_expand_numbers_basic():
    assert _norm(g2p2.expand_numbers("12 rue de la Paix", "fr")) == "douze rue de la paix"
    assert _norm(g2p2.expand_numbers("3 cats", "en")) == "three cats"
    assert _norm(g2p2.expand_numbers("21", "es")) == "veintiuno"


def test_expand_numbers_ordinal_and_decimal():
    # num2words2 handles ordinals in context and decimals.
    assert "premier" in _norm(g2p2.expand_numbers("1er étage", "fr"))
    assert "point" in _norm(g2p2.expand_numbers("3.14", "en"))


def test_yue_falls_back_to_mandarin():
    # num2words2 has no Cantonese key; g2p2 maps yue -> zh.
    assert g2p2.expand_numbers("5", "yue") == g2p2.expand_numbers("5", "zh")


def test_unsupported_language_returns_unchanged(monkeypatch):
    # A language num2words2 declines is passed through unchanged, not an error.
    monkeypatch.setattr(g2p2, "_N2W_LANG", {"xx": "xx"})
    # 'xx' isn't a real key -> num2words2 raises NotImplementedError -> unchanged
    assert g2p2._expand_numbers("7", "xx", required=False) == "7"


def test_missing_extra_is_graceful(monkeypatch):
    # When num2words2 is unavailable, expansion is a no-op (best-effort) but
    # the explicit public call raises with an install hint.
    monkeypatch.setattr(g2p2, "_num2words_sentence", lambda: None)
    assert g2p2._expand_numbers("12", "fr", required=False) == "12"
    with pytest.raises(ImportError, match="g2p2\\[numbers\\]"):
        g2p2.expand_numbers("12", "fr")


MODELS = os.environ.get("G2P2_MODELS")


@pytest.mark.skipif(not MODELS, reason="set G2P2_MODELS to a dir with fr.g2p")
def test_phonemize_expands_numbers():
    # '12' spells to 'douze' and phonemizes identically.
    assert g2p2.phonemize("12", language="fr") == g2p2.phonemize("douze", language="fr")
    # With expansion off, the raw digits are decoded (different result).
    assert g2p2.phonemize("2026", language="fr", expand_numbers=False) != g2p2.phonemize(
        "2026", language="fr", expand_numbers=True
    )
    # A multi-word expansion phonemizes token-by-token, space-joined.
    out = g2p2.phonemize("42", language="en")
    assert " " in out  # 'forty two' -> two phoneme groups
