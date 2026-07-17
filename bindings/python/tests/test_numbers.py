"""Numeral normalization (g2p2.expand_numbers + phonemize auto-expansion).

Integer cardinals are built into the wheel (Rust `numbers` feature, no deps).
The optional `num2words2` Python engine (`pip install g2p2[numbers]`) adds
ordinals and decimals and takes precedence when present.

Run locally after building the wheel:

    maturin develop         # builds with the numbers feature (default)
    G2P2_MODELS=/path/to/models pytest bindings/python/tests/test_numbers.py

Tests skip cleanly when no engine or no model dir is available.
"""

import os

import pytest

import g2p2

_HAS_NATIVE = g2p2._native_expand is not None
_HAS_N2W = g2p2._num2words_sentence() is not None
_ANY_ENGINE = _HAS_NATIVE or _HAS_N2W


def _norm(s):
    # the num2words2 engine capitalizes sentence-initially; compare lowercased.
    return s.lower()


@pytest.mark.skipif(not _ANY_ENGINE, reason="no number engine available")
def test_expand_numbers_cardinals():
    assert _norm(g2p2.expand_numbers("12 rue de la Paix", "fr")) == "douze rue de la paix"
    assert _norm(g2p2.expand_numbers("3 cats", "en")) == "three cats"
    assert _norm(g2p2.expand_numbers("21", "es")) == "veintiuno"


@pytest.mark.skipif(not _ANY_ENGINE, reason="no number engine available")
def test_yue_falls_back_to_mandarin():
    # neither engine has Cantonese; g2p2 maps yue -> zh.
    assert g2p2.expand_numbers("5", "yue") == g2p2.expand_numbers("5", "zh")


@pytest.mark.skipif(not _HAS_NATIVE, reason="native engine not built in")
def test_native_spell_cardinal():
    assert g2p2._native.spell_cardinal("100", "es") == "cien"
    assert g2p2._native.spell_cardinal("rue", "fr") is None  # not an integer


@pytest.mark.skipif(not _HAS_N2W, reason="num2words2 extra not installed")
def test_ordinal_and_decimal_need_extra():
    # the richer engine handles ordinals in context and decimals.
    assert "premier" in _norm(g2p2.expand_numbers("1er étage", "fr"))
    assert "point" in _norm(g2p2.expand_numbers("3.14", "en"))


def test_missing_all_engines_is_graceful(monkeypatch):
    # With no engine, best-effort expansion is a no-op; the explicit public call
    # raises with an install hint.
    monkeypatch.setattr(g2p2, "_native_expand", None)
    monkeypatch.setattr(g2p2, "_num2words_sentence", lambda: None)
    assert g2p2._expand_numbers("12", "fr", required=False) == "12"
    with pytest.raises(ImportError, match="g2p2\\[numbers\\]"):
        g2p2.expand_numbers("12", "fr")


MODELS = os.environ.get("G2P2_MODELS")


@pytest.mark.skipif(not (MODELS and _ANY_ENGINE), reason="need G2P2_MODELS + an engine")
def test_phonemize_expands_numbers():
    # '12' spells to 'douze' and phonemizes identically.
    assert g2p2.phonemize("12", language="fr") == g2p2.phonemize("douze", language="fr")
    # With expansion off, raw digits decode differently.
    assert g2p2.phonemize("2026", language="fr", expand_numbers=False) != g2p2.phonemize(
        "2026", language="fr", expand_numbers=True
    )
    # A multi-word expansion phonemizes token-by-token, space-joined.
    assert " " in g2p2.phonemize("42", language="en")  # 'forty two'
