"""g2p2 — grapheme-to-phoneme for the 100 Whisper languages.

Low-level (explicit model file):

    from g2p2 import Model
    m = Model.load("fr.g2p")
    m.phonemize("bonjour")            # 'bɔ̃ʒuʁ'

High-level (pick by language — model auto-loaded, downloaded + cached once):

    import g2p2
    g2p2.phonemize("hello", language="en")
    g2p2.word_similarity("light", "night", language="en", method="fast")

Model resolution order for a language `xx`:
  1. ``$G2P2_MODELS/xx.g2p``           (a directory you point at)
  2. ``<cache>/g2p2/xx.g2p``           (XDG cache, else ~/.cache)
  3. download ``$G2P2_MODELS_URL/xx.g2p`` into the cache (GitHub release by default)
"""

from __future__ import annotations

import functools
import os
import urllib.request
from pathlib import Path

from ._native import Model, __version__
from ._native import distance as _distance
from ._native import similarity as _similarity

_MODELS_URL = os.environ.get(
    "G2P2_MODELS_URL",
    "https://github.com/jqueguiner/g2p2/releases/download/models-v2",
)

__all__ = [
    "Model",
    "phonemize",
    "phonemize_many",
    "word_similarity",
    "similarity",
    "distance",
    "get_model",
    "model_path",
    "__version__",
]


def _bundled_dir() -> Path | None:
    """Directory of models shipped inside the wheel (present in the default
    `pip install g2p2`), or None for the lean/no-data install."""
    d = Path(__file__).resolve().parent / "models"
    return d if d.is_dir() else None


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "g2p2"


# Whisper language code -> family group (matches scripts/groups.json). Each
# `pip install g2p2[<group>]` installs the g2p2-group-<group> data package.
_LANG_GROUP = {"af": "germanic", "am": "semitic", "ar": "semitic", "as": "indic", "az": "turkic", "ba": "turkic", "be": "slavic", "bg": "slavic", "bn": "indic", "bo": "other", "br": "celtic", "bs": "slavic", "ca": "romance", "cs": "slavic", "cy": "celtic", "da": "germanic", "de": "germanic", "el": "other-euro", "en": "germanic", "es": "romance", "et": "other-euro", "eu": "other-euro", "fa": "iranian", "fi": "other-euro", "fo": "germanic", "fr": "romance", "gl": "romance", "gu": "indic", "ha": "african", "haw": "pacific", "he": "semitic", "hi": "indic", "hr": "slavic", "ht": "romance", "hu": "other-euro", "hy": "other-euro", "id": "sea", "is": "germanic", "it": "romance", "ja": "cjk", "jw": "sea", "ka": "other-euro", "kk": "turkic", "km": "sea", "kn": "indic", "ko": "cjk", "la": "romance", "lb": "germanic", "ln": "african", "lo": "sea", "lt": "other-euro", "lv": "other-euro", "mg": "african", "mi": "pacific", "mk": "slavic", "ml": "indic", "mn": "other", "mr": "indic", "ms": "sea", "mt": "semitic", "my": "sea", "ne": "indic", "nl": "germanic", "nn": "germanic", "no": "germanic", "oc": "romance", "pa": "indic", "pl": "slavic", "ps": "iranian", "pt": "romance", "ro": "romance", "ru": "slavic", "sa": "indic", "sd": "indic", "si": "indic", "sk": "slavic", "sl": "slavic", "sn": "african", "so": "african", "sq": "other-euro", "sr": "slavic", "su": "sea", "sv": "germanic", "sw": "african", "ta": "indic", "te": "indic", "tg": "iranian", "th": "sea", "tk": "turkic", "tl": "sea", "tr": "turkic", "tt": "turkic", "uk": "slavic", "ur": "indic", "uz": "turkic", "vi": "sea", "yi": "germanic", "yo": "african", "yue": "cjk", "zh": "cjk"}


def _installed_pack(language: str) -> Path | None:
    """Model from an installed language-family package.

    ``pip install g2p2[romance]`` installs ``g2p2-group-romance`` (module
    ``g2p2_group_romance``) carrying ``fr.g2p.xz``, ``es.g2p.xz``, …; ``g2p2[all]``
    installs every group via the g2p2-models meta. The blob is decompressed once
    into the cache.
    """
    import importlib.util
    import lzma

    group = _LANG_GROUP.get(language)
    if group is None:
        return None
    module = f"g2p2_group_{group}".replace("-", "_")
    spec = importlib.util.find_spec(module)
    if spec is None or not spec.submodule_search_locations:
        return None
    src = Path(spec.submodule_search_locations[0]) / f"{language}.g2p.xz"
    if not src.exists():
        return None
    out = _cache_dir() / f"{language}.g2p"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".g2p.part")
        tmp.write_bytes(lzma.decompress(src.read_bytes()))
        tmp.replace(out)
    return out


def model_path(language: str) -> str:
    """Resolve `language`'s ``.g2p`` blob. Order: ``$G2P2_MODELS`` dir ->
    models bundled in the wheel -> the ``g2p2[all]`` offline pack ->
    download+cache from the release."""
    # 1. explicit override directory
    override = os.environ.get("G2P2_MODELS")
    if override:
        p = Path(override) / f"{language}.g2p"
        if p.exists():
            return str(p)
    # 2. bundled with the package (default install ships all languages)
    bundled = _bundled_dir()
    if bundled:
        p = bundled / f"{language}.g2p"
        if p.exists():
            return str(p)
    # 3. offline pack installed via `pip install g2p2[all]`
    packed = _installed_pack(language)
    if packed is not None:
        return str(packed)
    # 4. download into the user cache, once
    d = _cache_dir()
    p = d / f"{language}.g2p"
    if not p.exists():
        d.mkdir(parents=True, exist_ok=True)
        url = f"{_MODELS_URL}/{language}.g2p"
        tmp = p.with_suffix(".g2p.part")
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(p)
        except Exception as e:  # noqa: BLE001
            tmp.unlink(missing_ok=True)
            raise FileNotFoundError(
                f"no model for language {language!r}: not bundled, {p} missing, and "
                f"download from {url} failed ({e}). Set $G2P2_MODELS to a directory "
                f"with {language}.g2p, or build it: "
                f"`cargo run -p xtask -- build {language}`."
            ) from e
    return str(p)


@functools.lru_cache(maxsize=None)
def get_model(language: str) -> Model:
    """Load (and cache in-process) the model for a Whisper language code."""
    return Model.load(model_path(language))


def phonemize(word: str, language: str) -> str:
    """IPA for `word` in `language`."""
    return get_model(language).phonemize(word)


def phonemize_many(words, language: str):
    """IPA for many `words` in `language`."""
    return get_model(language).phonemize_many(list(words))


def word_similarity(a: str, b: str, language: str, method: str = "weighted") -> float:
    """Phonemize `a` and `b` in `language`, then score similarity (0..1).

    method: "weighted" (default, articulatory features) or "fast" (Levenshtein).
    """
    return get_model(language).word_similarity(a, b, method)


def similarity(a: str, b: str, method: str = "weighted") -> float:
    """Similarity (0..1) between two IPA strings. Language-agnostic."""
    return _similarity(a, b, method)


def distance(a: str, b: str, method: str = "weighted") -> float:
    """Distance (0..1) between two IPA strings. Language-agnostic."""
    return _distance(a, b, method)
