# g2p

[![CI](https://github.com/jqueguiner/g2p2/actions/workflows/ci.yml/badge.svg)](https://github.com/jqueguiner/g2p2/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jqueguiner/g2p2/branch/main/graph/badge.svg)](https://codecov.io/gh/jqueguiner/g2p2)
[![Release](https://img.shields.io/github/v/release/jqueguiner/g2p2?sort=semver)](https://github.com/jqueguiner/g2p2/releases)
![Rust](https://img.shields.io/badge/rust-stable-orange?logo=rust)
![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)

Zero-dependency, CPU, pure-Rust **grapheme-to-phoneme** for the **100 Whisper languages**.

The runtime crate (`g2p/`) is `std`-only — a pair joint **n-gram** with beam
decoding plus a **lexicon tier** for logographic languages. Models are compiled
offline into compact `.g2p` blobs. No neural net, no linear algebra, no external
crates at runtime.

## Quick start

```bash
# phonemize with a prebuilt model blob
g2p data/fr.g2p bonjour merci
# bɔ̃ʒuʁ
# mɛʁsi

echo "水 日本" | g2p data/zh.g2p
```

As a library:

```rust
let model = g2p::Model::from_bytes(&std::fs::read("fr.g2p")?);
println!("{}", g2p::phonemize(&model, "bonjour")); // bɔ̃ʒuʁ
```

## Python (`g2p2`)

Native binding (PyO3, no runtime deps). One prebuilt wheel per platform —
Linux/macOS/Windows, Python 3.8+.

```bash
pip install g2p2
```

### Quickstart — just pick a language

The default wheel is lean; a language's model is **downloaded and cached on
first use** (from the `models-v2` GitHub release). Since v0.2.0 the models
carry the full attested lexicon — French alone is 1.6M words — which put the
complete set past what a single PyPI wheel can bundle.

For a fully offline install, ship every model at install time instead:

```bash
pip install "g2p2[all]"
```

(`$G2P2_MODELS` pointing at a directory of `.g2p` files still overrides
everything, for air-gapped or custom-model setups.)

```python
import g2p2

g2p2.phonemize("hello", language="en")                # 'hɛloʊ'
g2p2.phonemize("bonjour", language="fr")              # 'bɔ̃ʒuʁ'
g2p2.phonemize("你好", language="zh")                 # 'ni²¹⁴⁻³⁵xɑʊ̯…'
g2p2.phonemize_many(["chat", "eau"], language="fr")   # ['ʃa', 'o']
```

Language codes are the 100 Whisper codes (`en`, `fr`, `zh`, `de`, `ja`, …).

### Sound-alike similarity

```python
# compare two words by how they SOUND (0..1), in a given language
g2p2.word_similarity("light", "night", language="en")                 # 0.95  weighted (default)
g2p2.word_similarity("light", "night", language="en", method="fast")  # 0.75  levenshtein

# or compare IPA strings directly (language-agnostic)
g2p2.similarity("pat", "bat")                          # 0.967
g2p2.distance("pat", "bat")                            # 0.033
```

### Explicit model files (offline / bundled)

Skip the download by pointing at a directory of `.g2p` blobs, or load one
directly:

```python
import os
os.environ["G2P2_MODELS"] = "/path/to/models"   # dir with en.g2p, fr.g2p, …
# ...or fully explicit:
from g2p2 import Model
m = Model.load("data/fr.g2p")
m.phonemize("bonjour")                            # 'bɔ̃ʒuʁ'
```

Build blobs yourself: `cargo run --release -p xtask -- build fr` → `data/fr.g2p`.

**Env vars:** `G2P2_MODELS` (local model dir, checked first) · `G2P2_MODELS_URL`
(download base, defaults to the GitHub release).

### API reference

| call | returns | notes |
|------|---------|-------|
| `g2p2.phonemize(word, language)` | `str` | auto-loads the language model |
| `g2p2.phonemize_many(words, language)` | `list[str]` | batch |
| `g2p2.word_similarity(a, b, language, method="weighted")` | `float` | phonemize both, compare |
| `g2p2.similarity(ipa_a, ipa_b, method="weighted")` | `float` | 0..1, 1=identical |
| `g2p2.distance(ipa_a, ipa_b, method="weighted")` | `float` | 0..1, 0=identical |
| `g2p2.get_model(language)` | `Model` | cached `Model` for a language |
| `Model.load(path)` / `Model.from_bytes(b)` | `Model` | explicit load |
| `Model.phonemize(word)` / `.phonemize_many(words)` | `str` / `list[str]` | on a loaded model |
| `Model.word_similarity(a, b, method="weighted")` | `float` | on a loaded model |

`method`: `"weighted"` (default, articulatory-feature distance) or `"fast"` (Levenshtein).

### Tutorial: a "sounds-like" finder

```python
import g2p2

vocab = ["knight", "night", "light", "bite", "note", "dog", "cat"]

def sounds_like(word, lang="en", top=3):
    scored = [(w, g2p2.word_similarity(word, w, language=lang)) for w in vocab if w != word]
    return sorted(scored, key=lambda x: -x[1])[:top]

sounds_like("nite")
# [('night', 1.0), ('knight', 1.0), ('light', 0.95)]  ← homophones score 1.0
```

Same idea powers fuzzy search, rhyme detection, homophone/typo correction, and
cross-language cognate matching (all languages share the IPA space).

## Building the models

The `xtask` build tool fetches data and compiles blobs (build-time only — its
deps never enter the runtime crate).

```bash
# from a kaikki.org Wiktionary dump (the v2 models' primary source)
cargo run -p xtask -- kaikki fr fr.jsonl.gz data/fr.tsv
cargo run -p xtask -- build  fr data/fr.tsv   # align (EM) -> train n-gram + lexicon -> .g2p

# or from WikiPron (English-Wiktionary scrape), still supported
cargo run -p xtask -- fetch-all
cargo run -p xtask -- build-all
cargo run -p xtask -- say fr bonjour           # phonemize from a compiled blob
```

## How coverage is achieved (100/100 languages)

Since **models-v2** the primary source is [kaikki.org](https://kaikki.org)
wiktextract dumps — each language's *own* Wiktionary edition, far richer than
the English-Wiktionary scrape WikiPron uses. Every attested word is stored in
the lexicon (French alone is 1.6M), so the n-gram only decodes out-of-vocabulary
words.

| tier | languages | source |
|------|-----------|--------|
| kaikki native edition | 19 | a language's own Wiktionary (de 836k words, es 829k, ru 379k, zh 207k, en 94k, …) |
| kaikki cross-edition merge | 65 | the same language pooled across up to 14 editions, deduped; LLM-validated (0 wrong-language) |
| WikiPron + silver | 16 | English-Wiktionary scrape, plus epitran/LLM silver for the few languages with no dump |
| hani lexicon | zh, ja, yue | word→IPA exact match (+ OpenCC simplified fold, kanji supplement) |

## Pipeline

```
kaikki|fetch → (silver) → align (many-to-many EM) → train (weighted Witten-Bell n-gram)
             → full-lexicon fill → compile .g2p → load → phonemize
```

- **Alignment**: forward-backward EM, Viterbi to joint tokens, parallelized with `std::thread`.
- **Model**: interpolated n-gram stored per-gram (runtime backoff is a clean recursion), quantized to varint ids + i16 logprobs.
- **Decode**: exact lexicon → (logographic ? per-char lexicon : n-gram beam) → fallback.

## Phonetic similarity

Score how similar two words *sound* by comparing their IPA. Two methods, chosen
by the caller; **`Weighted` is the default (better)**:

- **`Weighted`** — substitution cost = articulatory feature distance (p/b cost
  less than p/k). Graded, phonetically meaningful.
- **`Levenshtein`** — 0/1 per differing phoneme. Faster, coarse.

```rust
use g2p::{phonemize, similarity, Method};
let a = phonemize(&model, "light");      // laɪt
let b = phonemize(&model, "night");      // naɪt
similarity(&a, &b, Method::Weighted);    // 0.95   (default)
similarity(&a, &b, Method::Levenshtein); // 0.75
```

```python
m.word_similarity("light", "night")          # 0.95  (weighted default)
m.word_similarity("light", "night", "fast")  # 0.75  (levenshtein)
g2p2.similarity("pat", "bat")                # 0.967
```

### Benchmark

`cargo run --release -p xtask -- bench data/en.g2p` (single core):

| method | speed | throughput | distinct scores | near>far |
|--------|-------|-----------|-----------------|----------|
| **Weighted** (default) | ~600 ns/op | 1.7 M ops/s/core | **186** | 12/12 |
| Levenshtein (fast) | ~395 ns/op | 2.5 M ops/s/core | 13 | 12/12 |

Both rank near-vs-far pairs correctly; **Weighted gives ~14× finer resolution**
(186 vs 13 distinct scores) — it distinguishes degrees of similarity that
Levenshtein flattens, at ~1.5× the cost. **Hardware:** CPU-only, single-thread,
O(n·m) DP per pair (n,m = phonemes/word) → a few KB freed immediately, no heap
growth; embarrassingly parallel across pairs.

## Tests & coverage

```bash
cargo test --workspace
cargo llvm-cov -p g2p --summary-only   # >95% line coverage enforced in CI + pre-commit
```

A git **pre-commit hook** (`.githooks/pre-commit`) runs fmt + clippy + tests and
fails the commit if `g2p` line coverage drops below 95%. Enable it once:

```bash
git config core.hooksPath .githooks
```

## Data licenses

- WikiPron data: CC BY-SA. epitran maps: MIT. OpenCC tables: Apache-2.0.
- Code: MIT OR Apache-2.0.
