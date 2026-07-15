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

### 1. Get a model file

Phonemization needs a compiled `.g2p` blob for the language. Build one from the
repo (models are not shipped in the wheel — they're data, not code):

```bash
git clone https://github.com/jqueguiner/g2p2 && cd g2p2
cargo run --release -p xtask -- fetch fr    # download WikiPron data
cargo run --release -p xtask -- build fr    # -> data/fr.g2p
```

### 2. Phonemize

```python
from g2p2 import Model

fr = Model.load("data/fr.g2p")
fr.phonemize("bonjour")                 # 'bɔ̃ʒuʁ'
fr.phonemize_many(["chat", "eau"])      # ['ʃa', 'o']

# logographic works too
zh = Model.load("data/zh.g2p")
zh.phonemize("你好")                    # 'ni²¹⁴⁻³⁵xɑʊ̯²¹⁴⁻²¹⁽⁴⁾'
```

### 3. Sound-alike similarity

```python
import g2p2

en = g2p2.Model.load("data/en.g2p")

# compare two words by how they SOUND (0..1)
en.word_similarity("light", "night")          # 0.95  (weighted, default)
en.word_similarity("light", "night", "fast")  # 0.75  (levenshtein)

# or compare IPA strings directly
g2p2.similarity("pat", "bat")                 # 0.967
g2p2.distance("pat", "bat")                   # 0.033
```

### API reference

| call | returns | notes |
|------|---------|-------|
| `Model.load(path)` | `Model` | load a `.g2p` file |
| `Model.from_bytes(b)` | `Model` | load from `bytes` |
| `Model.phonemize(word)` | `str` | IPA for one word |
| `Model.phonemize_many(words)` | `list[str]` | batch |
| `Model.word_similarity(a, b, method="weighted")` | `float` | phonemize both, then compare |
| `g2p2.similarity(ipa_a, ipa_b, method="weighted")` | `float` | 0..1, 1=identical |
| `g2p2.distance(ipa_a, ipa_b, method="weighted")` | `float` | 0..1, 0=identical |
| `g2p2.__version__` | `str` | package version |

`method`: `"weighted"` (default, articulatory-feature distance) or `"fast"` (Levenshtein).

### Tutorial: a "sounds-like" finder

```python
import g2p2

m = g2p2.Model.load("data/en.g2p")
vocab = ["knight", "night", "light", "bite", "note", "dog", "cat"]

def sounds_like(word, top=3):
    q = m.phonemize(word)
    scored = [(w, g2p2.similarity(q, m.phonemize(w))) for w in vocab if w != word]
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
cargo run -p xtask -- fetch-all      # download WikiPron TSVs (curl)
cargo run -p xtask -- build-all      # align (EM) -> train (Witten-Bell) -> .g2p
cargo run -p xtask -- say fr bonjour # phonemize from a compiled blob
```

## How coverage is achieved (100/100 languages)

| tier | languages | source |
|------|-----------|--------|
| WikiPron gold | 92 | scraped Wiktionary pronunciations |
| epitran silver | sn, so | epitran rule maps + Wikipedia wordlists |
| LLM silver | tt, ln, su | Sonnet-generated word→IPA (phonemic orthographies) |
| hani lexicon | zh, ja, yue | word→IPA exact match (+ OpenCC simplified fold, kanji supplement) |

## Pipeline

```
fetch → (silver) → align (many-to-many EM) → train (weighted Witten-Bell n-gram)
      → compile .g2p → load → phonemize
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
