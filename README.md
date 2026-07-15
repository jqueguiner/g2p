# g2p

<!-- Replace OWNER/g2p with your GitHub org/repo once pushed. -->
[![CI](https://github.com/OWNER/g2p/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/g2p/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/OWNER/g2p/branch/main/graph/badge.svg)](https://codecov.io/gh/OWNER/g2p)
[![Release](https://img.shields.io/github/v/release/OWNER/g2p?sort=semver)](https://github.com/OWNER/g2p/releases)
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

## Tests & coverage

```bash
cargo test --workspace
cargo llvm-cov -p g2p --summary-only   # >95% line coverage enforced in CI
```

## Data licenses

- WikiPron data: CC BY-SA. epitran maps: MIT. OpenCC tables: Apache-2.0.
- Code: MIT OR Apache-2.0.
