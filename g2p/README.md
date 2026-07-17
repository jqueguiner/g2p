# g2p2-core

Zero-dependency, CPU, pure-Rust **grapheme-to-phoneme** for the **100 Whisper
languages**. A pair joint **n-gram** with beam decoding plus a **lexicon tier**
for logographic languages. Models are compiled offline into compact `.g2p`
blobs. No neural net, no linear algebra, no external crates at runtime.

> The crate is named `g2p2-core` on crates.io (the name `g2p` was taken) but is
> imported as `g2p` — `use g2p::...`.

```rust
let model = g2p::Model::from_bytes(&std::fs::read("fr.g2p")?);
println!("{}", g2p::phonemize(&model, "bonjour")); // bɔ̃ʒuʁ
```

## Features

- `numbers` (off by default) — spell digits as words in-language before G2P
  (`"12"` → `"douze"`), 120+ languages, via
  [`num2words2-core`](https://crates.io/crates/num2words2-core). Enabling it
  adds that dependency; the default build stays zero-dependency.

```rust
# #[cfg(feature = "numbers")] {
assert_eq!(g2p::expand_numbers("12 rue", "fr"), "douze rue");
# }
```

Models, the build tool, and the Python package (`pip install g2p2`) live in the
[g2p2 repository](https://github.com/jqueguiner/g2p2).

## License

MIT OR Apache-2.0.
