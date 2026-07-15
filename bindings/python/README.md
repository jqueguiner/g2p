# g2p-rs

Zero-dependency grapheme-to-phoneme for the **100 Whisper languages**, with a
Rust core exposed to Python.

```bash
pip install g2p-rs
```

```python
from g2p_rs import Model

m = Model.load("fr.g2p")
print(m.phonemize("bonjour"))            # bɔ̃ʒuʁ
print(m.phonemize_many(["chat", "eau"])) # ['ʃa', 'o']
```

Model blobs (`*.g2p`) are built from WikiPron / epitran / LLM data — see the
[main repo](https://github.com/jqueguiner/g2p). Download a language's `.g2p`
from the GitHub releases or build with `cargo run -p xtask -- build-all`.

## API

- `Model.load(path) -> Model` — load a `.g2p` file
- `Model.from_bytes(bytes) -> Model` — load from raw bytes
- `Model.phonemize(word) -> str` — IPA for one word
- `Model.phonemize_many(words) -> list[str]` — IPA for many

License: MIT OR Apache-2.0.
