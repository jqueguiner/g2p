# g2p2-server

HTTP REST service around the zero-dependency [g2p2](../README.md) grapheme-to-phoneme engine.

Phonemize a word or a sequence of words in any of the 100 Whisper languages, with
optional **language auto-detection**, spoken-**number** expansion, **phonetic
similarity**, and an **alternative-names** ranking endpoint (find the candidates
that *sound* closest to a query).

Built on `axum` + `tokio`. Depends on the local runtime crate `g2p2-core`
(path dep, `numbers` feature on). It is a **detached workspace** so its async deps
never leak into the repo's intentionally zero-dependency runtime workspace.

## Run

```bash
# 1. get the model blobs (from the g2p2 `models-v2` GitHub release)
./scripts/fetch-models.sh fr en zh          # a few languages
# ./scripts/fetch-models.sh                  # or all 100

# 2. run
cargo run --release
# G2P_MODELS_DIR=./models  G2P_DEFAULT_LANG=en  G2P_BIND=0.0.0.0:8080
```

| Env | Default | Meaning |
|-----|---------|---------|
| `G2P_MODELS_DIR`   | `models`        | dir of `<whisper>.g2p` blobs |
| `G2P_NAMES_DIR`    | `names`         | dir of `<lang>.txt` first-name lists |
| `G2P_CALIBRATION_DIR` | `calibration` | dir of `<lang>.json` similarity profiles |
| `G2P_DEFAULT_LANG` | `en`            | fallback when detection fails |
| `G2P_BIND`         | `0.0.0.0:8080`  | listen address |

Models are **lazy-loaded** on first request per language and kept resident.

## Endpoints

### `GET /health`
```json
{ "status":"ok", "models_available":3, "models_loaded":1, "default_lang":"en" }
```

### `GET /languages`
All 100 Whisper languages with `iso`, `logographic`, `model_available`, `loaded`.

### `POST /g2p` (or `GET /g2p?text=...&lang=...&numbers=...`)
Phonemize a word or sequence. Omit `lang` to auto-detect.
```bash
curl -s localhost:8080/g2p -H 'content-type: application/json' \
  -d '{"text":"bonjour le monde 12","lang":"fr","numbers":true}'
```
```json
{
  "text":"bonjour le monde douze",
  "lang":"fr", "detected":false, "numbers_expanded":true,
  "words":[
    {"word":"bonjour","phonemes":"bɔ̃ʒuʁ"},
    {"word":"le","phonemes":"lə"},
    {"word":"monde","phonemes":"mɔ̃d"},
    {"word":"douze","phonemes":"duz"}
  ],
  "ipa":"bɔ̃ʒuʁ lə mɔ̃d duz"
}
```
With `lang` omitted, the response also carries a `detection` block
(`{lang, iso, script, confidence, reliable}`).

### `GET /detect?text=...`  ·  `POST /detect {text}`
Language detection (whatlang), mapped to a Whisper code.
```json
{ "lang":"fr", "iso":"fra", "script":"Latin", "confidence":0.97, "reliable":true }
```

### `POST /similarity`
Phonetic similarity in `0..1`. Phonemizes both sides first by default; set
`phonemize:false` to compare raw IPA. `method` is `weighted` (default, articulatory
feature distance) or `levenshtein`.
```bash
curl -s localhost:8080/similarity -H 'content-type: application/json' \
  -d '{"a":"Caitlin","b":"Katelyn","lang":"en","method":"weighted"}'
```
```json
{ "a_ipa":"...", "b_ipa":"...", "method":"weighted", "lang":"en",
  "similarity":0.94, "distance":0.06 }
```

### `POST /alternatives`
Rank candidate names by how close they *sound* to `query`. Omit `lang` to detect.
```bash
curl -s localhost:8080/alternatives -H 'content-type: application/json' -d '{
  "query":"Caitlin",
  "candidates":["Kaitlyn","Katelynn","Caitlyn","Katherine","Kaylin"],
  "lang":"en", "method":"weighted", "top_k":3, "min_similarity":0.5
}'
```
```json
{
  "query":"Caitlin", "query_ipa":"...", "lang":"en", "method":"weighted",
  "results":[
    {"name":"Caitlyn","ipa":"...","similarity":0.98},
    {"name":"Kaitlyn","ipa":"...","similarity":0.95},
    {"name":"Katelynn","ipa":"...","similarity":0.88}
  ]
}
```

### `POST /similar-names` (or `GET /similar-names?name=...&lang=...&top_k=...`)
Given **one** first name, return the phonetically closest first names from the
server's built-in corpus — no candidate list required (unlike `/alternatives`).
Omit `lang` to auto-detect.
```bash
curl -s localhost:8080/similar-names -H 'content-type: application/json' \
  -d '{"name":"Caitlin","lang":"en","top_k":5}'
```
```json
{
  "query":"Caitlin", "query_ipa":"keɪtlaɪn", "lang":"en", "method":"calibrated",
  "results":[
    {"name":"Kaitlyn","ipa":"kaitlaɪn","similarity":0.86},
    {"name":"Katelyn","ipa":"keɪtlən","similarity":0.80},
    {"name":"Kaylin","ipa":"keɪlɪn","similarity":0.75}
  ]
}
```
Fields: `method` (default **`calibrated`** — see below; or `weighted` / `levenshtein`),
`top_k` (default 10), `min_similarity` (default 0), `exclude_exact` (default true —
drops the query name itself), `gender` (see below), `calibration` (per-request
override, see below). Each result carries its `gender` (`m`/`f`/`u`).

**Gender filter** — `gender` forces or leaves neutral:
- `"m"` / `"male"` → male results only · `"f"` / `"female"` → female only
- omitted / anything else → **neutral** (all genders)
- unisex names always pass a male or female filter

Pass the query's own gender to get same-gender alternatives:
```bash
curl -s "localhost:8080/similar-names?name=Jean&lang=fr&gender=m"   # Jacques, Gilles… (no Jeanne)
```

**Popularity** — `popularity` (0..1, default 0) adds a name-frequency prior to
the ranking: `0` = pure phonetic (frequency only breaks ties), higher nudges
common names up. Each result carries its `frequency`.

Corpus lives in `names/<lang>.txt`, one `Name<TAB>gender<TAB>frequency` per line
(gender `m`/`f`/`u` and frequency both optional; `#` comments ok). Each name is
phonemized once with the language model and cached. The shipped corpora
(31k names across 42 languages) are built from a first-name **census** by
`scripts/build-names-from-census.py first_names.tsv` — country→language mapping,
gender + frequency aggregated per language. Add or edit any `<lang>.txt` and
restart (`G2P_NAMES_DIR`, default `./names`).

### `POST /similar-surnames` (or `GET /similar-surnames?name=...&lang=...`)
Same as `/similar-names` but over a **surname** corpus and with **no gender**
(surnames aren't gendered — results omit `gender`). Supports `method`, `top_k`,
`min_similarity`, `popularity`, `calibration`.
```bash
curl -s "localhost:8080/similar-surnames?name=Smith&lang=en&top_k=3"   # Smythe, Small, Still…
```
Corpus in `surnames/<lang>.txt` (`Name<TAB>u<TAB>frequency`), built from the
surname census by `scripts/build-names-from-census.py <surnames.tsv> --surname
--out surnames`. Env `G2P_SURNAMES_DIR` (default `./surnames`).

## Similarity calibration

Every similarity endpoint (`/similar-names`, `/similarity`, `/alternatives`)
defaults to `method: "calibrated"` — a **per-language** metric that blends the two
core signals and adds **per-phoneme-class** penalties. It fixes cases the fixed
core metrics get wrong: `weighted` over-scores coincidental feature overlap
(French `Jean /ʒɑ̃/` ↔ `Guy /ɡi/` → 0.8), `levenshtein` throws away real closeness.
It also reads nasal diacritics (`ɑ̃`), which core scores on the base `ɑ` alone.

**All 100 Whisper languages** ship a profile (`calibration/<lang>.json` + `default.json`),
generated by `scripts/gen-calibration.py`, which derives **every knob from the
language's phonological typology** — not a shared constant:

| typology axis | drives |
|---------------|--------|
| speech rhythm (mora / syllable / stress-timed) | `syllable_penalty` (ja 0.6, es 0.55, en 0.4) |
| stress position (initial / final) | `onset_penalty` (fi 0.4, fr 0.25) |
| morphology (isolating / agglutinative) | `length_ratio_penalty` (zh 0.3, tr 0.15) |
| vowel-inventory richness | `vowel_scale`, `blend` |
| consonant heaviness | `consonant_scale` (ru/ar 1.15) |
| tone (contour / register / pitch-accent) | `tone_penalty` (zh 0.9, yo 0.65, ja 0.3) |
| nasal vowels | `nasal_penalty` (fr 0.2, hi 0.15) |
| phonemic vowel length | `length_penalty` |
| diphthong inventory | `diphthongs` |

Values are a linguistically-informed first pass — tune any file and restart, or
regenerate with `python3 scripts/gen-calibration.py`. Inspect all live with
`GET /calibration`. Fields:

| field | meaning |
|-------|---------|
| `blend` | 0 = pure exact-phoneme (levenshtein-like) … 1 = pure articulatory feature distance |
| `gap` | insertion/deletion cost |
| `nasal_penalty` | extra cost when only one segment is nasalized |
| `tone_penalty` | extra cost when tone marks differ (set high for tonal langs, 0 for others) |
| `length_penalty` | extra cost when length (`ː`) differs |
| `vowel_consonant_penalty` | extra cost for a vowel↔consonant substitution |
| `vowel_scale` / `consonant_scale` | multiply the feature distance for vowel/consonant subs |
| `onset_penalty` | word-level: added when the two words' **first phonemes** differ (onset dominates name similarity) |
| `length_ratio_penalty` | word-level: scaled by `\|n-m\|/max(n,m)` — a short and a long name rarely sound alike |
| `syllable_penalty` | word-level: scaled by the relative **syllable-count** (vowel-nucleus) difference — Michel (2) vs Mickaël (3) are structurally apart |
| `diphthongs` | list of single-nucleus vowel pairs for this language (e.g. `["aɪ","eɪ","aʊ"]` for English). Empty for languages with no phonemic diphthongs (French `a-ɛ` = hiatus = 2 syllables) |

Vowel-nucleus counting is **language-specific**: a diphthong is one nucleus, a
hiatus is two. English `Michael /maɪkəl/` = 2 nuclei (the `aɪ` folds); French
`Mickaël /mikaɛl/` = 3 (`a-ɛ` stays split). The counter also honours the
non-syllabic mark `◌̯` (glide) when the model emits it. Onset, syllable count and
segmentation are **precomputed once per corpus name** (cached `Analyzed`) and once
per query, not re-derived on each comparison.

Tune live per request without editing files — send a partial `calibration` object:
```bash
# feature-only (reproduces core 'weighted'): Jean↔Guy jumps back to ~0.7
curl -s localhost:8080/similarity -H 'content-type: application/json' \
  -d '{"a":"Jean","b":"Guy","lang":"fr","calibration":{"blend":1.0,"nasal_penalty":0.0}}'
```
Per-language example — the *same* toned pair `ma˥`/`ma˩`: `zh` (tone_penalty 0.9) → 0.50,
`fr` (tone_penalty 0.0) → 0.70.

## Tests

```bash
cargo test          # unit + end-to-end
```

- **Unit** (`src/*.rs`, `#[cfg(test)]`): calibrated metric (nasal/tone/syllable/
  diphthong/onset behaviour, overrides), `Gender` parse+filter, `MethodArg`
  parsing & request defaults, `ApiError` status mapping, language detection.
- **End-to-end** (`tests/e2e.rs`): builds the real router over a temporary
  fixture — a toy in-memory `.g2p` model, a small gendered corpus, a calibration
  profile — and drives every endpoint over HTTP (`tower::ServiceExt::oneshot`):
  `/health`, `/languages`, `/g2p` (+404/+400), `/detect`, `/similarity`,
  `/alternatives`, `/similar-names` (+ gender filter m/f/neutral), `/calibration`.

The crate is split lib (`g2p2_server`, all logic + `build_router`) + a thin bin,
so tests exercise the exact router the binary serves.

## Notes
- Logographic languages (`zh`, `ja`, `yue`) resolve from the lexicon tier; detection
  falls back to script when whatlang can't pin a trigram model.
- Numeral expansion uses the core `numbers` feature (`12` → `douze`), 120+ languages.
