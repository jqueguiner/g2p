#!/usr/bin/env python3
"""Generate per-language similarity calibration profiles for all 100 Whisper
languages served by g2p2.

Every knob in `calibration/<lang>.json` is derived from the language's
phonological *typology* (not a shared constant), so the calibrated scorer in
../src/calib.rs behaves per-language:

  axis (typology)                      -> knob
  -----------------------------------------------------------------
  speech rhythm (mora/syllable/stress) -> syllable_penalty
  stress position (initial/final)      -> onset_penalty
  morphology (isolating/agglutinative) -> length_ratio_penalty
  vowel-inventory richness             -> vowel_scale, blend
  consonant heaviness                  -> consonant_scale
  tone (contour/register/pitch-accent) -> tone_penalty
  nasal vowels                         -> nasal_penalty
  phonemic vowel length                -> length_penalty
  diphthong inventory                  -> diphthongs (syllable counting)

Values are a linguistically-informed first pass; tune any file and restart, or
regenerate: `python3 scripts/gen-calibration.py`.
"""
import json
import os

LANGS = (
    "en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms cs "
    "ro da hu ta no th ur hr bg lt la mi ml cy sk te fa lv bn sr az sl kn et mk "
    "br eu is hy ne mn bs kk sq sw gl mr pa si km sn yo so af oc ka be tg sd gu "
    "am yi lo uz fo ht ps tk nn mt sa lb my bo tl mg as tt haw ln ha ba jw su yue"
).split()

# --- typology axes ---------------------------------------------------------

# Rhythm. Default (not listed) = syllable-timed.
MORA = {"ja"}
STRESS_TIMED = {
    "en", "de", "nl", "sv", "no", "nn", "da", "ru", "pl", "bg", "uk", "be",
    "cs", "sk", "hr", "sr", "bs", "sl", "mk", "lb", "yi", "af", "sq", "el",
}

# Stress position. Default = free/other (moderate onset weight).
STRESS_INITIAL = {"fi", "hu", "cs", "sk", "is", "et", "lv", "fo", "mn"}
STRESS_FINAL = {"fr", "tr", "fa", "az", "uz", "kk", "tk", "hy", "so", "tt", "ba"}

# Morphology. Default = fusional.
ISOLATING = {"zh", "yue", "vi", "th", "lo", "my"}
AGGLUTINATIVE = {
    "tr", "fi", "hu", "et", "az", "uz", "kk", "tk", "ja", "ko", "ta", "te",
    "kn", "ml", "mn", "ka", "eu", "sw", "mg", "tl", "jw", "su", "am", "ba", "tt",
}

# Large vowel inventory / front-rounded vowels.
VOWEL_RICH = {
    "fr", "de", "nl", "sv", "no", "nn", "da", "fi", "hu", "et", "tr", "az",
    "uz", "en", "sq", "tk", "kk", "lb", "yue", "km",
}
# Rich consonant clusters / consonant-heavy phonotactics.
CONSONANT_HEAVY = {
    "ru", "pl", "cs", "sk", "uk", "be", "hr", "sr", "bs", "sl", "mk", "bg",
    "ka", "hy", "ar", "he", "ps", "cy",
}

# Tone, graded by type.
TONE_CONTOUR = {"zh", "yue", "vi", "th"}            # many contour tones
TONE_REGISTER = {"yo", "ha", "sn", "ln", "pa", "my", "lo"}  # level/register tone
PITCH_ACCENT = {"ja", "sr", "hr", "sv", "no", "nn", "lt", "lv"}  # pitch accent

# Nasal vowels.
NASAL_HEAVY = {"fr", "pt", "pl"}
NASAL_LIGHT = {"hi", "bn", "pa", "gu", "yo", "ur"}

# Phonemic vowel length.
VOWEL_LENGTH = {
    "fi", "et", "ja", "cs", "sk", "hu", "la", "ar", "mi", "haw", "sa", "lv",
    "lt", "cy", "mt", "kk",
}

# Diphthong inventories: base-char vowel pairs that form ONE syllable nucleus.
DIPH = {
    "en": ["aɪ", "eɪ", "ɔɪ", "aʊ", "oʊ", "əʊ", "ɪə", "eə", "ʊə", "ju"],
    "de": ["aɪ", "aʊ", "ɔʏ", "ɔɪ"],
    "nl": ["ɛi", "œy", "ʌu", "ɔu", "ɑu", "ɛɪ"],
    "da": ["aɪ", "ɔɪ", "aʊ", "ɐu", "ɛi"],
    "no": ["æɪ", "œʏ", "ɔy", "ɛi", "øy"],
    "nn": ["æɪ", "œʏ", "ɔy", "ɛi", "øy"],
    "sv": ["ɛi", "au", "ɔy"],
    "is": ["ei", "ou", "ai", "au", "œy", "ɔi"],
    "fo": ["ai", "ɔi", "ɛi", "ʉu", "œu", "ɛa"],
    "af": ["əi", "œy", "ɐu", "əu"],
    "lb": ["ai", "au", "ɜɪ", "oɪ", "əʊ"],
    "yi": ["ai", "ɔi", "ɛi", "ɔy"],
    "es": ["ai", "ei", "oi", "au", "eu", "ou", "ia", "ie", "io", "iu",
           "ua", "ue", "ui", "uo"],
    "it": ["ai", "ei", "oi", "au", "ɛi", "ɔi", "ja", "je", "jo", "wa", "we", "wo"],
    "pt": ["ai", "ei", "ɛi", "oi", "ɔi", "ui", "au", "ɐu", "eu", "ɛu", "ou", "iu"],
    "ca": ["ai", "ei", "oi", "au", "eu", "iu", "ou", "ɛu", "ɔi"],
    "ro": ["ai", "ei", "oi", "au", "eu", "iu", "ou", "ea", "oa"],
    "gl": ["ai", "ei", "oi", "au", "eu", "ou", "ui"],
    "oc": ["ai", "ei", "ɔi", "au", "ɔu", "eu"],
    "la": ["ae", "au", "oe", "ei", "eu", "ui"],
    "fr": [],
    "cs": ["ou", "au", "eu"],
    "sk": ["ou", "au", "eu", "ia", "ie", "iu"],
    "lt": ["ai", "au", "ei", "ui", "ie", "uo"],
    "lv": ["ai", "au", "ei", "ie", "uo", "oi", "ui"],
    "fi": ["ɑi", "ei", "oi", "ui", "æi", "øi", "ɑu", "eu", "ou", "iu", "ie", "uo", "yø"],
    "et": ["ɑi", "ei", "oi", "ui", "æi", "øi", "ɑu", "eu", "ou", "æu"],
    "cy": ["ai", "au", "ei", "eu", "oi", "ou", "ɨu", "əi"],
    "br": ["ai", "au", "ei", "ou", "ɛu", "ɔi"],
    "eu": ["ai", "ei", "oi", "au", "eu"],
    "mt": ["ai", "au", "ei", "ie", "ou", "ɔi"],
    "mi": ["ae", "ai", "ao", "au", "ei", "oe", "oi", "ou", "ɛi"],
    "haw": ["ai", "ae", "ao", "au", "ei", "eu", "iu", "oi", "ou"],
    "id": ["ai", "au", "oi", "ei"],
    "ms": ["ai", "au", "oi"],
    "mg": ["ai", "au", "ei", "ao"],
    "zh": ["ai", "ei", "au", "ou", "ia", "ie", "ua", "uo", "yɛ", "ye", "ao"],
    "yue": ["ai", "ɐi", "au", "ɐu", "ei", "ɵy", "iu", "ou", "ui", "ɔi"],
    "vi": ["ie", "ɯɤ", "uo", "ai", "au", "ao", "ɐi", "ɐu", "ɔi", "ɤi", "ui"],
    "th": ["ia", "ɯa", "ua", "ai", "au", "ao", "ɔi", "oi", "ui"],
    "my": ["ai", "au", "ei", "ou"],
    "lo": ["ai", "au", "ia", "ua", "ɯa"],
    "hi": ["ai", "au", "əi", "əu"],
    "ur": ["ai", "au", "əi", "əu"],
    "bn": ["ai", "au", "oi", "ou", "æe", "eu", "iu", "ei"],
    "as": ["ai", "au", "oi", "ou", "ei", "eu", "iu", "ui"],
    "ne": ["ai", "au", "ui", "ei", "oi"],
    "mr": ["ai", "au"],
    "gu": ["ai", "au", "əi", "əu"],
    "pa": ["ai", "au", "əi", "əu"],
    "ta": ["ai", "au"],
    "te": ["ai", "au"],
    "kn": ["ai", "au"],
    "ml": ["ai", "au"],
    "si": ["ai", "au"],
    "fa": ["ei", "ou", "ɑi", "ɑu"],
    "ps": ["ai", "əi", "aw", "ay"],
    "sd": ["ai", "au"],
    "el": ["ai", "ei", "oi"],
    "ka": ["ai", "ei", "oi", "au"],
    "hy": ["ai", "au", "ei", "oi"],
    "sq": ["ai", "au", "ei", "ie", "ua", "ye"],
    "ht": ["ui", "wa", "we", "wi"],
    "tl": ["ai", "au", "iw", "uy", "oi"],
    "sw": ["ai", "au"],
    "am": ["əi", "əu"],
    "ba": ["aj", "əj", "uj"],
    "tt": ["aj", "əj", "uj"],
    "kk": ["aj", "əj", "uj"],
    "tg": ["ai", "au", "ei"],
    "km": ["iə", "ɨə", "uə", "ei", "ou", "ae", "ao", "aə"],
    "bo": ["ai", "au", "eu", "iu"],
}

# --- typology -> knobs -----------------------------------------------------

def profile(code):
    # rhythm -> syllable count salience
    if code in MORA:
        syllable = 0.6
    elif code in STRESS_TIMED:
        syllable = 0.4      # unstressed syllables reduce/elide -> count less rigid
    else:
        syllable = 0.55     # syllable-timed: every syllable is crisp

    # stress position -> onset salience
    if code in STRESS_INITIAL:
        onset = 0.4
    elif code in STRESS_FINAL:
        onset = 0.25
    else:
        onset = 0.3

    # morphology -> length tolerance
    if code in ISOLATING:
        length_ratio = 0.3   # short words, length very meaningful
    elif code in AGGLUTINATIVE:
        length_ratio = 0.15  # naturally long, variable-length words
    else:
        length_ratio = 0.2

    vowel_scale = 1.25 if code in VOWEL_RICH else 1.15
    consonant_scale = 1.15 if code in CONSONANT_HEAVY else 1.0
    blend = 0.42 if code in VOWEL_RICH else 0.4

    if code in TONE_CONTOUR:
        tone = 0.9
    elif code in TONE_REGISTER:
        tone = 0.65
    elif code in PITCH_ACCENT:
        tone = 0.3
    else:
        tone = 0.0

    if code in NASAL_HEAVY:
        nasal = 0.2
    elif code in NASAL_LIGHT:
        nasal = 0.15
    else:
        nasal = 0.12

    length = 0.25 if code in VOWEL_LENGTH else 0.1

    return {
        "lang": code,
        "blend": round(blend, 3),
        "gap": 1.0,
        "nasal_penalty": nasal,
        "tone_penalty": tone,
        "length_penalty": length,
        "vowel_consonant_penalty": 0.8,
        "vowel_scale": vowel_scale,
        "consonant_scale": consonant_scale,
        "onset_penalty": onset,
        "length_ratio_penalty": length_ratio,
        "syllable_penalty": syllable,
        "diphthongs": DIPH.get(code, []),
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.normpath(os.path.join(here, "..", "calibration"))
    os.makedirs(out, exist_ok=True)

    default = profile("default")
    default["diphthongs"] = []
    with open(os.path.join(out, "default.json"), "w", encoding="utf-8") as f:
        json.dump(default, f, ensure_ascii=False, indent=2)
        f.write("\n")

    counts = {"tonal": 0, "nasal": 0, "length": 0, "diph": 0,
              "stress_timed": 0, "agglutinative": 0, "cons_heavy": 0}
    for code in LANGS:
        p = profile(code)
        counts["tonal"] += p["tone_penalty"] > 0
        counts["nasal"] += code in NASAL_HEAVY or code in NASAL_LIGHT
        counts["length"] += p["length_penalty"] > 0.1
        counts["diph"] += bool(p["diphthongs"])
        counts["stress_timed"] += code in STRESS_TIMED
        counts["agglutinative"] += code in AGGLUTINATIVE
        counts["cons_heavy"] += code in CONSONANT_HEAVY
        with open(os.path.join(out, f"{code}.json"), "w", encoding="utf-8") as f:
            json.dump(p, f, ensure_ascii=False, indent=2)
            f.write("\n")

    print(f"wrote {len(LANGS)} language profiles + default to {out}")
    print("  " + "  ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
