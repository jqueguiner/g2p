#!/usr/bin/env python3
"""Build the per-language first-name corpus (names/<lang>.txt) from the
`first_names` census table (name, name_ascii, country, gender, unisex, freq).

The census `name` column is inconsistently de-accented per country (French rows
often store "Sebastien", "Elisa"), but the *correct* accented spelling for a
name usually exists under the same `name_ascii` in some other row. So we:

  1. group every display spelling globally by `name_ascii`;
  2. for each language, pick the spelling that is **orthographically valid for
     that language** (its accent set for Latin scripts, its Unicode script for
     others) and carries the most native characters, breaking ties by frequency.

This re-accentuates names per language (Sébastien/Céline/François for fr,
Nadège… ; Cyrillic for ru/uk/bg; Greek for el; native script for zh/ja/ko/ar/…)
so the phonemizer sees language-correct orthography.

Output: `names/<lang>.txt` as `Name<TAB>gender<TAB>frequency`, freq-sorted.

    python3 scripts/build-names-from-census.py first_names.tsv [--min 40] [--cap 6000]
"""
import os
import sys
import unicodedata
from collections import defaultdict

MIN_FREQ = 40
CAP = 6000
UNISEX_SHARE = 0.20

STOPWORDS = {
    "king", "queen", "prince", "princesse", "princess", "super", "little",
    "maison", "dark", "france", "french", "une", "ma", "so", "just", "the",
    "leroy", "durand", "dubois", "garcia", "petite", "tonton", "coco", "baby",
    "mom", "dad", "papa", "mama", "boss", "love", "star", "lil", "big", "user",
    "test", "admin", "null", "none", "prince",
}

COUNTRY_LANG = {
    "US": "en", "GB": "en", "AU": "en", "NZ": "en", "IE": "en",
    "FR": "fr", "MC": "fr",
    "ES": "es", "MX": "es", "AR": "es", "CO": "es", "PE": "es", "VE": "es",
    "CL": "es", "EC": "es", "GT": "es", "CU": "es", "BO": "es", "DO": "es",
    "HN": "es", "PY": "es", "SV": "es", "NI": "es", "CR": "es", "PA": "es",
    "UY": "es",
    "DE": "de", "AT": "de",
    "IT": "it",
    "PT": "pt", "BR": "pt",
    "NL": "nl",
    "SE": "sv", "NO": "no", "DK": "da", "FI": "fi", "IS": "is",
    "RU": "ru", "PL": "pl", "CZ": "cs", "SK": "sk", "UA": "uk", "BG": "bg",
    "RS": "sr", "HR": "hr", "SI": "sl",
    "TR": "tr", "GR": "el", "RO": "ro", "HU": "hu", "ID": "id", "VN": "vi",
    "TH": "th", "JP": "ja", "KR": "ko", "CN": "zh", "TW": "zh", "IL": "he",
    "SA": "ar", "EG": "ar", "IR": "fa", "IN": "hi", "LT": "lt", "LV": "lv",
    "EE": "et", "AL": "sq", "AM": "hy", "GE": "ka", "AZ": "az",
}

# Allowed lowercase accented letters per Latin-script language.
LATIN_DIAC = {
    "fr": "àâäçéèêëîïôöùûüÿœæ",
    "es": "áéíóúüñ",
    "pt": "áàâãçéêíóôõú",
    "de": "äöüß",
    "it": "àèéìíîòóùú",
    "nl": "áéíóúàèëïöü",
    "sv": "åäöé",
    "no": "åæøéèóòâ",
    "da": "åæøé",
    "fi": "äöå",
    "et": "äöõüšž",
    "is": "áéíóúýþðæö",
    "pl": "ąćęłńóśźż",
    "cs": "áčďéěíňóřšťúůýž",
    "sk": "áäčďéíĺľňóôŕšťúýž",
    "hr": "čćđšž",
    "sl": "čšž",
    "ro": "ăâîșțşţ",
    "hu": "áéíóöőúüű",
    "tr": "çğıöşü",
    "sq": "ëç",
    "az": "çəğıöşü",
    "lt": "ąčęėįšųūž",
    "lv": "āčēģīķļņšūž",
    "id": "",
    "vi": ("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợ"
           "ùúủũụưừứửữựỳýỷỹỵđ"),
}
# Fallback accent set for any un-configured Latin-script language.
DEFAULT_LATIN = "àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿœšžčćđāēīū"

# Non-Latin languages → Unicode script ranges of the target orthography.
SCRIPT_RANGE = {
    "ru": [(0x0400, 0x04FF)], "uk": [(0x0400, 0x04FF)], "bg": [(0x0400, 0x04FF)],
    "sr": [(0x0400, 0x04FF)],
    "el": [(0x0370, 0x03FF)],
    "ar": [(0x0600, 0x06FF)], "fa": [(0x0600, 0x06FF)],
    "he": [(0x0590, 0x05FF)],
    "hi": [(0x0900, 0x097F)],
    "th": [(0x0E00, 0x0E7F)],
    "ka": [(0x10A0, 0x10FF)],
    "hy": [(0x0530, 0x058F)],
    "ja": [(0x3040, 0x30FF), (0x4E00, 0x9FFF)],
    "ko": [(0xAC00, 0xD7A3), (0x1100, 0x11FF)],
    "zh": [(0x4E00, 0x9FFF)],
}

BASE_OK = set("abcdefghijklmnopqrstuvwxyz -'.")

# Targeted accent fixes for names the census carries only de-accented (no correct
# spelling exists under any country, so the country-count heuristic can't recover
# them). Keyed by language then name_ascii. Mainly French word-initial "É-".
SUPPLEMENT = {
    "fr": {
        "elisa": "Élisa", "eliza": "Éliza", "eloise": "Éloïse", "elise": "Élise",
        "emilie": "Émilie", "emile": "Émile", "etienne": "Étienne", "eric": "Éric",
        "edouard": "Édouard", "eleonore": "Éléonore", "edith": "Édith",
        "evelyne": "Évelyne", "eugenie": "Eugénie", "eva": "Éva", "elie": "Élie",
    },
}


VOWELS = set("aeiouyàáâãäåèéêëìíîïòóôõöøùúûüýÿœæ")
ASCII_LOWER = set("abcdefghijklmnopqrstuvwxyz")


def is_namelike(name):
    if name.lower() in STOPWORDS:
        return False
    n = name.replace("-", "").replace("'", "").replace(" ", "").replace(".", "")
    if len(n) < 2 or len(name) > 24:
        return False
    if not any(c.isalpha() for c in n):
        return False
    # ASCII-only tokens must contain a vowel — drops initials like "Jl", "Kc".
    nl = n.lower()
    if all(c in ASCII_LOWER for c in nl) and not any(c in VOWELS for c in nl):
        return False
    return True


def in_ranges(cp, ranges):
    return any(lo <= cp <= hi for lo, hi in ranges)


def native_score(disp, lang):
    """`-1` if the spelling is invalid for `lang` (a char outside its script /
    an accent it doesn't use); else `1` if it carries any native (accented /
    in-script) character, `0` if plain ASCII. Binary on purpose: we prefer *a*
    native spelling over ASCII, then let frequency pick the clean one — counting
    native chars would reward corrupted longer spellings (e.g. "Сергейъ")."""
    ranges = SCRIPT_RANGE.get(lang)
    diac = LATIN_DIAC.get(lang, None if ranges else DEFAULT_LATIN)
    has_native = False
    for c in disp:
        cl = c.lower()
        if cl in BASE_OK:
            continue
        if ranges is not None:
            if in_ranges(ord(c), ranges):
                has_native = True
                continue
            return -1  # a non-ascii char outside the language's script
        if diac and cl in diac:
            has_native = True
            continue
        return -1  # accent not valid for this language
    return 1 if has_native else 0


# A native (accented / in-script) spelling is treated as CANONICAL — and thus
# preferred over the plain ASCII form — only when it is attested across at least
# this many distinct countries. Canonical accents (José 39, Sébastien 32,
# François 47, María 6, Élodie 3) appear in many countries; data-glitch accents
# (Dávid 2, Marküs 1, Sèbastien 1) appear in one or two.
MIN_COUNTRIES = 3


def best_display(ascii_key, lang, variants):
    """Pick the language-appropriate spelling among all global variants.
    `variants` maps a display spelling to `[freq, {country codes}]`."""
    fix = SUPPLEMENT.get(lang, {}).get(ascii_key)
    if fix:
        return fix
    # (freq, ncountries, native_bool, display) for spellings valid in `lang`
    valid = []
    for disp, (freq, ccs) in variants.items():
        ns = native_score(disp, lang)
        if ns >= 0:
            valid.append((freq, len(ccs), ns, disp))
    if not valid:
        asc = [(f, d) for d, (f, _) in variants.items() if all(c.lower() in BASE_OK for c in d)]
        return max(asc)[1] if asc else ascii_key.capitalize()

    # Non-Latin: require the native script (a Latin transliteration would not
    # phonemize in that language's model); pick the most widely-attested form.
    if lang in SCRIPT_RANGE:
        native = [v for v in valid if v[2] == 1]
        pool = native or valid
        return max(pool, key=lambda v: (v[1], v[0]))[3]

    # Latin: use a canonical accented spelling (native, in ≥ MIN_COUNTRIES
    # countries) — most widely attested wins; else the most frequent ASCII form.
    natives = [v for v in valid if v[2] == 1 and v[1] >= MIN_COUNTRIES]
    if natives:
        return max(natives, key=lambda v: (v[1], v[0]))[3]
    asciis = [v for v in valid if v[2] == 0]
    if asciis:
        return max(asciis, key=lambda v: v[0])[3]
    return max(valid, key=lambda v: (v[1], v[0]))[3]


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: build-names-from-census.py first_names.tsv [--min N] [--cap N]")
    src = sys.argv[1]
    args = sys.argv[2:]
    min_freq = int(args[args.index("--min") + 1]) if "--min" in args else MIN_FREQ
    cap = int(args[args.index("--cap") + 1]) if "--cap" in args else CAP

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.normpath(os.path.join(here, "..", "names"))
    os.makedirs(out_dir, exist_ok=True)

    # global: name_ascii -> {display spelling: [summed freq, {country codes}]}
    variants = defaultdict(lambda: defaultdict(lambda: [0, set()]))
    # per language: name_ascii -> {"m":freq,"f":freq,"uni":bool,"off":bool}
    agg = defaultdict(lambda: defaultdict(lambda: {"m": 0, "f": 0, "uni": False, "off": False}))
    has_official = set()  # languages that carry any official=t names

    with open(src, encoding="utf-8", errors="replace") as f:
        for line in f:
            col = line.rstrip("\n").split("\t")
            if len(col) < 11:
                continue
            name, ascii_key, cc, official, gender, unisex = (
                col[1], col[2].lower(), col[3], col[6], col[7], col[8]
            )
            try:
                freq = int(col[10])
            except ValueError:
                freq = 0
            if not name or name == "\\N" or not ascii_key or not is_namelike(name):
                continue
            is_off = official == "t"
            v = variants[ascii_key][name]
            v[0] += max(freq, 1)
            v[1].add(cc)
            lang = COUNTRY_LANG.get(cc)
            if not lang:
                continue
            if is_off:
                has_official.add(lang)
            e = agg[lang][ascii_key]
            if gender == "m":
                e["m"] += freq
            elif gender == "f":
                e["f"] += freq
            if unisex == "t":
                e["uni"] = True
            e["off"] = e["off"] or is_off

    total_written = 0
    summary = []
    for lang, names in sorted(agg.items()):
        official_lang = lang in has_official
        rows = []
        for ascii_key, e in names.items():
            # In languages that have official names, keep only official ones
            # (drops nicknames/junk). Languages with no official census data keep
            # all name-like entries.
            if official_lang and not e["off"]:
                continue
            tot = e["m"] + e["f"]
            if tot < min_freq:
                continue
            minor = min(e["m"], e["f"])
            if e["uni"] or (tot > 0 and minor / tot >= UNISEX_SHARE):
                g = "u"
            elif e["m"] >= e["f"]:
                g = "m"
            else:
                g = "f"
            disp = best_display(ascii_key, lang, variants[ascii_key])
            rows.append((tot, disp, g))
        rows.sort(key=lambda r: -r[0])
        rows = rows[:cap]
        if not rows:
            continue
        path = os.path.join(out_dir, f"{lang}.txt")
        with open(path, "w", encoding="utf-8") as fo:
            fo.write(f"# {lang} first names from the first_names census, re-accentuated "
                     f"per language (name<TAB>gender<TAB>frequency; freq-sorted; "
                     f"min={min_freq}, cap={cap}).\n")
            for tot, disp, g in rows:
                fo.write(f"{disp}\t{g}\t{tot}\n")
        total_written += len(rows)
        summary.append((lang, len(rows)))

    print(f"wrote {total_written} names across {len(summary)} languages to {out_dir}")
    print("  " + "  ".join(f"{l}:{n}" for l, n in sorted(summary, key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
