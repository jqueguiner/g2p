//! The 100 Whisper languages mapped to their WikiPron source.
//!
//! `pat` is an extra substring the WikiPron filename must contain, used to
//! disambiguate multi-script / multi-dialect languages (e.g. pick Latin pinyin
//! for Chinese, Cyrillic for Serbian, US English). Empty `pat` = take the
//! best-ranked file for the ISO code.
//!
//! `logo` marks logographic languages whose grapheme->phoneme mapping is not an
//! n-gram problem (kanji/hanzi need a word->reading lexicon); they are trained
//! for completeness but should be served from the lexicon tier.

pub struct Lang {
    pub whisper: &'static str,
    pub iso: &'static str,
    pub pat: &'static str,
    pub logo: bool,
}

const fn l(whisper: &'static str, iso: &'static str, pat: &'static str, logo: bool) -> Lang {
    Lang {
        whisper,
        iso,
        pat,
        logo,
    }
}

/// Whisper large-v3 language set (100). Five have no WikiPron data
/// (sn/so/tt/ln/su) — resolved via epitran silver.
pub const LANGS: &[Lang] = &[
    l("en", "eng", "us", false),
    l("zh", "cmn", "hani", true),
    l("de", "deu", "", false),
    l("es", "spa", "", false),
    l("ru", "rus", "", false),
    l("ko", "kor", "", false),
    l("fr", "fra", "", false),
    l("ja", "jpn", "", true),
    l("pt", "por", "", false),
    l("tr", "tur", "", false),
    l("pl", "pol", "", false),
    l("ca", "cat", "", false),
    l("nl", "nld", "", false),
    l("ar", "ara", "", false),
    l("sv", "swe", "", false),
    l("it", "ita", "", false),
    l("id", "ind", "", false),
    l("hi", "hin", "", false),
    l("fi", "fin", "", false),
    l("vi", "vie", "", false),
    l("he", "heb", "", false),
    l("uk", "ukr", "", false),
    l("el", "ell", "", false),
    l("ms", "msa", "", false),
    l("cs", "ces", "", false),
    l("ro", "ron", "", false),
    l("da", "dan", "", false),
    l("hu", "hun", "", false),
    l("ta", "tam", "", false),
    l("no", "nor", "", false),
    l("th", "tha", "", false),
    l("ur", "urd", "arab", false),
    l("hr", "hbs", "latn", false),
    l("bg", "bul", "", false),
    l("lt", "lit", "", false),
    l("la", "lat", "", false),
    l("mi", "mri", "", false),
    l("ml", "mal", "", false),
    l("cy", "cym", "", false),
    l("sk", "slk", "", false),
    l("te", "tel", "", false),
    l("fa", "fas", "", false),
    l("lv", "lav", "", false),
    l("bn", "ben", "", false),
    l("sr", "hbs", "cyrl", false),
    l("az", "aze", "latn", false),
    l("sl", "slv", "", false),
    l("kn", "kan", "", false),
    l("et", "est", "", false),
    l("mk", "mkd", "", false),
    l("br", "bre", "", false),
    l("eu", "eus", "", false),
    l("is", "isl", "", false),
    l("hy", "hye", "", false),
    l("ne", "nep", "", false),
    l("mn", "mon", "cyrl", false),
    l("bs", "hbs", "latn", false),
    l("kk", "kaz", "cyrl", false),
    l("sq", "sqi", "", false),
    l("sw", "swa", "", false),
    l("gl", "glg", "", false),
    l("mr", "mar", "", false),
    l("pa", "pan", "guru", false),
    l("si", "sin", "", false),
    l("km", "khm", "", false),
    l("sn", "sna", "", false), // no wikipron -> silver
    l("yo", "yor", "", false),
    l("so", "som", "", false), // no wikipron -> silver
    l("af", "afr", "", false),
    l("oc", "oci", "", false),
    l("ka", "kat", "", false),
    l("be", "bel", "", false),
    l("tg", "tgk", "cyrl", false),
    l("sd", "snd", "arab", false),
    l("gu", "guj", "", false),
    l("am", "amh", "", false),
    l("yi", "yid", "", false),
    l("lo", "lao", "", false),
    l("uz", "uzb", "latn", false),
    l("fo", "fao", "", false),
    l("ht", "hat", "", false),
    l("ps", "pus", "", false),
    l("tk", "tuk", "latn", false),
    l("nn", "nno", "", false),
    l("mt", "mlt", "", false),
    l("sa", "san", "", false),
    l("lb", "ltz", "", false),
    l("my", "mya", "", false),
    l("bo", "bod", "", false),
    l("tl", "tgl", "", false),
    l("mg", "mlg", "", false),
    l("as", "asm", "", false),
    l("tt", "tat", "", false), // no wikipron -> silver
    l("haw", "haw", "", false),
    l("ln", "lin", "", false), // no wikipron -> silver
    l("ha", "hau", "", false),
    l("ba", "bak", "", false),
    l("jw", "jav", "", false),
    l("su", "sun", "", false), // no wikipron -> silver
    l("yue", "yue", "", true),
];

/// Look up a language entry by Whisper code.
pub fn by_whisper(code: &str) -> Option<&'static Lang> {
    LANGS.iter().find(|l| l.whisper == code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covers_100_whisper_langs() {
        assert_eq!(LANGS.len(), 100);
    }

    #[test]
    fn lookup_hit_and_miss() {
        let fr = by_whisper("fr").expect("fr present");
        assert_eq!(fr.iso, "fra");
        assert!(!fr.logo);
        assert!(by_whisper("zh").unwrap().logo); // logographic flag set
        assert!(by_whisper("nonsense").is_none());
    }

    #[test]
    fn whisper_codes_unique() {
        let mut seen = std::collections::HashSet::new();
        for l in LANGS {
            assert!(seen.insert(l.whisper), "duplicate {}", l.whisper);
        }
    }
}
