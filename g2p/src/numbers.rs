//! Numeral normalization (feature `numbers`).
//!
//! Spell digit tokens as words in a language before G2P — `"12"` in French
//! becomes `"douze"`, which then phonemizes correctly instead of the n-gram
//! trying to decode the raw digit graphemes. Backed by
//! [`num2words2-core`](https://crates.io/crates/num2words2-core) (120+
//! languages). Gated behind the `numbers` feature so the default runtime stays
//! zero-dependency; the Python wheel is built with it on.

use num2words2_core::{get_lang_by_key, strnum::python_int_parse};

/// Map a Whisper language code to a num2words2 key. They coincide for 99/100
/// languages; only Cantonese (`yue`) is absent there → read its numerals as
/// Mandarin (`zh`).
fn n2w_key(lang: &str) -> &str {
    match lang {
        "yue" => "zh",
        other => other,
    }
}

/// Spell a plain integer numeral `token` as cardinal words in `lang`.
///
/// Returns `None` when `token` is not a bare integer, or the language / number
/// is unsupported — so callers can fall back to the token unchanged.
///
/// ```
/// # #[cfg(feature = "numbers")] {
/// assert_eq!(g2p::numbers::spell_cardinal("12", "fr").as_deref(), Some("douze"));
/// assert_eq!(g2p::numbers::spell_cardinal("rue", "fr"), None);
/// # }
/// ```
pub fn spell_cardinal(token: &str, lang: &str) -> Option<String> {
    let n = python_int_parse(token)?;
    let lang = get_lang_by_key(n2w_key(lang))?;
    lang.to_cardinal(&n).ok()
}

/// Replace whitespace-separated integer tokens in `text` with their spelled
/// form in `lang`; every other token passes through unchanged.
///
/// ```
/// # #[cfg(feature = "numbers")] {
/// assert_eq!(g2p::numbers::expand_numbers("12 rue", "fr"), "douze rue");
/// # }
/// ```
pub fn expand_numbers(text: &str, lang: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for (i, tok) in text.split_whitespace().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        match spell_cardinal(tok, lang) {
            Some(words) => out.push_str(&words),
            None => out.push_str(tok),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cardinal_fr_en_es() {
        assert_eq!(spell_cardinal("12", "fr").as_deref(), Some("douze"));
        assert_eq!(spell_cardinal("42", "en").as_deref(), Some("forty-two"));
        assert_eq!(spell_cardinal("21", "es").as_deref(), Some("veintiuno"));
    }

    #[test]
    fn non_numeric_is_none() {
        assert_eq!(spell_cardinal("rue", "fr"), None);
        assert_eq!(spell_cardinal("", "fr"), None);
        assert_eq!(spell_cardinal("3.14", "fr"), None); // not a bare integer
    }

    #[test]
    fn yue_falls_back_to_mandarin() {
        assert_eq!(spell_cardinal("5", "yue"), spell_cardinal("5", "zh"));
    }

    #[test]
    fn expand_mixed_text() {
        assert_eq!(expand_numbers("12 rue", "fr"), "douze rue");
        assert_eq!(expand_numbers("no digits here", "en"), "no digits here");
    }
}
