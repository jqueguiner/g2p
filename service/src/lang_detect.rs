//! Language detection built on `whatlang` (pure-Rust trigram + script model),
//! mapped onto g2p2's Whisper language codes.
//!
//! whatlang returns an ISO 639-3 code; g2p2's [`g2p::lang::LANGS`] table keys
//! models by Whisper code and carries the matching ISO. We bridge the two, with
//! a script tie-break for languages that share an ISO (Serbo-Croatian) and a
//! script-only fallback for text whatlang can't pin to a trigram model.

use serde::Serialize;

/// Resolved detection result, already mapped to a Whisper code when possible.
#[derive(Serialize, Clone)]
pub struct Detection {
    /// Whisper language code (e.g. `fr`), or `null` if it could not be mapped.
    pub lang: Option<String>,
    /// whatlang's ISO 639-3 guess (e.g. `fra`).
    pub iso: String,
    /// Detected script (e.g. `Latin`, `Cyrillic`, `Han`).
    pub script: String,
    /// whatlang confidence in `0.0..=1.0`.
    pub confidence: f64,
    /// whatlang's own reliability flag for this call.
    pub reliable: bool,
}

/// Detect the language of `text`. Returns `None` only for empty/degenerate input.
pub fn detect(text: &str) -> Option<Detection> {
    let info = whatlang::detect(text)?;
    let iso = info.lang().code().to_string();
    let script = format!("{:?}", info.script());
    let lang = whisper_from(&iso, info.script());
    Some(Detection {
        lang,
        iso,
        script,
        confidence: info.confidence(),
        reliable: info.is_reliable(),
    })
}

/// Map an ISO 639-3 code (+ script for tie-breaks) to a Whisper code that has
/// an entry in the g2p2 language table.
fn whisper_from(iso: &str, script: whatlang::Script) -> Option<String> {
    use whatlang::Script;

    // Serbo-Croatian macro-ISO `hbs` covers hr/bs/sr; split on script.
    if iso == "hbs" {
        return Some(match script {
            Script::Cyrillic => "sr",
            _ => "hr",
        }
        .to_string());
    }

    // Exact ISO match against the g2p2 table (first entry wins for shared ISOs).
    if let Some(l) = g2p::lang::LANGS.iter().find(|l| l.iso == iso) {
        return Some(l.whisper.to_string());
    }

    // Fallback: pick a language purely from the script for common writing systems.
    let by_script = match script {
        Script::Mandarin => "zh",
        Script::Hiragana | Script::Katakana => "ja",
        Script::Hangul => "ko",
        Script::Arabic => "ar",
        Script::Hebrew => "he",
        Script::Cyrillic => "ru",
        Script::Greek => "el",
        Script::Thai => "th",
        Script::Devanagari => "hi",
        _ => return None,
    };
    Some(by_script.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_language_fields() {
        let d = detect("bonjour tout le monde comment allez vous aujourd hui").unwrap();
        assert!(!d.iso.is_empty());
        assert!((0.0..=1.0).contains(&d.confidence));
        assert!(!d.script.is_empty());
    }

    #[test]
    fn empty_is_none() {
        assert!(detect("").is_none());
    }
}
