//! Request and response payloads for the JSON API.

use serde::{Deserialize, Serialize};

use crate::calib::CalibrationOverride;
use crate::lang_detect::Detection;

/// Similarity method. `calibrated` (default) is the per-language blend defined in
/// `calib`; `weighted`/`levenshtein` fall through to the fixed core metrics.
#[derive(Deserialize, Default, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MethodArg {
    Levenshtein,
    Weighted,
    #[default]
    Calibrated,
}

impl MethodArg {
    pub fn as_str(self) -> &'static str {
        match self {
            MethodArg::Levenshtein => "levenshtein",
            MethodArg::Weighted => "weighted",
            MethodArg::Calibrated => "calibrated",
        }
    }
}

fn default_true() -> bool {
    true
}

// ---- /g2p ----

#[derive(Deserialize)]
pub struct G2pRequest {
    /// Word or whitespace-separated sequence of words to phonemize.
    pub text: String,
    /// Whisper language code. Omit/`null` to auto-detect from `text`.
    #[serde(default)]
    pub lang: Option<String>,
    /// Spell integer numerals as words before phonemizing (e.g. `12` -> `douze`).
    #[serde(default = "default_true")]
    pub numbers: bool,
}

#[derive(Serialize)]
pub struct WordPhonemes {
    pub word: String,
    pub phonemes: String,
}

#[derive(Serialize)]
pub struct G2pResponse {
    /// The (possibly numeral-expanded) text that was phonemized.
    pub text: String,
    /// Whisper code actually used.
    pub lang: String,
    /// `true` when `lang` came from auto-detection rather than the request.
    pub detected: bool,
    /// Detection detail, present only when auto-detection ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detection: Option<Detection>,
    /// Whether numeral expansion was applied.
    pub numbers_expanded: bool,
    pub words: Vec<WordPhonemes>,
    /// All word phoneme strings joined with a single space.
    pub ipa: String,
}

// ---- /detect ----

#[derive(Deserialize)]
pub struct DetectRequest {
    pub text: String,
}

// ---- /similarity ----

#[derive(Deserialize)]
pub struct SimilarityRequest {
    pub a: String,
    pub b: String,
    /// When `true` (default), `a`/`b` are graphemes to phonemize first;
    /// when `false`, they are treated as raw IPA.
    #[serde(default = "default_true")]
    pub phonemize: bool,
    /// Language for phonemization. Ignored when `phonemize` is `false`.
    /// Omit to auto-detect from `a`.
    #[serde(default)]
    pub lang: Option<String>,
    #[serde(default)]
    pub method: MethodArg,
    /// Per-request calibration override (only used by `method: calibrated`).
    #[serde(default)]
    pub calibration: Option<CalibrationOverride>,
}

#[derive(Serialize)]
pub struct SimilarityResponse {
    pub a_ipa: String,
    pub b_ipa: String,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lang: Option<String>,
    pub similarity: f32,
    pub distance: f32,
}

// ---- /alternatives ----

#[derive(Deserialize)]
pub struct AlternativesRequest {
    /// Reference name/word to match against.
    pub query: String,
    /// Candidate names/words to rank by phonetic closeness to `query`.
    pub candidates: Vec<String>,
    /// Language for phonemization. Omit to auto-detect from `query`.
    #[serde(default)]
    pub lang: Option<String>,
    #[serde(default)]
    pub method: MethodArg,
    /// Keep only the top-K results. `0`/omitted returns all.
    #[serde(default)]
    pub top_k: usize,
    /// Drop candidates below this similarity (`0.0..=1.0`).
    #[serde(default)]
    pub min_similarity: f32,
    /// Per-request calibration override (only used by `method: calibrated`).
    #[serde(default)]
    pub calibration: Option<CalibrationOverride>,
}

#[derive(Serialize)]
pub struct Alternative {
    pub name: String,
    pub ipa: String,
    pub similarity: f32,
    /// Gender of the matched name (`m`/`f`/`u`); present for corpus-based
    /// endpoints (`/similar-names`), absent for caller-supplied candidates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gender: Option<String>,
    /// Census frequency of the matched name (corpus endpoints only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency: Option<u32>,
}

#[derive(Serialize)]
pub struct AlternativesResponse {
    pub query: String,
    pub query_ipa: String,
    pub lang: String,
    pub method: String,
    pub results: Vec<Alternative>,
}

// ---- /similar-names ----

#[derive(Deserialize)]
pub struct SimilarNamesRequest {
    /// The first name to find phonetic neighbours for.
    pub name: String,
    /// Language whose name corpus + phonemizer to use. Omit to auto-detect.
    #[serde(default)]
    pub lang: Option<String>,
    /// Similarity method. Defaults to `calibrated` (per-language blend).
    #[serde(default)]
    pub method: MethodArg,
    /// Keep only the top-K matches. `0`/omitted defaults to 10.
    #[serde(default)]
    pub top_k: usize,
    /// Drop matches below this similarity (`0.0..=1.0`).
    #[serde(default)]
    pub min_similarity: f32,
    /// Exclude the query name itself (case-insensitive) from results.
    #[serde(default = "default_true")]
    pub exclude_exact: bool,
    /// Restrict results to a gender: `"m"` or `"f"` (unisex names always pass).
    /// Omit for no filter. Pass the query's own gender to get same-gender
    /// alternatives.
    #[serde(default)]
    pub gender: Option<String>,
    /// Weight (0..1) of a name-popularity prior added to the phonetic score for
    /// ranking. `0` (default) = pure phonetic, frequency only breaks ties;
    /// higher nudges common names up.
    #[serde(default)]
    pub popularity: f32,
    /// Per-request calibration override (only used by `method: calibrated`).
    #[serde(default)]
    pub calibration: Option<CalibrationOverride>,
}

// ---- /languages ----

#[derive(Serialize)]
pub struct LanguageInfo {
    pub whisper: String,
    pub iso: String,
    pub logographic: bool,
    pub model_available: bool,
    pub loaded: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn method_arg_parse_and_default() {
        let p = |s: &str| serde_json::from_str::<MethodArg>(s).unwrap();
        assert_eq!(p("\"levenshtein\""), MethodArg::Levenshtein);
        assert_eq!(p("\"weighted\""), MethodArg::Weighted);
        assert_eq!(p("\"calibrated\""), MethodArg::Calibrated);
        assert_eq!(MethodArg::default(), MethodArg::Calibrated);
        assert_eq!(MethodArg::Weighted.as_str(), "weighted");
    }

    #[test]
    fn similar_names_request_defaults() {
        let r: SimilarNamesRequest = serde_json::from_str(r#"{"name":"Ana"}"#).unwrap();
        assert_eq!(r.method, MethodArg::Calibrated);
        assert!(r.exclude_exact);
        assert_eq!(r.top_k, 0);
        assert!(r.gender.is_none());
        assert!(r.calibration.is_none());
    }
}
