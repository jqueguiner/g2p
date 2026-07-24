//! HTTP handlers. Every handler is thin: parse -> resolve language ->
//! call into the g2p2 core -> serialize.

use std::sync::Arc;

use axum::extract::{Query, State};
use axum::Json;
use g2p::Model;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::calib::{self, Analyzed, Calibration, CalibrationOverride};
use crate::error::ApiError;
use crate::lang_detect::{self, Detection};
use crate::state::{AppState, Gender};
use crate::types::*;

type St = State<Arc<AppState>>;

/// Score two IPA strings in `0..1` under the chosen method. `calibrated` uses
/// the resolved per-language calibration; the others fall through to core.
fn score(a: &str, b: &str, method: MethodArg, calib: &Calibration) -> f32 {
    match method {
        MethodArg::Levenshtein => g2p::similarity(a, b, g2p::Method::Levenshtein),
        MethodArg::Weighted => g2p::similarity(a, b, g2p::Method::Weighted),
        MethodArg::Calibrated => calib::similarity(a, b, calib),
    }
}

/// Like [`score`] but reusing precomputed [`Analyzed`] structure on both sides,
/// so a query scored against a whole corpus segments/counts each name only once.
fn score_analyzed(q: &Analyzed, cand: &Analyzed, method: MethodArg, calib: &Calibration) -> f32 {
    match method {
        MethodArg::Levenshtein => g2p::similarity(&q.ipa, &cand.ipa, g2p::Method::Levenshtein),
        MethodArg::Weighted => g2p::similarity(&q.ipa, &cand.ipa, g2p::Method::Weighted),
        MethodArg::Calibrated => calib::similarity_of(q, cand, calib),
    }
}

/// Resolve the calibration to use: the language profile (or default), with any
/// per-request override merged on top.
fn resolve_calib(
    st: &AppState,
    lang: Option<&str>,
    ov: &Option<CalibrationOverride>,
) -> Calibration {
    let base = match lang {
        Some(l) => st.calibration(l),
        None => st.calibration_default(),
    };
    match ov {
        Some(o) => base.merged(o),
        None => (*base).clone(),
    }
}

/// `GET /` — the single-page web UI (served from the same origin as the API).
pub async fn ui() -> axum::response::Html<&'static str> {
    axum::response::Html(include_str!("../static/index.html"))
}

/// `GET /health`
pub async fn health(State(st): St) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "models_available": st.available.len(),
        "models_loaded": st.loaded().len(),
        "default_lang": st.default_lang,
    }))
}

/// `GET /languages` — the full Whisper set with per-language model status.
pub async fn languages(State(st): St) -> Json<Vec<LanguageInfo>> {
    let loaded = st.loaded();
    let list = g2p::lang::LANGS
        .iter()
        .map(|l| LanguageInfo {
            whisper: l.whisper.to_string(),
            iso: l.iso.to_string(),
            logographic: l.logo,
            model_available: st.has_model(l.whisper),
            loaded: loaded.contains(l.whisper),
        })
        .collect();
    Json(list)
}

/// `GET /calibration` — all similarity calibration profiles (default + per-lang).
pub async fn calibration(State(st): St) -> Json<std::collections::BTreeMap<String, Calibration>> {
    Json(st.all_calibrations())
}

// ---- language resolution ----

/// Resolve the language to use: an explicit request code wins; otherwise
/// auto-detect, then fall back to the server default. Returns the chosen code,
/// the detection record (if detection ran), and whether it was auto-detected.
fn resolve_lang(
    st: &AppState,
    requested: &Option<String>,
    text: &str,
) -> Result<(String, Option<Detection>, bool), ApiError> {
    if let Some(code) = requested {
        if !st.has_model(code) {
            return Err(ApiError::no_model(code, &st.available));
        }
        return Ok((code.clone(), None, false));
    }
    let detection = lang_detect::detect(text);
    let chosen = detection
        .as_ref()
        .and_then(|d| d.lang.clone())
        .filter(|c| st.has_model(c))
        .unwrap_or_else(|| st.default_lang.clone());
    if !st.has_model(&chosen) {
        return Err(ApiError::bad_request(format!(
            "could not detect a supported language and default '{}' has no model",
            st.default_lang
        )));
    }
    Ok((chosen, detection, true))
}

/// Phonemize a name/phrase: phonemize each whitespace token and concatenate the
/// IPA with no separator, so phonetic comparison isn't polluted by space
/// "segments".
fn phonemize_name(model: &Model, s: &str) -> String {
    s.split_whitespace()
        .map(|w| g2p::phonemize(model, w))
        .collect::<Vec<_>>()
        .concat()
}

// ---- /g2p ----

#[derive(Deserialize)]
pub struct G2pQuery {
    text: String,
    lang: Option<String>,
    numbers: Option<bool>,
}

/// `GET /g2p?text=bonjour&lang=fr&numbers=true` — convenience wrapper.
pub async fn g2p_get(State(st): St, Query(q): Query<G2pQuery>) -> Result<Json<G2pResponse>, ApiError> {
    let req = G2pRequest {
        text: q.text,
        lang: q.lang,
        numbers: q.numbers.unwrap_or(true),
    };
    g2p_run(&st, req)
}

/// `POST /g2p` — phonemize a word or sequence of words.
pub async fn g2p_post(State(st): St, Json(req): Json<G2pRequest>) -> Result<Json<G2pResponse>, ApiError> {
    g2p_run(&st, req)
}

fn g2p_run(st: &AppState, req: G2pRequest) -> Result<Json<G2pResponse>, ApiError> {
    if req.text.trim().is_empty() {
        return Err(ApiError::bad_request("`text` is empty"));
    }
    let (lang, detection, detected) = resolve_lang(st, &req.lang, &req.text)?;
    let model = st.model(&lang)?;

    let text = if req.numbers {
        g2p::expand_numbers(&req.text, &lang)
    } else {
        req.text.clone()
    };

    let words: Vec<WordPhonemes> = text
        .split_whitespace()
        .map(|w| WordPhonemes {
            word: w.to_string(),
            phonemes: g2p::phonemize(&model, w),
        })
        .collect();

    let ipa = words
        .iter()
        .map(|w| w.phonemes.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    Ok(Json(G2pResponse {
        text,
        lang,
        detected,
        detection: if detected { detection } else { None },
        numbers_expanded: req.numbers,
        words,
        ipa,
    }))
}

// ---- /detect ----

#[derive(Deserialize)]
pub struct DetectQuery {
    text: String,
}

/// `GET /detect?text=...`
pub async fn detect_get(Query(q): Query<DetectQuery>) -> Result<Json<Detection>, ApiError> {
    detect_run(&q.text)
}

/// `POST /detect`
pub async fn detect_post(Json(req): Json<DetectRequest>) -> Result<Json<Detection>, ApiError> {
    detect_run(&req.text)
}

fn detect_run(text: &str) -> Result<Json<Detection>, ApiError> {
    if text.trim().is_empty() {
        return Err(ApiError::bad_request("`text` is empty"));
    }
    lang_detect::detect(text)
        .map(Json)
        .ok_or_else(|| ApiError::bad_request("could not detect language"))
}

// ---- /similarity ----

/// `POST /similarity` — phonetic similarity between two strings.
pub async fn similarity(State(st): St, Json(req): Json<SimilarityRequest>) -> Result<Json<SimilarityResponse>, ApiError> {
    let method = req.method;

    let (a_ipa, b_ipa, lang) = if req.phonemize {
        let (lang, _, _) = resolve_lang(&st, &req.lang, &req.a)?;
        let model = st.model(&lang)?;
        (
            phonemize_name(&model, &req.a),
            phonemize_name(&model, &req.b),
            Some(lang),
        )
    } else {
        // Raw IPA: no phonemization, but a requested `lang` still selects the
        // calibration profile (tone/nasal weights differ by language).
        (req.a.clone(), req.b.clone(), req.lang.clone())
    };

    let calib = resolve_calib(&st, lang.as_deref(), &req.calibration);
    let sim = score(&a_ipa, &b_ipa, method, &calib);
    Ok(Json(SimilarityResponse {
        a_ipa,
        b_ipa,
        method: method.as_str().to_string(),
        lang,
        similarity: sim,
        distance: 1.0 - sim,
    }))
}

// ---- /alternatives ----

/// `POST /alternatives` — rank candidate names by phonetic closeness to `query`.
pub async fn alternatives(State(st): St, Json(req): Json<AlternativesRequest>) -> Result<Json<AlternativesResponse>, ApiError> {
    if req.query.trim().is_empty() {
        return Err(ApiError::bad_request("`query` is empty"));
    }
    if req.candidates.is_empty() {
        return Err(ApiError::bad_request("`candidates` is empty"));
    }
    let method = req.method;

    let (lang, _, _) = resolve_lang(&st, &req.lang, &req.query)?;
    let model = st.model(&lang)?;
    let calib = resolve_calib(&st, Some(&lang), &req.calibration);

    let query_ipa = phonemize_name(&model, &req.query);

    let mut results: Vec<Alternative> = req
        .candidates
        .iter()
        .map(|name| {
            let ipa = phonemize_name(&model, name);
            let similarity = score(&query_ipa, &ipa, method, &calib);
            Alternative {
                name: name.clone(),
                ipa,
                similarity,
                gender: None,
                frequency: None,
            }
        })
        .filter(|a| a.similarity >= req.min_similarity)
        .collect();

    results.sort_by(|a, b| b.similarity.total_cmp(&a.similarity));
    if req.top_k > 0 {
        results.truncate(req.top_k);
    }

    Ok(Json(AlternativesResponse {
        query: req.query,
        query_ipa,
        lang,
        method: method.as_str().to_string(),
        results,
    }))
}

// ---- /similar-names ----

#[derive(Deserialize)]
pub struct SimilarNamesQuery {
    name: String,
    lang: Option<String>,
    method: Option<MethodArg>,
    top_k: Option<usize>,
    min_similarity: Option<f32>,
    gender: Option<String>,
    popularity: Option<f32>,
}

/// `GET /similar-names?name=Caitlin&lang=en&top_k=5&gender=f`
pub async fn similar_names_get(State(st): St, Query(q): Query<SimilarNamesQuery>) -> Result<Json<AlternativesResponse>, ApiError> {
    let req = SimilarNamesRequest {
        name: q.name,
        lang: q.lang,
        method: q.method.unwrap_or_default(),
        top_k: q.top_k.unwrap_or(0),
        min_similarity: q.min_similarity.unwrap_or(0.0),
        exclude_exact: true,
        gender: q.gender,
        popularity: q.popularity.unwrap_or(0.0),
        calibration: None,
    };
    similar_names_run(&st, req)
}

/// `POST /similar-names` — given ONE first name, return the phonetically closest
/// first names from the server's built-in corpus (no candidate list needed).
pub async fn similar_names_post(State(st): St, Json(req): Json<SimilarNamesRequest>) -> Result<Json<AlternativesResponse>, ApiError> {
    similar_names_run(&st, req)
}

fn similar_names_run(st: &AppState, req: SimilarNamesRequest) -> Result<Json<AlternativesResponse>, ApiError> {
    if req.name.trim().is_empty() {
        return Err(ApiError::bad_request("`name` is empty"));
    }
    // Default method is `calibrated` (per-language blend); see `calib`.
    let method = req.method;

    let (lang, _, _) = resolve_lang(st, &req.lang, &req.name)?;
    let model = st.model(&lang)?;
    let calib = resolve_calib(st, Some(&lang), &req.calibration);

    let index = st.name_index(&lang, &model);
    if index.is_empty() {
        return Err(ApiError::bad_request(format!(
            "no name corpus for '{lang}' — add {lang}.txt to the names dir"
        )));
    }

    // Analyze the query once (same diphthong set as the cached corpus); every
    // corpus entry is already analyzed.
    let query = calib::analyze(&phonemize_name(&model, &req.name), &calib.diphthongs);
    let qnorm = req.name.to_lowercase();
    // Force a gender with `m`/`f`; anything else (omitted, `u`, `any`, `neutral`)
    // leaves it neutral = no filter (returns all genders).
    let gender_filter = match req.gender.as_deref().map(str::trim) {
        Some(s) if s.eq_ignore_ascii_case("m") || s.eq_ignore_ascii_case("male") => {
            Some(Gender::Male)
        }
        Some(s) if s.eq_ignore_ascii_case("f") || s.eq_ignore_ascii_case("female") => {
            Some(Gender::Female)
        }
        _ => None,
    };

    // Candidates surviving the gender filter, scored phonetically + frequency.
    let scored: Vec<(Alternative, u32)> = index
        .iter()
        .filter(|e| !(req.exclude_exact && e.name.to_lowercase() == qnorm))
        .filter(|e| gender_filter.map_or(true, |g| e.gender.passes(g)))
        .map(|e| {
            let similarity = score_analyzed(&query, &e.phon, method, &calib);
            (
                Alternative {
                    name: e.name.clone(),
                    ipa: e.phon.ipa.clone(),
                    similarity,
                    gender: Some(e.gender.code().to_string()),
                    frequency: Some(e.freq),
                },
                e.freq,
            )
        })
        .filter(|(a, _)| a.similarity >= req.min_similarity)
        .collect();

    // Rank by phonetic score plus an optional popularity prior; ties break on
    // raw frequency so the more common name wins.
    let max_freq = scored.iter().map(|(_, f)| *f).max().unwrap_or(0).max(1) as f32;
    let pop = req.popularity.clamp(0.0, 1.0);
    let rank_key = |a: &Alternative, f: u32| a.similarity + pop * (f as f32 / max_freq);

    let mut scored = scored;
    scored.sort_by(|(a, fa), (b, fb)| {
        rank_key(b, *fb)
            .total_cmp(&rank_key(a, *fa))
            .then(fb.cmp(fa))
    });

    let k = if req.top_k == 0 { 10 } else { req.top_k };
    let results: Vec<Alternative> = scored.into_iter().take(k).map(|(a, _)| a).collect();

    Ok(Json(AlternativesResponse {
        query: req.name,
        query_ipa: query.ipa.clone(),
        lang,
        method: method.as_str().to_string(),
        results,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calib::Calibration;

    #[test]
    fn score_dispatch_all_methods() {
        let c = Calibration::default();
        for m in [MethodArg::Levenshtein, MethodArg::Weighted, MethodArg::Calibrated] {
            assert_eq!(score("aba", "aba", m, &c), 1.0, "identical must be 1.0 for {m:?}");
        }
        assert!(score("aba", "opo", MethodArg::Calibrated, &c) < 1.0);
    }
}

#[cfg(test)]
mod more_tests {
    use super::*;
    use crate::calib::{analyze, Calibration};

    #[test]
    fn score_analyzed_agrees_with_score() {
        let c = Calibration::default();
        let (a, b) = (analyze("aba", &c.diphthongs), analyze("abo", &c.diphthongs));
        for m in [MethodArg::Levenshtein, MethodArg::Weighted, MethodArg::Calibrated] {
            let s1 = score_analyzed(&a, &b, m, &c);
            let s2 = score("aba", "abo", m, &c);
            assert!((s1 - s2).abs() < 1e-6, "mismatch for {m:?}: {s1} vs {s2}");
        }
    }
}
