//! End-to-end tests: build the real router over a temporary fixture (a toy
//! in-memory `.g2p` model, a small gendered name corpus, a calibration profile)
//! and drive every endpoint over HTTP via `oneshot`.

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use g2p::model::{Model, EOS};
use g2p2_server::state::AppState;
use serde_json::{json, Value};
use tower::ServiceExt;

/// A toy model: each listed grapheme maps 1:1 to a phoneme, unigram n-gram so
/// the beam decodes each grapheme deterministically. `to_bytes()` → a real blob.
fn toy_model_bytes() -> Vec<u8> {
    let maps: &[(char, &str)] = &[
        ('a', "a"), ('e', "e"), ('i', "i"), ('o', "o"), ('u', "u"), ('y', "y"),
        ('b', "b"), ('c', "k"), ('d', "d"), ('f', "f"), ('g', "ɡ"), ('j', "ʒ"),
        ('k', "k"), ('l', "l"), ('m', "m"), ('n', "n"), ('p', "p"), ('r', "ʁ"),
        ('s', "s"), ('t', "t"), ('v', "v"), ('z', "z"), ('h', "h"),
    ];
    let mut tokens: Vec<(Box<str>, Box<str>)> =
        vec![(String::new().into(), String::new().into())]; // EOS = id 0
    let mut ngram: HashMap<Box<[u32]>, f32> = HashMap::new();
    ngram.insert(vec![EOS].into_boxed_slice(), -0.5);
    for (g, p) in maps {
        let id = tokens.len() as u32;
        tokens.push((g.to_string().into(), p.to_string().into()));
        ngram.insert(vec![id].into_boxed_slice(), -0.5);
    }
    let mut m = Model {
        tokens,
        order: 2,
        logo: false,
        max_chunk: 0,
        by_graph: HashMap::new(),
        ngram,
        backoff: HashMap::new(),
        unk: -5.0,
        lexicon: HashMap::new(),
    };
    m.index();
    m.to_bytes()
}

const DEFAULT_CALIB: &str = r#"{
  "lang":"default","blend":0.4,"gap":1.0,"nasal_penalty":0.12,"tone_penalty":0.0,
  "length_penalty":0.1,"vowel_consonant_penalty":0.8,"vowel_scale":1.2,
  "consonant_scale":1.0,"onset_penalty":0.3,"length_ratio_penalty":0.2,
  "syllable_penalty":0.5,"diphthongs":[]
}"#;

/// Build the router over a temp fixture. The returned `TempDir` must be kept
/// alive for the duration of the test (models are read lazily from disk).
fn fixture() -> (tempfile::TempDir, Router) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let models = root.join("models");
    let names = root.join("names");
    let calib = root.join("calibration");
    std::fs::create_dir_all(&models).unwrap();
    std::fs::create_dir_all(&names).unwrap();
    std::fs::create_dir_all(&calib).unwrap();

    std::fs::write(models.join("xx.g2p"), toy_model_bytes()).unwrap();
    std::fs::write(
        names.join("xx.txt"),
        "Ana\tf\nAnna\tf\nNana\tf\nBob\tm\nTom\tm\n",
    )
    .unwrap();
    std::fs::write(calib.join("default.json"), DEFAULT_CALIB).unwrap();

    let state = Arc::new(AppState::new(models, names, calib, "xx".into()));
    (dir, g2p2_server::build_router(state))
}

/// Send a request and return (status, parsed JSON body).
async fn call(app: &Router, req: Request<Body>) -> (StatusCode, Value) {
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let val: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, val)
}

fn get(uri: &str) -> Request<Body> {
    Request::get(uri).body(Body::empty()).unwrap()
}

fn post(uri: &str, body: Value) -> Request<Body> {
    Request::post(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

#[tokio::test]
async fn health_ok() {
    let (_d, app) = fixture();
    let (status, body) = call(&app, get("/health")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ok");
    assert_eq!(body["models_available"], 1);
}

#[tokio::test]
async fn languages_lists_100() {
    let (_d, app) = fixture();
    let (status, body) = call(&app, get("/languages")).await;
    assert_eq!(status, StatusCode::OK);
    let arr = body.as_array().unwrap();
    assert_eq!(arr.len(), 100);
    assert!(arr.iter().any(|l| l["whisper"] == "en"));
}

#[tokio::test]
async fn g2p_phonemizes_word_and_sequence() {
    let (_d, app) = fixture();
    let (status, body) = call(
        &app,
        post("/g2p", json!({"text":"ana bob","lang":"xx","numbers":false})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["lang"], "xx");
    assert_eq!(body["ipa"], "ana bob");
    assert_eq!(body["words"][0]["phonemes"], "ana");
    assert_eq!(body["words"][1]["phonemes"], "bob");
}

#[tokio::test]
async fn g2p_unknown_lang_is_404() {
    let (_d, app) = fixture();
    let (status, _) = call(&app, get("/g2p?text=ana&lang=zz")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn g2p_empty_text_is_400() {
    let (_d, app) = fixture();
    let (status, _) = call(&app, post("/g2p", json!({"text":"  ","lang":"xx"}))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn detect_returns_language_info() {
    let (_d, app) = fixture();
    let (status, body) = call(
        &app,
        get("/detect?text=the%20quick%20brown%20fox%20jumps%20over%20the%20lazy%20dog"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["iso"].is_string());
    assert!(body["confidence"].is_number());
}

#[tokio::test]
async fn similarity_raw_ipa() {
    let (_d, app) = fixture();
    let (_s, same) = call(
        &app,
        post("/similarity", json!({"a":"ana","b":"ana","phonemize":false})),
    )
    .await;
    assert_eq!(same["similarity"], 1.0);

    let (_s, diff) = call(
        &app,
        post("/similarity", json!({"a":"ana","b":"bob","phonemize":false})),
    )
    .await;
    assert!(diff["similarity"].as_f64().unwrap() < 1.0);
}

#[tokio::test]
async fn similarity_method_override_and_calibration() {
    let (_d, app) = fixture();
    let (status, body) = call(
        &app,
        post(
            "/similarity",
            json!({"a":"ana","b":"anna","phonemize":false,
                   "method":"calibrated","calibration":{"blend":1.0}}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["method"], "calibrated");
    assert!(body["similarity"].as_f64().unwrap() > 0.0);
}

#[tokio::test]
async fn alternatives_ranks_candidates() {
    let (_d, app) = fixture();
    let (status, body) = call(
        &app,
        post(
            "/alternatives",
            json!({"query":"ana","candidates":["anna","bob"],"lang":"xx"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let res = body["results"].as_array().unwrap();
    // "anna" (shares a/n) must outrank "bob"
    assert_eq!(res[0]["name"], "anna");
    assert!(
        res[0]["similarity"].as_f64().unwrap() > res[1]["similarity"].as_f64().unwrap()
    );
}

#[tokio::test]
async fn similar_names_from_corpus() {
    let (_d, app) = fixture();
    let (status, body) = call(&app, get("/similar-names?name=Ana&lang=xx&top_k=3")).await;
    assert_eq!(status, StatusCode::OK);
    let res = body["results"].as_array().unwrap();
    assert!(!res.is_empty());
    // query itself excluded; top match is a phonetic neighbour, not Ana
    assert_ne!(res[0]["name"], "Ana");
    assert!(res.iter().all(|r| r["gender"].is_string()));
    // Anna/Nana (share structure with Ana) should rank above Bob
    let top = res[0]["name"].as_str().unwrap();
    assert!(top == "Anna" || top == "Nana", "unexpected top: {top}");
}

#[tokio::test]
async fn similar_names_gender_filter() {
    let (_d, app) = fixture();

    // force female -> only female names
    let (_s, f) = call(&app, get("/similar-names?name=Ana&lang=xx&gender=f&top_k=10")).await;
    let fr = f["results"].as_array().unwrap();
    assert!(!fr.is_empty());
    assert!(fr.iter().all(|r| r["gender"] == "f"), "expected only f: {fr:?}");

    // force male -> only male names
    let (_s, m) = call(&app, get("/similar-names?name=Ana&lang=xx&gender=m&top_k=10")).await;
    let mr = m["results"].as_array().unwrap();
    assert!(mr.iter().all(|r| r["gender"] == "m"));

    // neutral (omitted) -> both genders present
    let (_s, n) = call(&app, get("/similar-names?name=Ana&lang=xx&top_k=10")).await;
    let genders: Vec<&str> = n["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["gender"].as_str().unwrap())
        .collect();
    assert!(genders.contains(&"f") && genders.contains(&"m"));
}

#[tokio::test]
async fn calibration_profiles_endpoint() {
    let (_d, app) = fixture();
    let (status, body) = call(&app, get("/calibration")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["default"].is_object());
    assert_eq!(body["default"]["blend"], 0.4);
}
