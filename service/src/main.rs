//! g2p2-server — HTTP REST front end for the zero-dependency g2p2 engine.
//!
//! Endpoints:
//!   GET  /health                  liveness + model counts
//!   GET  /languages               all 100 Whisper langs + per-lang model status
//!   GET  /detect?text=            language detection (whatlang -> Whisper code)
//!   POST /detect                  { text }
//!   GET  /g2p?text=&lang=&numbers= phonemize (query form)
//!   POST /g2p                     { text, lang?, numbers? }
//!   POST /similarity              { a, b, phonemize?, lang?, method?, calibration? }
//!   POST /alternatives            { query, candidates[], lang?, method?, top_k?, min_similarity?, calibration? }
//!   GET  /similar-names?name=&lang=&gender= closest first names from the corpus
//!   POST /similar-names           { name, lang?, method?, top_k?, min_similarity?, exclude_exact?, gender?, calibration? }
//!   GET  /calibration             per-language similarity calibration profiles
//!
//! Config via env:
//!   G2P_MODELS_DIR   directory of `<whisper>.g2p` blobs   (default: ./models)
//!   G2P_NAMES_DIR    directory of `<lang>.txt` name lists (default: ./names)
//!   G2P_CALIBRATION_DIR directory of `<lang>.json` profiles (default: ./calibration)
//!   G2P_DEFAULT_LANG fallback language                    (default: en)
//!   G2P_BIND         listen address                       (default: 0.0.0.0:8080)

use std::path::PathBuf;
use std::sync::Arc;

use g2p2_server::build_router;
use g2p2_server::state::AppState;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,tower_http=info".into()),
        )
        .init();

    let models_dir =
        PathBuf::from(std::env::var("G2P_MODELS_DIR").unwrap_or_else(|_| "models".into()));
    let names_dir =
        PathBuf::from(std::env::var("G2P_NAMES_DIR").unwrap_or_else(|_| "names".into()));
    let surnames_dir =
        PathBuf::from(std::env::var("G2P_SURNAMES_DIR").unwrap_or_else(|_| "surnames".into()));
    let calib_dir =
        PathBuf::from(std::env::var("G2P_CALIBRATION_DIR").unwrap_or_else(|_| "calibration".into()));
    let default_lang = std::env::var("G2P_DEFAULT_LANG").unwrap_or_else(|_| "en".into());
    let bind = std::env::var("G2P_BIND").unwrap_or_else(|_| "0.0.0.0:8080".into());

    let state = Arc::new(AppState::new(
        models_dir.clone(),
        names_dir.clone(),
        surnames_dir.clone(),
        calib_dir.clone(),
        default_lang,
    ));
    tracing::info!(
        dir = %surnames_dir.display(),
        langs = state.surname_langs().len(),
        "surname corpora loaded"
    );

    tracing::info!(
        dir = %calib_dir.display(),
        profiles = state.all_calibrations().len(),
        "similarity calibration profiles loaded"
    );

    let name_langs = state.name_langs();
    if name_langs.is_empty() {
        tracing::warn!(dir = %names_dir.display(), "no name corpus (*.txt) found — /similar-names disabled until you add one");
    } else {
        tracing::info!(dir = %names_dir.display(), langs = ?name_langs, "name corpora loaded");
    }

    if state.available.is_empty() {
        tracing::warn!(
            dir = %models_dir.display(),
            "no .g2p models found — run scripts/fetch-models.sh to download them"
        );
    } else {
        tracing::info!(dir = %models_dir.display(), count = state.available.len(), "models available");
    }

    let app = build_router(state);

    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .unwrap_or_else(|e| panic!("bind {bind}: {e}"));
    tracing::info!(%bind, "g2p2-server listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutdown");
}
