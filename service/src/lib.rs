//! g2p2-server library: the HTTP handlers, calibrated similarity, language
//! detection, and app state, plus [`build_router`] that wires them into an
//! `axum` `Router`. The binary (`main.rs`) is a thin wrapper; integration tests
//! build the same router against a temporary fixture.

pub mod calib;
pub mod error;
pub mod handlers;
pub mod lang_detect;
pub mod state;
pub mod types;

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use state::AppState;

/// Build the full API router over a shared [`AppState`].
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(handlers::ui))
        .route("/health", get(handlers::health))
        .route("/languages", get(handlers::languages))
        .route("/detect", get(handlers::detect_get).post(handlers::detect_post))
        .route("/g2p", get(handlers::g2p_get).post(handlers::g2p_post))
        .route("/similarity", post(handlers::similarity))
        .route("/alternatives", post(handlers::alternatives))
        .route(
            "/similar-names",
            get(handlers::similar_names_get).post(handlers::similar_names_post),
        )
        .route("/calibration", get(handlers::calibration))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(state)
}
