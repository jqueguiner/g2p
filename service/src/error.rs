//! A single JSON error type that maps cleanly onto HTTP status codes.

use std::collections::BTreeSet;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
    /// Optional list of valid values (e.g. supported language codes).
    pub hint: Option<Vec<String>>,
}

impl ApiError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
            hint: None,
        }
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: msg.into(),
            hint: None,
        }
    }

    /// No model blob available for the requested language.
    pub fn no_model(lang: &str, available: &BTreeSet<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: format!("no model for language '{lang}'"),
            hint: Some(available.iter().cloned().collect()),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut body = json!({ "error": self.message });
        if let Some(hint) = self.hint {
            body["available_languages"] = json!(hint);
        }
        (self.status, Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;
    use std::collections::BTreeSet;

    #[test]
    fn status_codes() {
        assert_eq!(ApiError::bad_request("x").status, StatusCode::BAD_REQUEST);
        assert_eq!(ApiError::internal("x").status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(
            ApiError::no_model("zz", &BTreeSet::new()).status,
            StatusCode::NOT_FOUND
        );
    }

    #[test]
    fn into_response_keeps_status() {
        let resp = ApiError::bad_request("nope").into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}
