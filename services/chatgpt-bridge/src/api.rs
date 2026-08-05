use std::{io::Write, sync::Arc};

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Multipart, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;

use crate::{
    chatgpt::ChatGptClient,
    control::ControlToken,
    credentials::{CredentialStore, Credentials},
    pairing::PairingState,
};

const MAX_AUDIO_BYTES: usize = 25 * 1024 * 1024;
const EXTENSION_ORIGIN_PREFIX: &str = "chrome-extension://";

fn trusted_pairing_origin(origin: &str) -> bool {
    origin.starts_with(EXTENSION_ORIGIN_PREFIX)
}

#[derive(Clone)]
pub struct AppState {
    pub store: Arc<dyn CredentialStore>,
    pub pairing: Arc<PairingState>,
    pub client: ChatGptClient,
    pub default_language: String,
    pub api_token: ControlToken,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/status", get(status))
        .route("/internal/pair", post(pair))
        .route("/v1/audio/transcriptions", post(transcribe))
        .layer(
            tower_http::cors::CorsLayer::new()
                .allow_origin(tower_http::cors::AllowOrigin::predicate(|origin, _| {
                    origin.to_str().is_ok_and(trusted_pairing_origin)
                }))
                .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
                .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION])
                .allow_private_network(true),
        )
        .layer(DefaultBodyLimit::max(MAX_AUDIO_BYTES + 1024 * 1024))
        .with_state(state)
}

#[derive(Serialize)]
struct Health<'a> {
    status: &'a str,
    service: &'a str,
    version: &'a str,
}

async fn health() -> Json<Health<'static>> {
    Json(Health {
        status: "ok",
        service: "chatgpt-transcribe-connect",
        version: env!("CARGO_PKG_VERSION"),
    })
}

#[derive(Serialize)]
struct Status {
    connected: bool,
    pairing_active: bool,
    expires_at: Option<String>,
}

async fn status(State(state): State<AppState>) -> Result<Json<Status>, ApiError> {
    let credentials = state.store.load().map_err(ApiError::internal)?;
    Ok(Json(Status {
        connected: credentials.is_some(),
        pairing_active: state.pairing.is_active(),
        expires_at: credentials.and_then(|c| c.expires_at.clone()),
    }))
}

#[derive(Deserialize)]
struct PairRequest {
    pairing_code: String,
    access_token: String,
    cookies: std::collections::BTreeMap<String, String>,
    expires_at: Option<String>,
}

async fn pair(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<PairRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let origin = headers
        .get(header::ORIGIN)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if !trusted_pairing_origin(origin) {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "trusted_origin_required",
            "Pairing accepts only browser-extension requests",
        ));
    }
    if !state.pairing.consume(&payload.pairing_code) {
        return Err(ApiError::new(
            StatusCode::UNAUTHORIZED,
            "invalid_pairing_code",
            "Pairing code invalid, expired, or already used",
        ));
    }
    let credentials = Credentials {
        access_token: payload.access_token,
        cookies: payload.cookies,
        expires_at: payload.expires_at,
    };
    credentials.validate().map_err(ApiError::bad_request)?;
    state.store.save(&credentials).map_err(ApiError::internal)?;
    Ok(Json(serde_json::json!({"ok": true, "connected": true})))
}

async fn transcribe(
    State(state): State<AppState>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Result<Response, ApiError> {
    let authorized = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .is_some_and(|candidate| state.api_token.matches(candidate));
    if !authorized {
        return Err(ApiError::new(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "Bearer token required",
        ));
    }
    let mut audio: Option<NamedTempFile> = None;
    let mut language: Option<String> = None;
    let mut response_format = "json".to_string();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(ApiError::bad_request)?
    {
        match field.name().unwrap_or("") {
            "file" => {
                let bytes = field.bytes().await.map_err(ApiError::bad_request)?;
                if bytes.is_empty() || bytes.len() > MAX_AUDIO_BYTES {
                    return Err(ApiError::new(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        "invalid_audio",
                        "Audio must be between 1 byte and 25 MiB",
                    ));
                }
                let mut file = NamedTempFile::new().map_err(ApiError::internal)?;
                file.as_file_mut()
                    .write_all(&bytes)
                    .map_err(ApiError::internal)?;
                audio = Some(file);
            }
            "language" => language = Some(field.text().await.map_err(ApiError::bad_request)?),
            "response_format" => {
                response_format = field.text().await.map_err(ApiError::bad_request)?
            }
            _ => {}
        }
    }

    let audio = audio.ok_or_else(|| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "missing_file",
            "Multipart field 'file' is required",
        )
    })?;
    let mut credentials = state
        .store
        .load()
        .map_err(ApiError::internal)?
        .ok_or_else(|| {
            ApiError::new(
                StatusCode::UNAUTHORIZED,
                "not_connected",
                "Connect ChatGPT account first",
            )
        })?;
    let language = language
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or(&state.default_language);
    // Refresh short-lived access token only when pairing supplied session
    // cookies. Some browser profiles expose a valid bearer token but withhold
    // HttpOnly cookies; that transient pairing remains usable until expiry.
    if !credentials.cookies.is_empty() {
        state
            .client
            .refresh_credentials(&mut credentials)
            .await
            .map_err(ApiError::upstream)?;
        state.store.save(&credentials).map_err(ApiError::internal)?;
    }
    let result = state
        .client
        .transcribe(audio.path(), &credentials, language)
        .await
        .map_err(ApiError::upstream)?;

    if response_format == "text" {
        Ok((
            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            result.text,
        )
            .into_response())
    } else {
        Ok(Json(serde_json::json!({"text": result.text})).into_response())
    }
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }
    fn bad_request(error: impl std::fmt::Display) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            error.to_string(),
        )
    }
    fn internal(error: impl std::fmt::Display) -> Self {
        tracing::error!(%error, "internal error");
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            "Internal daemon error",
        )
    }
    fn upstream(error: impl std::fmt::Display) -> Self {
        Self::new(StatusCode::BAD_GATEWAY, "upstream_error", error.to_string())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(serde_json::json!({"error":{"type":self.code,"message":self.message}})),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Mutex, time::Duration};

    use axum::{
        body::{Body, to_bytes},
        http::{Method, Request, StatusCode},
    };
    use tower::ServiceExt;

    use super::*;

    #[derive(Default)]
    struct MemoryStore(Mutex<Option<Credentials>>);

    impl CredentialStore for MemoryStore {
        fn load(&self) -> anyhow::Result<Option<Credentials>> {
            Ok(self.0.lock().unwrap().clone())
        }
        fn save(&self, credentials: &Credentials) -> anyhow::Result<()> {
            *self.0.lock().unwrap() = Some(credentials.clone());
            Ok(())
        }
        fn delete(&self) -> anyhow::Result<()> {
            *self.0.lock().unwrap() = None;
            Ok(())
        }
    }

    fn test_app(store: Arc<MemoryStore>, pairing: Arc<PairingState>) -> Router {
        router(AppState {
            store,
            pairing,
            client: ChatGptClient::new().unwrap(),
            default_language: "en-US".into(),
            api_token: ControlToken::from_value_for_tests("test-api-token"),
        })
    }

    #[tokio::test]
    async fn health_and_status_work_without_credentials() {
        let app = test_app(
            Arc::new(MemoryStore::default()),
            Arc::new(PairingState::new(Duration::from_secs(30))),
        );
        let health = app
            .clone()
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(health.status(), StatusCode::OK);
        let status = app
            .oneshot(Request::get("/v1/status").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body = to_bytes(status.into_body(), 4096).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&body).unwrap()["connected"],
            false
        );
    }

    #[tokio::test]
    async fn transcription_requires_bearer_token() {
        let app = test_app(
            Arc::new(MemoryStore::default()),
            Arc::new(PairingState::new(Duration::from_secs(30))),
        );
        let request = Request::post("/v1/audio/transcriptions")
            .header(header::CONTENT_TYPE, "multipart/form-data; boundary=x")
            .body(Body::from("--x--\r\n"))
            .unwrap();
        assert_eq!(
            app.oneshot(request).await.unwrap().status(),
            StatusCode::UNAUTHORIZED
        );
    }

    #[tokio::test]
    async fn correct_api_token_reaches_request_validation() {
        let app = test_app(
            Arc::new(MemoryStore::default()),
            Arc::new(PairingState::new(Duration::from_secs(30))),
        );
        let request = Request::post("/v1/audio/transcriptions")
            .header(header::AUTHORIZATION, "Bearer test-api-token")
            .header(header::CONTENT_TYPE, "multipart/form-data; boundary=x")
            .body(Body::from("--x--\r\n"))
            .unwrap();
        assert_eq!(
            app.oneshot(request).await.unwrap().status(),
            StatusCode::BAD_REQUEST
        );
    }

    #[tokio::test]
    async fn wrong_token_is_rejected() {
        let app = test_app(
            Arc::new(MemoryStore::default()),
            Arc::new(PairingState::new(Duration::from_secs(30))),
        );
        let request = Request::post("/v1/audio/transcriptions")
            .header(header::AUTHORIZATION, "Bearer wrong-token")
            .header(header::CONTENT_TYPE, "multipart/form-data; boundary=x")
            .body(Body::from("--x--\r\n"))
            .unwrap();
        assert_eq!(
            app.oneshot(request).await.unwrap().status(),
            StatusCode::UNAUTHORIZED
        );
    }

    #[tokio::test]
    async fn extension_preflight_allows_authorization_header() {
        let app = test_app(
            Arc::new(MemoryStore::default()),
            Arc::new(PairingState::new(Duration::from_secs(30))),
        );
        let request = Request::builder()
            .method(Method::OPTIONS)
            .uri("/v1/audio/transcriptions")
            .header(header::ORIGIN, "chrome-extension://test-extension")
            .header(header::ACCESS_CONTROL_REQUEST_METHOD, "POST")
            .header(
                header::ACCESS_CONTROL_REQUEST_HEADERS,
                "authorization,content-type",
            )
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let allowed = response
            .headers()
            .get(header::ACCESS_CONTROL_ALLOW_HEADERS)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_ascii_lowercase();
        assert!(allowed.contains("authorization"));
    }

    #[tokio::test]
    async fn extension_can_pair_once() {
        let store = Arc::new(MemoryStore::default());
        let pairing = Arc::new(PairingState::new(Duration::from_secs(30)));
        let code = pairing.issue();
        let app = test_app(store.clone(), pairing);
        let payload = serde_json::json!({
            "pairing_code": code,
            "access_token": "test-access-token",
            "cookies": {"__Secure-authjs.session-token.0": "test-cookie"},
            "expires_at": null
        });
        let request = Request::post("/internal/pair")
            .header(header::ORIGIN, "chrome-extension://test-extension")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(payload.to_string()))
            .unwrap();
        let response = app.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(store.load().unwrap().is_some());

        let retry = Request::post("/internal/pair")
            .header(header::ORIGIN, "chrome-extension://test-extension")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(payload.to_string()))
            .unwrap();
        assert_eq!(
            app.oneshot(retry).await.unwrap().status(),
            StatusCode::UNAUTHORIZED
        );
    }

    #[tokio::test]
    async fn chatgpt_web_origin_cannot_pair() {
        let pairing = Arc::new(PairingState::new(Duration::from_secs(30)));
        let payload = serde_json::json!({
            "pairing_code": pairing.issue(),
            "access_token": "test",
            "cookies": BTreeMap::from([("__Secure-next-auth.session-token", "test")])
        });
        let app = test_app(Arc::new(MemoryStore::default()), pairing);
        let request = Request::post("/internal/pair")
            .header(header::ORIGIN, "https://chatgpt.com")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(payload.to_string()))
            .unwrap();
        assert_eq!(
            app.oneshot(request).await.unwrap().status(),
            StatusCode::FORBIDDEN
        );
    }

    #[tokio::test]
    async fn regular_web_origin_cannot_pair() {
        let pairing = Arc::new(PairingState::new(Duration::from_secs(30)));
        let payload = serde_json::json!({
            "pairing_code": pairing.issue(),
            "access_token": "test",
            "cookies": BTreeMap::from([("__Secure-next-auth.session-token", "test")])
        });
        let app = test_app(Arc::new(MemoryStore::default()), pairing);
        let request = Request::post("/internal/pair")
            .header(header::ORIGIN, "https://evil.example")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(payload.to_string()))
            .unwrap();
        assert_eq!(
            app.oneshot(request).await.unwrap().status(),
            StatusCode::FORBIDDEN
        );
    }
}
