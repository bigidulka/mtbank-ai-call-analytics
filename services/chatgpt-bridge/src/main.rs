mod api;
mod chatgpt;
mod config;
mod control;
mod credentials;
mod pairing;
mod service;

use std::{sync::Arc, time::Duration};

use std::net::IpAddr;

use anyhow::{Context, Result, bail};
use clap::Parser;
use config::{Cli, Command};
use control::ControlToken;
use credentials::{CredentialStore, OsCredentialStore};
use pairing::PairingState;

/// Upstream only ever allows loopback binds. This fork additionally allows
/// unspecified (container-internal `0.0.0.0`/`::`) and private/ULA addresses
/// so the daemon can run inside an isolated Docker network and be reached by
/// sibling containers over Docker DNS. Genuinely public/global addresses are
/// still rejected. The daemon must never be published to a public interface
/// by the surrounding container/network configuration; bearer-token auth on
/// `/v1/audio/transcriptions` and origin-checked one-time pairing codes on
/// `/internal/pair` are unaffected by this and remain the primary defenses.
fn is_bind_address_allowed(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.is_loopback() || v4.is_unspecified() || v4.is_private(),
        IpAddr::V6(v6) => v6.is_loopback() || v6.is_unspecified() || v6.is_unique_local(),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "chatgpt_transcribe_connect=info".into()),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();
    if !is_bind_address_allowed(&cli.listen.ip()) {
        bail!(
            "refusing public listen address: {} (loopback, unspecified, or private/ULA required)",
            cli.listen
        );
    }
    let config_dir = config::config_dir()?;
    let store: Arc<dyn CredentialStore> = Arc::new(OsCredentialStore::new(
        config_dir.join("credentials.json.b64"),
    ));
    let control_token = ControlToken::load_or_create(&config_dir.join("control-token"))?;
    let api_token = ControlToken::load_or_create(&config_dir.join("api-token"))?;

    match cli.command.unwrap_or(Command::Serve) {
        Command::Serve => serve(cli.listen, cli.language, store, control_token, api_token).await,
        Command::Pair => {
            let code = issue_via_running_daemon(cli.listen, &control_token).await?;
            println!("Pairing code: {code}");
            println!("Open extension, paste code, then click Connect.");
            Ok(())
        }
        Command::Status => {
            match store.load()? {
                Some(c) => println!(
                    "Connected{}",
                    c.expires_at
                        .as_ref()
                        .map(|e| format!("; expires {e}"))
                        .unwrap_or_default()
                ),
                None => println!("Not connected"),
            }
            Ok(())
        }
        Command::ApiTokenPath => {
            println!("{}", config_dir.join("api-token").display());
            Ok(())
        }
        Command::Logout => {
            store.delete()?;
            println!("Credentials removed");
            Ok(())
        }
        Command::InstallService => {
            println!("Installed: {}", service::install()?.display());
            Ok(())
        }
        Command::UninstallService => {
            service::uninstall()?;
            println!("Background service removed");
            Ok(())
        }
    }
}

async fn serve(
    listen: std::net::SocketAddr,
    language: String,
    store: Arc<dyn CredentialStore>,
    control_token: ControlToken,
    api_token: ControlToken,
) -> Result<()> {
    let pairing = Arc::new(PairingState::new(Duration::from_secs(300)));
    let state = api::AppState {
        store,
        pairing: pairing.clone(),
        client: chatgpt::ChatGptClient::new()?,
        default_language: language,
        api_token,
    };
    let app = api::router(state).route(
        "/internal/pairing-code",
        axum::routing::post({
            move |headers: axum::http::HeaderMap| {
                let pairing = pairing.clone();
                async move {
                    let authorized = headers
                        .get("x-control-token")
                        .and_then(|v| v.to_str().ok())
                        .is_some_and(|candidate| control_token.matches(candidate));
                    if !authorized {
                        return (axum::http::StatusCode::UNAUTHORIZED, "unauthorized")
                            .into_response();
                    }
                    axum::Json(
                        serde_json::json!({"pairing_code": pairing.issue(), "expires_in": 300}),
                    )
                    .into_response()
                }
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind(listen)
        .await
        .with_context(|| format!("bind {listen}"))?;
    tracing::info!(%listen, "daemon ready");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown())
        .await?;
    Ok(())
}

async fn issue_via_running_daemon(
    listen: std::net::SocketAddr,
    control_token: &ControlToken,
) -> Result<String> {
    let response: serde_json::Value = reqwest::Client::new()
        .post(format!("http://{listen}/internal/pairing-code"))
        .header("x-control-token", control_token.as_str())
        .send()
        .await
        .context("daemon unavailable; start it first")?
        .error_for_status()?
        .json()
        .await?;
    response["pairing_code"]
        .as_str()
        .map(ToOwned::to_owned)
        .context("invalid daemon response")
}

async fn shutdown() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        let mut signal = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler");
        signal.recv().await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! { _ = ctrl_c => {}, _ = terminate => {} }
}

use axum::response::IntoResponse;
