use std::{collections::BTreeMap, path::Path, process::Stdio};

use anyhow::{Context, Result, bail};
use reqwest::{Client, multipart};
use serde::Deserialize;
use tempfile::NamedTempFile;
use tokio::process::Command;

use crate::credentials::Credentials;

const TRANSCRIBE_URL: &str = "https://chatgpt.com/backend-api/transcribe";
const SESSION_URL: &str = "https://chatgpt.com/api/auth/session";

#[derive(Clone)]
pub struct ChatGptClient {
    http: Client,
}

#[derive(Debug, Deserialize)]
pub struct Transcription {
    pub text: String,
}

#[derive(Debug, Deserialize)]
struct WebSession {
    #[serde(rename = "accessToken")]
    access_token: String,
    expires: Option<String>,
}

impl ChatGptClient {
    pub fn new() -> Result<Self> {
        let http = Client::builder()
            .user_agent("Mozilla/5.0 ChatGPT-Transcribe-Connect/0.1")
            .timeout(std::time::Duration::from_secs(120))
            .build()?;
        Ok(Self { http })
    }

    pub async fn refresh_credentials(&self, credentials: &mut Credentials) -> Result<()> {
        let response = self
            .http
            .get(SESSION_URL)
            .header("Cookie", cookie_header(&credentials.cookies))
            .header("Referer", "https://chatgpt.com/")
            .send()
            .await
            .context("refresh ChatGPT session")?;
        let status = response.status();
        if !status.is_success() {
            bail!("ChatGPT session refresh failed ({status})");
        }
        let session: WebSession = response.json().await.context("parse ChatGPT session")?;
        if session.access_token.is_empty() {
            bail!("ChatGPT session refresh returned no access token");
        }
        credentials.access_token = session.access_token;
        credentials.expires_at = session.expires;
        Ok(())
    }

    pub async fn transcribe(
        &self,
        input: &Path,
        credentials: &Credentials,
        language: &str,
    ) -> Result<Transcription> {
        let (webm, duration_ms) = encode_webm(input).await?;
        let bytes = tokio::fs::read(webm.path()).await?;
        let file = multipart::Part::bytes(bytes)
            .file_name("whisper.webm")
            .mime_str("audio/webm;codecs=opus")?;
        let form = multipart::Form::new()
            .part("file", file)
            .text("duration_ms", duration_ms.to_string());

        let mut request = self
            .http
            .post(TRANSCRIBE_URL)
            .bearer_auth(&credentials.access_token)
            .header("OAI-Language", language)
            .header("X-OpenAI-Target-Path", "/backend-api/transcribe")
            .header("X-OpenAI-Target-Route", "/backend-api/transcribe")
            .header("Origin", "https://chatgpt.com")
            .header("Referer", "https://chatgpt.com/")
            .multipart(form);
        if !credentials.cookies.is_empty() {
            request = request.header("Cookie", cookie_header(&credentials.cookies));
        }
        let response = request
            .send()
            .await
            .context("send ChatGPT transcription request")?;

        let status = response.status();
        let body = response.bytes().await?;
        if !status.is_success() {
            let message = String::from_utf8_lossy(&body);
            bail!(
                "ChatGPT transcription failed ({status}): {}",
                truncate(&message, 300)
            );
        }
        serde_json::from_slice(&body).context("parse ChatGPT transcription response")
    }
}

fn cookie_header(cookies: &BTreeMap<String, String>) -> String {
    cookies
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join("; ")
}

fn truncate(value: &str, max: usize) -> String {
    value.chars().take(max).collect()
}

async fn encode_webm(input: &Path) -> Result<(NamedTempFile, u64)> {
    let output = NamedTempFile::with_suffix(".webm")?;
    let status = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(input)
        .args([
            "-c:a",
            "libopus",
            "-b:a",
            "32k",
            "-ac",
            "1",
            "-ar",
            "48000",
            "-application",
            "voip",
        ])
        .arg(output.path())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .status()
        .await
        .context("run ffmpeg (is it installed?)")?;
    if !status.success() {
        bail!("ffmpeg audio conversion failed");
    }

    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
        ])
        .arg(output.path())
        .output()
        .await
        .context("run ffprobe")?;
    if !probe.status.success() {
        bail!("ffprobe duration detection failed");
    }
    let seconds: f64 = String::from_utf8(probe.stdout)?.trim().parse()?;
    Ok((output, (seconds * 1000.0).round() as u64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cookies_are_deterministic() {
        let cookies = BTreeMap::from([("b".into(), "2".into()), ("a".into(), "1".into())]);
        assert_eq!(cookie_header(&cookies), "a=1; b=2");
    }
}
