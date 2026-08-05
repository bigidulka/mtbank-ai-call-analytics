use std::{collections::BTreeMap, fs, io::Write, path::PathBuf};

use anyhow::{Context, Result, bail};
use base64::{Engine, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use zeroize::Zeroize;

const SERVICE: &str = "chatgpt-transcribe-connect";
const ACCOUNT: &str = "chatgpt-web-session";

#[derive(Clone, Serialize, Deserialize)]
pub struct Credentials {
    pub access_token: String,
    pub cookies: BTreeMap<String, String>,
    pub expires_at: Option<String>,
}

impl Drop for Credentials {
    fn drop(&mut self) {
        self.access_token.zeroize();
        for value in self.cookies.values_mut() {
            value.zeroize();
        }
        self.cookies.clear();
    }
}

impl Credentials {
    pub fn validate(&self) -> Result<()> {
        if self.access_token.trim().is_empty() {
            bail!("access_token is required");
        }
        const SESSION_PREFIXES: [&str; 4] = [
            "__Secure-next-auth.session-token",
            "__Secure-authjs.session-token",
            "next-auth.session-token",
            "authjs.session-token",
        ];
        // Browser sessions can expose a valid short-lived access token while
        // withholding HttpOnly cookies. Accept this transient mode: requests
        // still need an explicit one-use pairing code, but credentials cannot
        // be refreshed after the token expires without pairing again.
        if !self.cookies.is_empty()
            && !self.cookies.keys().any(|name| {
                SESSION_PREFIXES.iter().any(|prefix| {
                    name == prefix
                        || name
                            .strip_prefix(prefix)
                            .is_some_and(|suffix| suffix.starts_with('.'))
                })
            })
        {
            bail!("cookies must include a ChatGPT session cookie when provided");
        }
        if self.access_token.len() > 32_768
            || self.cookies.len() > 128
            || self
                .cookies
                .iter()
                .any(|(k, v)| k.len() > 256 || v.len() > 32_768)
        {
            bail!("credential exceeds size limit");
        }
        Ok(())
    }
}

pub trait CredentialStore: Send + Sync {
    fn load(&self) -> Result<Option<Credentials>>;
    fn save(&self, credentials: &Credentials) -> Result<()>;
    fn delete(&self) -> Result<()>;
}

pub struct OsCredentialStore {
    fallback: PathBuf,
}

impl OsCredentialStore {
    pub fn new(fallback: PathBuf) -> Self {
        Self { fallback }
    }

    fn entry(&self) -> Result<keyring::Entry> {
        keyring::Entry::new(SERVICE, ACCOUNT).context("create OS keyring entry")
    }

    fn load_fallback(&self) -> Result<Option<Credentials>> {
        if !self.fallback.exists() {
            return Ok(None);
        }
        let encoded = fs::read_to_string(&self.fallback).context("read credential fallback")?;
        let bytes = STANDARD
            .decode(encoded.trim())
            .context("decode credential fallback")?;
        Ok(Some(
            serde_json::from_slice(&bytes).context("parse credential fallback")?,
        ))
    }

    fn save_fallback(&self, credentials: &Credentials) -> Result<()> {
        if let Some(parent) = self.fallback.parent() {
            fs::create_dir_all(parent)?;
        }
        let body = STANDARD.encode(serde_json::to_vec(credentials)?);
        let tmp = self.fallback.with_extension("tmp");
        let mut options = fs::OpenOptions::new();
        options.create(true).truncate(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&tmp).context("create credential fallback")?;
        file.write_all(body.as_bytes())?;
        file.sync_all()?;
        fs::rename(tmp, &self.fallback)?;
        Ok(())
    }
}

impl CredentialStore for OsCredentialStore {
    fn load(&self) -> Result<Option<Credentials>> {
        match self
            .entry()
            .and_then(|entry| entry.get_password().context("read OS keyring"))
        {
            Ok(value) => Ok(Some(
                serde_json::from_str(&value).context("parse keyring credentials")?,
            )),
            Err(_) => self.load_fallback(),
        }
    }

    fn save(&self, credentials: &Credentials) -> Result<()> {
        credentials.validate()?;
        let value = serde_json::to_string(credentials)?;
        match self
            .entry()
            .and_then(|entry| entry.set_password(&value).context("write OS keyring"))
        {
            Ok(()) => Ok(()),
            Err(error) => {
                tracing::warn!(%error, "OS keyring unavailable; using permission-restricted fallback");
                self.save_fallback(credentials)
            }
        }
    }

    fn delete(&self) -> Result<()> {
        if let Ok(entry) = self.entry() {
            let _ = entry.delete_credential();
        }
        if self.fallback.exists() {
            fs::remove_file(&self.fallback)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn credentials(cookie_name: &str) -> Credentials {
        Credentials {
            access_token: "access".into(),
            cookies: BTreeMap::from([(cookie_name.into(), "cookie".into())]),
            expires_at: None,
        }
    }

    #[test]
    fn accepts_legacy_authjs_and_chunked_session_cookies() {
        for name in [
            "__Secure-next-auth.session-token",
            "__Secure-next-auth.session-token.0",
            "__Secure-authjs.session-token.1",
            "authjs.session-token",
        ] {
            assert!(credentials(name).validate().is_ok(), "rejected {name}");
        }
    }

    #[test]
    fn rejects_unrelated_cookies() {
        assert!(credentials("cf_clearance").validate().is_err());
    }

    #[test]
    fn accepts_access_token_without_cookies_for_transient_browser_pairing() {
        let credentials = Credentials {
            access_token: "access".into(),
            cookies: BTreeMap::new(),
            expires_at: None,
        };
        assert!(credentials.validate().is_ok());
    }
}
