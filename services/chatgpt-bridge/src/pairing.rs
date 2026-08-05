use std::{
    sync::Mutex,
    time::{Duration, Instant},
};

use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use rand::RngCore;
use subtle::ConstantTimeEq;

pub struct PairingState {
    inner: Mutex<Option<PairingSecret>>,
    ttl: Duration,
}

struct PairingSecret {
    digest: [u8; 32],
    expires: Instant,
}

impl PairingState {
    pub fn new(ttl: Duration) -> Self {
        Self {
            inner: Mutex::new(None),
            ttl,
        }
    }

    pub fn issue(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut raw = [0u8; 24];
        rand::rng().fill_bytes(&mut raw);
        let code = URL_SAFE_NO_PAD.encode(raw);
        let digest: [u8; 32] = Sha256::digest(code.as_bytes()).into();
        *self.inner.lock().expect("pair mutex poisoned") = Some(PairingSecret {
            digest,
            expires: Instant::now() + self.ttl,
        });
        code
    }

    pub fn consume(&self, candidate: &str) -> bool {
        use sha2::{Digest, Sha256};
        let mut guard = self.inner.lock().expect("pair mutex poisoned");
        let Some(secret) = guard.take() else {
            return false;
        };
        if Instant::now() > secret.expires {
            return false;
        }
        let digest: [u8; 32] = Sha256::digest(candidate.as_bytes()).into();
        bool::from(secret.digest.ct_eq(&digest))
    }

    pub fn is_active(&self) -> bool {
        self.inner
            .lock()
            .expect("pair mutex poisoned")
            .as_ref()
            .is_some_and(|s| Instant::now() <= s.expires)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_is_single_use() {
        let state = PairingState::new(Duration::from_secs(30));
        let code = state.issue();
        assert!(state.consume(&code));
        assert!(!state.consume(&code));
    }

    #[test]
    fn wrong_code_consumes_attempt() {
        let state = PairingState::new(Duration::from_secs(30));
        let code = state.issue();
        assert!(!state.consume("wrong"));
        assert!(!state.consume(&code));
    }
}
