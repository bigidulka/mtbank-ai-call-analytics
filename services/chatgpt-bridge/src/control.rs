use std::{
    fs,
    io::{Read, Write},
    path::Path,
};

use anyhow::{Context, Result, bail};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use rand::RngCore;
use subtle::ConstantTimeEq;
use tempfile::Builder;
use zeroize::{Zeroize, ZeroizeOnDrop, Zeroizing};

const TOKEN_BYTES: usize = 32;

#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct ControlToken(String);

impl ControlToken {
    pub fn load_or_create(path: &Path) -> Result<Self> {
        if path
            .try_exists()
            .with_context(|| format!("inspect daemon token {}", path.display()))?
        {
            return Self::load_existing(path);
        }

        let parent = path.parent().context("daemon token path has no parent")?;
        fs::create_dir_all(parent).context("create config directory")?;
        if !fs::metadata(parent)?.is_dir() {
            bail!(
                "daemon token parent is not a directory: {}",
                parent.display()
            );
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(parent, fs::Permissions::from_mode(0o700))
                .context("secure config directory permissions")?;
        }

        let mut raw = [0u8; TOKEN_BYTES];
        rand::rng().fill_bytes(&mut raw);
        let value = URL_SAFE_NO_PAD.encode(raw);
        raw.zeroize();

        let mut temporary = Builder::new()
            .prefix(".daemon-token-")
            .tempfile_in(parent)
            .context("create temporary daemon token")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            temporary
                .as_file()
                .set_permissions(fs::Permissions::from_mode(0o600))
                .context("secure temporary daemon token permissions")?;
        }
        temporary
            .write_all(value.as_bytes())
            .context("write daemon token")?;
        temporary
            .as_file()
            .sync_all()
            .context("sync daemon token")?;

        match temporary.persist_noclobber(path) {
            Ok(file) => {
                file.sync_all().context("sync installed daemon token")?;
                sync_directory(parent)?;
                Self::load_existing(path)
            }
            Err(error) => {
                let tempfile::PersistError { error, file } = error;
                drop(file);
                if error.kind() == std::io::ErrorKind::AlreadyExists {
                    Self::load_existing(path)
                } else {
                    Err(error).context("install daemon token")
                }
            }
        }
    }

    fn load_existing(path: &Path) -> Result<Self> {
        let path_metadata = fs::symlink_metadata(path)
            .with_context(|| format!("inspect daemon token {}", path.display()))?;
        if !path_metadata.file_type().is_file() {
            bail!("daemon token is not a regular file: {}", path.display());
        }

        let mut file = fs::OpenOptions::new()
            .read(true)
            .open(path)
            .with_context(|| format!("open daemon token {}", path.display()))?;
        let opened_metadata = file.metadata().context("inspect opened daemon token")?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::{MetadataExt, PermissionsExt};
            if path_metadata.dev() != opened_metadata.dev()
                || path_metadata.ino() != opened_metadata.ino()
            {
                bail!("daemon token changed while opening: {}", path.display());
            }
            if opened_metadata.permissions().mode() & 0o777 != 0o600 {
                file.set_permissions(fs::Permissions::from_mode(0o600))
                    .context("secure daemon token permissions")?;
                let secured_mode = file.metadata()?.permissions().mode() & 0o777;
                if secured_mode != 0o600 {
                    bail!("daemon token permissions must be 0600: {}", path.display());
                }
            }
        }

        let mut value = String::new();
        file.read_to_string(&mut value)
            .with_context(|| format!("read daemon token {}", path.display()))?;
        validate_token(&value)
            .with_context(|| format!("validate daemon token {}", path.display()))?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    #[cfg(test)]
    pub fn from_value_for_tests(value: &str) -> Self {
        Self(value.to_owned())
    }

    pub fn matches(&self, candidate: &str) -> bool {
        self.0.as_bytes().ct_eq(candidate.as_bytes()).into()
    }
}

fn validate_token(value: &str) -> Result<()> {
    if value.len() != 43 || value.trim() != value {
        bail!("daemon token must be one canonical 256-bit URL-safe value");
    }
    let decoded = Zeroizing::new(
        URL_SAFE_NO_PAD
            .decode(value.as_bytes())
            .context("decode daemon token")?,
    );
    if decoded.len() != TOKEN_BYTES || URL_SAFE_NO_PAD.encode(decoded.as_slice()) != value {
        bail!("daemon token must be one canonical 256-bit URL-safe value");
    }
    Ok(())
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<()> {
    fs::File::open(path)
        .with_context(|| format!("open config directory {}", path.display()))?
        .sync_all()
        .context("sync config directory")
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};

    use super::*;

    #[test]
    fn concurrent_creation_converges_on_one_token() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("api-token");
        let barrier = Arc::new(Barrier::new(8));
        let mut threads = Vec::new();
        for _ in 0..8 {
            let path = path.clone();
            let barrier = barrier.clone();
            threads.push(std::thread::spawn(move || {
                barrier.wait();
                ControlToken::load_or_create(&path)
                    .unwrap()
                    .as_str()
                    .to_owned()
            }));
        }
        let values: Vec<String> = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();
        assert!(values.iter().all(|value| value == &values[0]));
        validate_token(&values[0]).unwrap();
    }

    #[test]
    fn stale_legacy_temp_file_does_not_block_creation() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("control-token");
        fs::write(path.with_extension("tmp"), "stale").unwrap();
        let token = ControlToken::load_or_create(&path).unwrap();
        validate_token(token.as_str()).unwrap();
    }

    #[test]
    fn invalid_existing_token_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("control-token");
        fs::write(&path, "not-a-token\n").unwrap();
        assert!(ControlToken::load_or_create(&path).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn existing_token_permissions_are_tightened_to_0600() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("control-token");
        let value = URL_SAFE_NO_PAD.encode([7u8; TOKEN_BYTES]);
        fs::write(&path, value).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();

        ControlToken::load_or_create(&path).unwrap();
        assert_eq!(
            fs::metadata(path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
}
