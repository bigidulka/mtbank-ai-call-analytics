use std::{net::SocketAddr, path::PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(author, version, about)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Loopback, unspecified, or private/ULA address to listen on. Never a public address.
    #[arg(long, default_value = "127.0.0.1:37182", global = true)]
    pub listen: SocketAddr,

    /// Default BCP-47 language sent to ChatGPT.
    #[arg(long, default_value = "en-US", global = true)]
    pub language: String,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run local transcription daemon.
    Serve,
    /// Generate one-time browser-extension pairing code.
    Pair,
    /// Show connection status without revealing credentials.
    Status,
    /// Print resolved transcription API-token path without revealing its value.
    ApiTokenPath,
    /// Delete stored ChatGPT credentials.
    Logout,
    /// Install per-user background service.
    InstallService,
    /// Remove per-user background service.
    UninstallService,
}

pub fn config_dir() -> Result<PathBuf> {
    let base = dirs::config_dir().context("cannot determine config directory")?;
    Ok(base.join("chatgpt-transcribe-connect"))
}
