use std::{env, path::PathBuf, process::Command};

#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::fs;

use anyhow::{Context, Result, bail};

#[allow(clippy::needless_return)]
pub fn install() -> Result<PathBuf> {
    let exe = env::current_exe()?.canonicalize()?;
    #[cfg(target_os = "linux")]
    {
        let dir = dirs::config_dir()
            .context("no config directory")?
            .join("systemd/user");
        fs::create_dir_all(&dir)?;
        let path = dir.join("chatgpt-transcribe-connect.service");
        fs::write(
            &path,
            render_linux_unit(&exe, &crate::config::config_dir()?),
        )?;
        run("systemctl", &["--user", "daemon-reload"])?;
        run(
            "systemctl",
            &["--user", "enable", "chatgpt-transcribe-connect.service"],
        )?;
        run(
            "systemctl",
            &["--user", "restart", "chatgpt-transcribe-connect.service"],
        )?;
        return Ok(path);
    }
    #[cfg(target_os = "macos")]
    {
        let dir = dirs::home_dir()
            .context("no home directory")?
            .join("Library/LaunchAgents");
        fs::create_dir_all(&dir)?;
        let path = dir.join("dev.chatgpt-transcribe-connect.plist");
        let log = dirs::home_dir()
            .context("no home directory")?
            .join("Library/Logs/chatgpt-transcribe-connect.log");
        fs::write(
            &path,
            format!(
                r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict><key>Label</key><string>dev.chatgpt-transcribe-connect</string><key>ProgramArguments</key><array><string>{}</string><string>serve</string></array><key>RunAtLoad</key><true/><key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict><key>StandardOutPath</key><string>{}</string><key>StandardErrorPath</key><string>{}</string></dict></plist>"#,
                exe.display(),
                log.display(),
                log.display()
            ),
        )?;
        run("launchctl", &["load", "-w", path.to_str().unwrap()])?;
        return Ok(path);
    }
    #[cfg(target_os = "windows")]
    {
        let status = Command::new("schtasks")
            .args([
                "/Create",
                "/F",
                "/SC",
                "ONLOGON",
                "/TN",
                "ChatGPT Transcribe Connect",
                "/TR",
                &format!("\"{}\" serve", exe.display()),
            ])
            .status()?;
        if !status.success() {
            bail!("schtasks failed");
        }
        return Ok(exe);
    }
}

#[cfg(target_os = "linux")]
fn render_linux_unit(exe: &std::path::Path, config_dir: &std::path::Path) -> String {
    format!(
        "[Unit]\nDescription=ChatGPT Transcribe Connect\nAfter=network-online.target\n\n[Service]\nExecStart={} serve\nRestart=on-failure\nRestartSec=5\nNice=5\nIOSchedulingClass=idle\nNoNewPrivileges=true\nPrivateTmp=true\nProtectSystem=strict\nProtectHome=read-only\nReadWritePaths={}\nRestrictAddressFamilies=AF_UNIX AF_INET AF_INET6\nLockPersonality=true\nUMask=0077\n\n[Install]\nWantedBy=default.target\n",
        systemd_quote(exe),
        systemd_quote(config_dir),
    )
}

#[cfg(target_os = "linux")]
fn systemd_quote(path: &std::path::Path) -> String {
    let escaped = path
        .to_string_lossy()
        .replace('%', "%%")
        .replace('\\', "\\\\")
        .replace('"', "\\\"");
    format!("\"{escaped}\"")
}

pub fn uninstall() -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        let _ = run(
            "systemctl",
            &[
                "--user",
                "disable",
                "--now",
                "chatgpt-transcribe-connect.service",
            ],
        );
        let path = dirs::config_dir()
            .context("no config directory")?
            .join("systemd/user/chatgpt-transcribe-connect.service");
        if path.exists() {
            fs::remove_file(path)?;
        }
        run("systemctl", &["--user", "daemon-reload"])?;
    }
    #[cfg(target_os = "macos")]
    {
        let path = dirs::home_dir()
            .context("no home directory")?
            .join("Library/LaunchAgents/dev.chatgpt-transcribe-connect.plist");
        if path.exists() {
            let _ = run("launchctl", &["bootout", path.to_str().unwrap()]);
            fs::remove_file(path)?;
        }
    }
    #[cfg(target_os = "windows")]
    {
        let _ = run(
            "schtasks",
            &["/Delete", "/F", "/TN", "ChatGPT Transcribe Connect"],
        );
    }
    Ok(())
}

fn run(program: &str, args: &[&str]) -> Result<()> {
    let status = Command::new(program)
        .args(args)
        .status()
        .with_context(|| format!("run {program}"))?;
    if !status.success() {
        bail!("{program} failed with {status}");
    }
    Ok(())
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn linux_unit_quotes_resolved_paths_and_hardens_service() {
        let unit = render_linux_unit(
            Path::new("/opt/My App/bin/chatgpt%connect"),
            Path::new("/tmp/custom config/chatgpt-transcribe-connect"),
        );
        assert!(unit.contains("ExecStart=\"/opt/My App/bin/chatgpt%%connect\" serve"));
        assert!(unit.contains("ReadWritePaths=\"/tmp/custom config/chatgpt-transcribe-connect\""));
        assert!(unit.contains("ProtectSystem=strict"));
        assert!(unit.contains("UMask=0077"));
    }
}
