# Security policy

## Sensitive data

ChatGPT session cookies and access tokens grant account access. Never paste them into issues, logs, screenshots, shell history, or support messages. If exposed, sign out all ChatGPT sessions and sign in again.

## Design boundaries

- Daemon refuses non-loopback listen addresses.
- Pairing codes contain 192 random bits, are stored only as SHA-256 digests, expire after five minutes, and are consumed on first attempt.
- Issuing a pairing code requires a separate 256-bit daemon control token stored with user-only permissions; arbitrary local web pages cannot create a code.
- Pairing endpoint accepts only `chrome-extension://` origins.
- Extension sends credentials only to `http://127.0.0.1:37182`.
- Credentials are never intentionally logged or returned by status APIs.
- Uploads are limited to 25 MiB and temporary audio files are deleted after requests.

Local malware or another process running as your OS user may access loopback services or user credentials. This project is not a sandbox against a compromised machine. Developer-mode extensions also carry supply-chain risk: inspect source before loading.

## Credential storage

Preferred storage:

- macOS Keychain
- Windows Credential Manager
- Linux Secret Service

When OS keyring is unavailable, daemon currently falls back to a Base64-encoded JSON file protected by filesystem mode `0600` on Unix. Base64 is **not encryption**. This fallback protects against other non-privileged users, not root, malware, backups, or disk access. Headless systems should use an encrypted home directory and strict permissions.

## Network risk

Transcription audio and credentials are sent to `https://chatgpt.com`. No project-operated server exists. ChatGPT's undocumented endpoint and Cloudflare controls may change. Review OpenAI terms and privacy policy before use.

## Reporting vulnerabilities

Do not open public issues containing exploit details or credentials. Use GitHub's **Report a vulnerability** private security advisory for this repository. Include affected version, reproduction, impact, and suggested mitigation. Revoke any credential included accidentally.
