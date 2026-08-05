# Architecture

```text
Chromium extension                     Local daemon
chatgpt.com cookies ── one-time pair ─▶ 127.0.0.1:37182
                                              │
OpenAI-compatible client ─ Bearer + audio ──▶│
                                              ├─ FFmpeg → WebM/Opus
                                              └─ HTTPS → chatgpt.com/backend-api/transcribe
```

## Pairing

CLI authenticates to running daemon with a local 256-bit control token, then asks it to issue a 32-character URL-safe random code. Daemon stores only SHA-256 digest and expiration. Extension explicitly reads ChatGPT cookies and `/api/auth/session`, then POSTs them with code. Any attempt consumes code, preventing guessing retries. CORS and handler reject non-extension origins.

## Transcription

Daemon requires its per-user transcription bearer token, then accepts OpenAI multipart fields. Audio is bounded at 25 MiB, written to an OS temporary file, converted to mono 48 kHz WebM/Opus, and sent with `duration_ms`, ChatGPT bearer token, cookie header, and language hint. Before each request, current access token is refreshed from ChatGPT session cookies. Backend JSON `text` becomes OpenAI-compatible response.

## Trust boundaries

- Browser extension has cookie permission only for `chatgpt.com` and host access to daemon; pairing accepts only `chrome-extension://` origins.
- Daemon binds loopback and requires a random `api-token` for transcription. Pairing-code issuance uses a separate `control-token`, so transcription clients cannot enter the pairing control plane.
- Credentials persist via OS keyring where available.
- ChatGPT endpoint is external and undocumented.

## Components

- `src/api.rs`: HTTP contract, limits, CORS, pairing handler
- `src/chatgpt.rs`: audio conversion and backend client
- `src/credentials.rs`: validation and keyring/fallback storage
- `src/pairing.rs`: one-use secret lifecycle
- `src/service.rs`: per-OS background startup
- `extension/`: Manifest V3 connector
