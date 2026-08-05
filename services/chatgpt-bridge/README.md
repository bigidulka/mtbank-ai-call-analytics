# ChatGPT Transcribe Connect

Local, OpenAI-compatible speech-to-text gateway backed by ChatGPT Web dictation. Connect a Plus/Pro browser session with one click, then use existing OpenAI-compatible applications against `http://127.0.0.1:37182/v1/audio/transcriptions`.

> [!WARNING]
> Unofficial project, not affiliated with or endorsed by OpenAI. It uses an undocumented ChatGPT Web endpoint that can change or stop working. A session cookie grants account access: install only trusted builds and review [SECURITY.md](SECURITY.md). For stable production use, prefer the official OpenAI API.

## Features

- OpenAI-compatible multipart transcription endpoint
- Explicit browser-extension pairing; no CDP or browser automation during transcription
- One-use, 192-bit pairing code expiring after five minutes
- Loopback-only daemon with separate per-user control and transcription bearer tokens
- OS credential store, with permission-restricted fallback when unavailable
- WAV/MP3/M4A/etc. conversion to WebM/Opus through FFmpeg
- Linux, macOS Intel/Apple Silicon, and Windows release binaries
- User-level background startup with automatic restart

## Requirements

- Active ChatGPT Web account with dictation access
- Chromium browser (Chrome, Brave, Edge, Chromium)
- `ffmpeg` and `ffprobe` on `PATH`

Install FFmpeg:

```bash
# Ubuntu/Debian
sudo apt install ffmpeg
# Arch Linux
sudo pacman -S ffmpeg
# macOS
brew install ffmpeg
# Windows
winget install Gyan.FFmpeg
```

## Quick start

1. Download binary and extension ZIP from [Releases](https://github.com/bigidulka/chatgpt-transcribe-connect/releases).
2. Put binary somewhere permanent and run:

   ```bash
   chatgpt-transcribe-connect install-service
   ```

3. Open browser extension page (`chrome://extensions`, `brave://extensions`, or `edge://extensions`), enable Developer mode, unzip extension and choose **Load unpacked**.
4. Sign in at [chatgpt.com](https://chatgpt.com).
5. Generate code:

   ```bash
   chatgpt-transcribe-connect pair
   ```

6. Open extension, paste code, press **Connect ChatGPT**.
7. Verify:

   ```bash
   chatgpt-transcribe-connect status
   curl http://127.0.0.1:37182/health
   ```

## Transcribe

```bash
API_TOKEN_PATH="$(chatgpt-transcribe-connect api-token-path)"
curl http://127.0.0.1:37182/v1/audio/transcriptions \
  -H "Authorization: Bearer $(cat "$API_TOKEN_PATH")" \
  -F model=gpt-4o-transcribe \
  -F language=ru-RU \
  -F response_format=json \
  -F file=@speech.wav
```

```json
{"text":"Recognized text"}
```

`model` is accepted for client compatibility but ChatGPT Web chooses its internal model. Supported response formats: `json` and `text`. Maximum upload: 25 MiB. The daemon creates `api-token` and `control-token` in its config directory with mode `0600`; transcription clients receive only `api-token`.

### OpenAI SDK

```python
import subprocess
from pathlib import Path
from openai import OpenAI

api_token_path = subprocess.check_output(
    ["chatgpt-transcribe-connect", "api-token-path"], text=True
).strip()
api_token = Path(api_token_path).read_text().strip()
client = OpenAI(base_url="http://127.0.0.1:37182/v1", api_key=api_token)
with open("speech.wav", "rb") as audio:
    result = client.audio.transcriptions.create(model="gpt-4o-transcribe", file=audio)
print(result.text)
```

## CLI

```text
serve               Run daemon (default command)
pair                Generate one-time pairing code through running daemon
status              Show connection status without credentials
api-token-path      Print resolved transcription-token path, never its value
logout              Delete credentials
install-service     Install/start per-user background service
uninstall-service   Stop/remove background service
```

Options:

```bash
chatgpt-transcribe-connect --listen 127.0.0.1:37182 --language en-US serve
```

Non-loopback addresses are rejected.

## Background services

- Linux: systemd user unit, `Restart=on-failure`
- macOS: LaunchAgent with `RunAtLoad` and `KeepAlive`
- Windows: per-user Task Scheduler task on login

Logs:

```bash
journalctl --user -u chatgpt-transcribe-connect.service -f  # Linux
```

## Credential lifecycle

Extension sends session cookies and current access token directly to loopback daemon. Daemon stores them in macOS Keychain, Windows Credential Manager, or Linux Secret Service. If OS keyring is unavailable, fallback file is created with user-only permissions. Before each transcription, daemon refreshes short-lived access token through ChatGPT session endpoint. Run `logout` to delete credentials; re-pair when session cookies expire.

## Build

```bash
cargo build --release --locked
cargo test --all-features
```

See [architecture](docs/ARCHITECTURE.md), [troubleshooting](docs/TROUBLESHOOTING.md), and [security policy](SECURITY.md).

## License

MIT
