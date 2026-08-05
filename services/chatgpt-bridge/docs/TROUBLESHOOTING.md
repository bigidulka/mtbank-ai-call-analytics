# Troubleshooting

## Extension says daemon unavailable

```bash
chatgpt-transcribe-connect serve
curl http://127.0.0.1:37182/health
```

Check another process is not using port 37182. Reload extension after daemon starts.

## Session cookie not visible

1. Sign in at `https://chatgpt.com` in same Brave/Chrome profile that loaded extension.
2. Reload ChatGPT tab and extension.
3. In Brave, check Shields/cookie settings for ChatGPT.
4. Extension supports legacy/new and chunked `next-auth`/`authjs` cookie names. Error lists visible cookie names but never values.

## Pairing code rejected

Code expires after five minutes and every attempt consumes it. Generate a new code and paste once:

```bash
chatgpt-transcribe-connect pair
```

## 401/403 or upstream error

Session expired or Cloudflare rejected it. Keep ChatGPT signed in, run `logout`, then pair again. Do not post credentials in an issue.

## FFmpeg error

Verify both executables:

```bash
ffmpeg -version
ffprobe -version
```

Ensure service process inherits a PATH containing them. On Windows restart session after `winget install Gyan.FFmpeg`.

## Background service

Linux:

```bash
systemctl --user status chatgpt-transcribe-connect.service
journalctl --user -u chatgpt-transcribe-connect.service -n 100
```

macOS:

```bash
launchctl list | grep chatgpt-transcribe
cat ~/Library/Logs/chatgpt-transcribe-connect.log
```

Windows:

```powershell
schtasks /Query /TN "ChatGPT Transcribe Connect" /V /FO LIST
```

## Reset

```bash
chatgpt-transcribe-connect logout
chatgpt-transcribe-connect uninstall-service
```
