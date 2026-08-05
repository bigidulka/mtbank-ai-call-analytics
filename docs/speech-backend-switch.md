# Speech backend switch

Production default remains the validated RunPod `faster-whisper large-v3-turbo + pyannote Community-1` profile. The custom profile is an explicit experimental no-GPU fallback:

```text
OpenAI `gpt-4o-transcribe` flat ASR
  -> typed Sol semantic turn reconstruction
  -> FFmpeg silencedetect VAD anchors
  -> ranked longest-gap alignment
  -> adaptive bounded padding (0.2/0.9 -> 0.1/0.2 -> 0/0)
  -> canonical /v1/transcribe response
```

This profile is not acoustical diarization and must not inherit RunPod or synthetic benchmark quality claims. It fails closed when semantic turns cannot fit VAD topology.

## Configuration

The existing application gateway variables provide semantic-role calls. Custom speech additionally requires `OPENAI_TRANSCRIPTION_API_KEY` in deployment `.env`; never put value in Git or command output. Optional allowlisted endpoint/model variables:

```dotenv
OPENAI_TRANSCRIPTION_ENDPOINT=https://api.openai.com/v1/audio/transcriptions
OPENAI_TRANSCRIPTION_MODEL=gpt-4o-transcribe
```

Optional model override:

```dotenv
MTBANK_CUSTOM_SPEECH__ROLE_MODEL=gpt-5.6-sol
```

## Switch

From the deployment checkout:

```bash
deploy/speech-backend status
deploy/speech-backend custom
deploy/speech-backend runpod
```

`custom` builds and starts `custom-speech`, waits for readiness, recreates API with `http://custom-speech:8010`, then verifies the active API environment. `runpod` recreates API with the existing HTTPS RunPod URL and bearer key, verifies it, then stops custom speech. It never starts or changes the RunPod pod itself.

RunPod must already be running before switching back. Both directions remain fail-closed: no implicit fallback occurs inside a request.

## Runtime verification

```bash
curl -fsS http://127.0.0.1:3000/health/ready
docker compose -f docker-compose.yml -f docker-compose.custom-speech.yml exec -T custom-speech \
  python -c "from urllib.request import urlopen; print(urlopen('http://127.0.0.1:8010/v1/runtime').read().decode())"
```

Expected custom attestation includes:

```json
{"runtime":{"profile":"experimental_no_gpu","device":"cpu","asr":{"model_id":"gpt-4o-transcribe"}}}
```

After switching, validate one approved synthetic/no-PII attachment through OpenWebUI. Do not use real customer calls for this experimental path without separate approval.

## Even more experimental: unofficial ChatGPT Web bridge

`docker-compose.chatgpt-bridge.yml` swaps the official OpenAI ASR call for `services/chatgpt-bridge`, a Rust daemon (vendored from a personal `chatgpt-transcribe-connect` project) that drives an unofficial ChatGPT Web transcription endpoint through a paired browser-extension session on a real ChatGPT account.

> **This is materially riskier than the official-API custom profile.** It moves a personal ChatGPT session credential (cookies + access token) onto shared infrastructure, is not affiliated with or supported by OpenAI, and depends on an undocumented endpoint that can change or stop working without notice. Prefer the official OpenAI API path above unless this is explicitly required. Only proceed with informed authorization from whoever owns the deployment target.

Upstream `chatgpt-transcribe-connect` refuses to bind any non-loopback address by design (`SECURITY.md`: "Daemon refuses non-loopback listen addresses"). This fork relaxes that single check (`services/chatgpt-bridge/src/main.rs::is_bind_address_allowed`) to additionally allow unspecified (`0.0.0.0`) and private/ULA binds, so the daemon can run inside an isolated Docker network reachable by sibling containers over Docker DNS — genuinely public/global addresses are still rejected in code as defense-in-depth. Every other protection is unmodified and unaffected by this: bearer-token auth on `/v1/audio/transcriptions`, a one-time 192-bit pairing code that expires in five minutes, and CORS-enforced `chrome-extension://` origin checks on `/internal/pair`.

The real exposure boundary is therefore entirely in `docker-compose.chatgpt-bridge.yml`: the daemon is published only to `127.0.0.1:37182` on the host (never `0.0.0.0`), so it is unreachable from outside the host itself; other containers reach it only via the internal Docker network as `chatgpt-bridge:37182`.

### Pairing cannot be automated by an assistant

Credential submission always originates from the operator's own machine, not from any script this repository runs unattended and not from any assistant session. Two supported paths exist depending on whether a session is already paired locally.

### Option A — migrate an already-paired local session (no browser, no tunnel)

If `chatgpt-transcribe-connect` is already paired on your own machine (`chatgpt-transcribe-connect status` reports `Connected` there), reuse that session instead of pairing again:

1. Start only the bridge on the server:

   ```bash
   deploy/speech-backend bridge-up
   ```

2. From your own machine, run:

   ```bash
   services/chatgpt-bridge/scripts/migrate-local-session.sh <ssh-host> <deploy-path>
   ```

   This issues a one-time pairing code on the server, then pipes your local session credential from the OS keyring (`secret-tool lookup service chatgpt-transcribe-connect username chatgpt-web-session`) directly over SSH into the bridge container's own `python3`, which POSTs it to `/internal/pair` — the identical endpoint and payload shape the browser extension itself would send. The credential is never written to a file on either machine, never passed as a command-line argument, and never captured by this script's own output; it also never passes through this repository's scripts, `deploy/speech-backend`, or any assistant session.

3. Verify, then switch:

   ```bash
   deploy/speech-backend bridge-status
   deploy/speech-backend custom-bridge
   ```

The bridge's `MTBANK_CUSTOM_SPEECH__ASR_API_KEY` still needs `CHATGPT_BRIDGE_API_TOKEN` set in `.env` — see step 7 under Option B; that token identifies callers to this daemon and is unrelated to the ChatGPT session credential migrated above.

### Option B — fresh pairing through a browser

Use this when no local session already exists, or the local one has expired.

The browser extension (`services/chatgpt-bridge/extension/`) has `const DAEMON = "http://127.0.0.1:37182"` hardcoded, so pairing needs a real browser that can reach that exact address. On a headless server that means tunnelling the port to your own machine:

1. Start only the bridge:

   ```bash
   deploy/speech-backend bridge-up
   ```

2. From a second local terminal, tunnel the daemon's loopback port to your machine (replace `<host>` with the deploy target):

   ```bash
   ssh -N -L 37182:127.0.0.1:37182 <host>
   ```

   Keep this running only for the duration of pairing.

3. In `chrome://extensions` (or the Brave/Edge equivalent), enable Developer mode, then **Load unpacked** and select `services/chatgpt-bridge/extension/`. Sign in at `https://chatgpt.com` in that same browser.

4. Back on the deploy host, issue a one-time pairing code:

   ```bash
   deploy/speech-backend bridge-pair
   ```

5. Open the extension popup, paste the code, click **Connect ChatGPT**. The tunnel carries the credential submission directly to the server-side daemon; it never passes through this repository, its scripts, or any assistant session.

6. Confirm pairing succeeded, then close the SSH tunnel:

   ```bash
   deploy/speech-backend bridge-status
   ```

7. Retrieve the token *path* (never its value) and read the file yourself on the host to copy it into `.env`:

   ```bash
   deploy/speech-backend bridge-token-path
   ```

   Set the printed value as `CHATGPT_BRIDGE_API_TOKEN` in the deployment `.env`. Do this by editing `.env` directly on the host; never paste the token into a chat session, issue, or log.

8. Switch:

   ```bash
   deploy/speech-backend custom-bridge
   ```

### Ongoing operation

- `deploy/speech-backend bridge-status` reports `connected`/`not connected` without revealing credentials.
- `deploy/speech-backend runpod` stops both `custom-speech` and `chatgpt-bridge` when rolling back.
- The account behind the bridge is subject to ChatGPT's own usage limits; a `usage_limit_reached` response surfaces as an upstream ASR failure, not a bridge crash.
- Re-run Option A or B whenever the session expires (`bridge-status` reports `not connected`) or after `chatgpt-transcribe-connect logout` inside the container.
- Credentials persist in the `chatgpt-bridge-config` named volume, protected at `0600`; the daemon prefers the Linux Secret Service when available and otherwise falls back automatically to a `0600` Base64-encoded file, which is access control, not encryption. Treat the host's disk and backups accordingly.
