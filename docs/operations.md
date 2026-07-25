# Operations

## Runtime

Canonical batch speech processing is local `faster-whisper` `large-v3-turbo` plus local offline `pyannote/speaker-diarization-community-1`. Runtime verifies both local artifacts from `models/manifest.json` and never downloads models.

Default Compose keeps streaming disabled and does not require `GROQ_API_KEY`. The opt-in WebSocket provisional mode additionally requires Groq credentials; its remote RunPod transport uses direct `wss`, exactly one bearer header, no proxy/compression, and rejects handshake redirects. Groq produces bounded provisional updates only; local ASR and Community-1 remain canonical reconciliation. Set one explicit browser origin and apply the overlay:

```bash
MTBANK_WEBSOCKET_ALLOWED_ORIGIN=https://approved.example \
  docker compose -f docker-compose.yml -f docker-compose.websocket.yml up --build --wait
```

GPU profile requires an NVIDIA-capable host:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile gpu up --build --wait
```

A successful Compose render or CPU diagnostic is not GPU performance evidence.

## Clean-clone setup

Default Compose needs Docker Compose, FFmpeg in the speech image, at least 16 GB RAM,
about 12 GB free disk for images and local model artifacts, and an OpenAI-compatible
LLM endpoint. CPU mode is functional but does not meet the five-minute demo SLA; use an
NVIDIA GPU with the GPU overlay for that target.

```bash
cp .env.example .env
# Set every blank secret, an approved OpenAI-compatible gateway, model ID, and current code SHA.
# Example: MTBANK_WORKFLOW__CODE_SHA=$(git rev-parse HEAD)

# Requires an accepted Hugging Face token only for the gated Community-1 artifact.
HF_TOKEN=... uv run python scripts/provision_speech_models.py \
  --artifact-root models/artifacts \
  --output-manifest models/manifest.json \
  --cache-dir .cache/mtbank-speech-models

docker compose up --build --wait
```

The provisioning command downloads both reviewed artifacts, verifies them, and writes
`models/manifest.json`. `models/artifacts/` and provisioning cache are intentionally
Git-ignored; copy neither secrets nor model weights into Git. Runtime runs offline and
fails readiness when the manifest or artifact hashes do not match.

## Controlled Cloudflare rollback

For a tunnel created by the controlled provisioner, remove the owned DNS record first, stop the `cloudflared` Compose service, and optionally delete the dedicated tunnel last.
Use only the recorded tunnel and DNS names; do not alter unrelated zones or tunnels.
The provisioner reads `CF_EMAIL` and `CF_GLOBAL_API_KEY` from its protected environment;
never place them in Compose files or command history. Actual external side effects were not run by offline validation.

## Offline validation

```bash
docker compose --env-file tmp/release-ci.env config --quiet
docker compose --env-file tmp/release-ci.env -f docker-compose.yml -f docker-compose.gpu.yml --profile gpu config --quiet
```

No live Groq or Hugging Face request is part of these commands. Keep credentials, model artifacts and generated benchmark output outside Git.
