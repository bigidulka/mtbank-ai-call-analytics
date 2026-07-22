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

## Offline validation

```bash
docker compose --env-file tmp/release-ci.env config --quiet
docker compose --env-file tmp/release-ci.env -f docker-compose.yml -f docker-compose.gpu.yml --profile gpu config --quiet
```

No live Groq or Hugging Face request is part of these commands. Keep credentials, model artifacts and generated benchmark output outside Git.
