# Operations

## Runtime

Canonical speech processing is Groq `whisper-large-v3-turbo` plus local offline `pyannote/speaker-diarization-community-1`. `GROQ_API_KEY` is supplied only to the speech service. Runtime does not download diarization artifacts; provision them separately before startup.

Default Compose keeps streaming disabled. To enable a controlled WebSocket rollout, set one explicit browser origin and apply the overlay:

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
