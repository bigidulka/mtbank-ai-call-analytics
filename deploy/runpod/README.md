# RunPod split-GPU speech deployment

RunPod GPU Pods run one container image. They cannot run Docker Compose or nested Docker. Keep the existing local Docker Compose stack as the control/app plane and deploy only the pinned GPU speech image to RunPod.

This is deployment foundation only. It makes no latency, SLA, availability, or benchmark claim.

## Boundary

```text
local Compose API -- HTTPS + Bearer --> RunPod speech container
local Compose API -- internal networks --> OpenWebUI, Pipelines, PostgreSQL, telemetry
```

No fallback to local speech is configured by `docker-compose.runpod.yml`. A failed remote speech service fails the API workflow closed.

## Build and publish the one speech image

Build outside a RunPod GPU Pod, for example in CI or a machine with Docker. Provisioned artifacts must be present before build. The runtime stays offline and verifies `models/manifest.json` against `/models/artifacts`.

```bash
# Example only: replace registry/image and immutable digest in deployment records.
docker build \
  --file docker/speech.gpu.Dockerfile \
  --tag registry.example/mtbank-speech-gpu:build-<code-sha> \
  .
docker push registry.example/mtbank-speech-gpu:build-<code-sha>
# Resolve pushed image to sha256 digest. Deploy image@sha256:<digest>, never a mutable tag.
```

Do not put Hugging Face credentials, model downloads, raw audio, or application `.env` into the RunPod image or template. Runtime model downloads are disabled by image configuration. The current repository `.dockerignore` excludes `models/`; use the persistent artifact-volume route above unless a separate audited release image intentionally embeds reviewed artifacts.

## RunPod Pod configuration

Create one GPU Pod using the published **image digest**. Configure:

- one RTX 4090 or stronger GPU;
- container port `8010` exposed as an authenticated HTTPS endpoint through RunPod's proxy/domain;
- persistent volume mounted at `/workspace/models` for `manifest.json` and both verified artifact directories; set `MTBANK_SPEECH__MODELS__MANIFEST_PATH=/workspace/models/manifest.json` and `MTBANK_SPEECH__MODELS__ARTIFACT_ROOT=/workspace/models/artifacts` when using this volume;
- provision artifacts into that volume before container start or CUDA warmup, then verify their manifest digests; models are Docker-ignored and the published runtime image neither copies nor downloads them;
- environment variables from [`env.example`](env.example), injected through RunPod secrets/environment UI, never command history; set `MTBANK_SPEECH__ACCESS__MODE=bearer`, `MTBANK_SPEECH__ACCESS__BEARER_KEY` equal to `MTBANK_RUNPOD_SPEECH_BEARER_KEY`, and `MTBANK_SPEECH__RUNTIME__IMAGE_DIGEST` to the deployed immutable `sha256:<64 lowercase hex>` image digest;
- no Docker Compose, Docker-in-Docker, or additional full application stack in the Pod.

The service has no public anonymous processing endpoint in bearer mode:

- anonymous: `GET /health/live` only;
- bearer required: `GET /health/ready`, `GET /v1/runtime`, `POST /v1/transcribe`, and `/v1/stream`;
- authorization is exactly one `Authorization: Bearer <key>` header.

Check non-content runtime facts after the Pod is healthy:

```bash
curl --fail --silent --show-error \
  --header "Authorization: Bearer $MTBANK_RUNPOD_SPEECH_BEARER_KEY" \
  "$MTBANK_RUNPOD_SPEECH_BASE_URL/v1/runtime"
```

`/v1/runtime` returns configured device, CTranslate2 compute type, model identifiers, and the configured declared image digest only. It returns no transcript, audio, artifact path, tag, token, or secret. The GPU runner must not call this split-plane endpoint directly: it uses the protected app-plane `/v1/benchmark-runtime-binding` endpoint on the same public authority as `/ws/transcribe`; the app fetches this response from its configured remote backend server-side. Neither a self-reported digest nor its hash proves external registry provenance or independent attestation.

To enable provisional streaming, set `MTBANK_SPEECH__STREAMING__ENABLED=true` and inject `MTBANK_SPEECH__GROQ__API_KEY` through RunPod secrets. The API connects directly to `/v1/stream` as `wss`, sends exactly one RunPod bearer header, disables proxy/compression, and rejects handshake redirects; Groq is used only for bounded provisional updates while local ASR and Community-1 remain canonical batch reconciliation.

## Local app-plane overlay

Create a protected local file from the example and apply only to the local Compose control plane:

```bash
cp deploy/runpod/env.example deploy/runpod/env.local
chmod 600 deploy/runpod/env.local
set -a && . deploy/runpod/env.local && set +a
docker compose -f docker-compose.yml -f docker-compose.runpod.yml up --build --wait
```

The overlay switches API speech transport to `remote_https`. It does not create, alter, or deploy anything on RunPod.

## Teardown

Stop the RunPod Pod after testing to stop GPU compute charges. Destroy any temporary volume only after retaining required non-content deployment provenance elsewhere. Do not retain raw audio or transcripts in Pod storage.
