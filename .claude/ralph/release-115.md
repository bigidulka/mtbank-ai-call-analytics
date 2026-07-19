# MTBank AI Call Analytics — release evidence

## Current architecture decision

- Canonical batch ASR is only Groq OpenAI-compatible
  `/openai/v1/audio/transcriptions` with `whisper-large-v3-turbo`, `language=ru`,
  `temperature=0`, `response_format=verbose_json`, and word/segment timestamps.
- Local runtime contains only pinned offline
  `pyannote/speaker-diarization-community-1`; no faster-whisper, WhisperX, local-ASR,
  Replicate, Deepgram or pyannoteAI fallback exists.
- Groq timestamps pass through as aligned evidence. Local deterministic maximum-overlap
  assignment preserves pyannote speaker IDs; the existing policy resolver maps roles and
  never uses a first-speaker heuristic.
- Speech has no host port. Only speech receives `GROQ_API_KEY`, on its controlled
  `speech-egress` network. Raw normalized audio is disclosed to Groq only after approved
  privacy/legal review. The key is never committed, logged, queried, or passed to API.
- `HF_TOKEN` remains only for controlled gated Community-1 provisioning. Runtime remains
  offline and requires the reviewed local artifact manifest.
- Groq-only rolling streaming is implemented: bounded PCM ring → temporary WAV → one Groq
  call with fixed cadence, call/audio budgets and no retry/fallback; canonical batch
  reconciles end audio. Compose default remains disabled pending controlled rollout, and no
  hidden local faster-whisper or Silero path is retained.

## Remaining release checklist

1. [x] Scope-limited synthetic remote-audio authorization recorded; Groq credential and
   real synthetic validation calls completed. Real customer data remains excluded.
2. [x] Reviewed Community-1 content digest, controlled provisioning and local
   diarization-only manifest completed at pinned revision.
3. [ ] Short controlled Groq+pyannote E2E passed; corpus-wide WER/DER/role metrics and
   five-minute SLA evidence remain.
4. [x] Compose up/readiness, REST and real OpenWebUI attachment Pipeline E2E passed.
5. [ ] Live Grafana/Prometheus, Trends, HTTPS and CPU-host provisional WebSocket p95 passed;
   GPU/GPU-WebSocket evidence and immutable signed release attestation remain.
6. [ ] Publish/commit/push only after explicit user approval.

## Verified offline scope

- Groq multipart/response-bound/no-fallback and deterministic overlap contracts are unit
  tested. Real external Groq ASR baseline on five synthetic files (714.802 s, 1 036
  reference words) measured micro WER 8.39768%; it is ASR-only and not canonical
  Groq+pyannote release evidence. A repeated sequential Groq-only script took 5.493 s
  wall time, not canonical Groq+pyannote latency.
- Community-1 revision `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee` is provisioned with
  reviewed content digest and local manifest; runtime remains offline.
- Full Compose migrations/readiness passed. Real REST and OpenWebUI attachment E2E returned
  schema-valid analyses with four distinct agent trajectories and 12 model calls.
- Five completed calls were persisted; live Trends returned `5/5 = 1.0`; Grafana dashboard
  and Prometheus successful-analysis series are live.
- Current local images: `mtbank-ai-speech-cpu:local`
  `sha256:d49698efed9c17000c57509e47e24242165b818be4798a1580fa994747c628fd` and
  `mtbank-ai-foundation:local`
  `sha256:5deb3e110428641529da869c138f23982e7f2884167fd1d3d2578376bbdde304`.
- Cloudflare tunnel and proxied DNS are published. Public HTTPS REST smoke returned HTTP 200;
  OpenWebUI and Grafana are reachable at `https://cloud.cloud-tunnel-mega-obx1.space`.

## Blockers

- Synthetic-only remote-audio authorization is recorded; real customer/production data remains excluded.
- No corpus-wide canonical WER/DER/role/speaker-attributed metrics or five-minute SLA evidence.
- Live CPU-host provisional streaming measured p50 0.427 s and p95/max 0.518 s over 4 updates; no GPU/GPU-WebSocket result or immutable signed release attestation.
- Host Docker bridge outbound NAT is unavailable; the live deployment uses a local
  host-network egress relay workaround and is not a portability claim.
- No commit or push was performed.
