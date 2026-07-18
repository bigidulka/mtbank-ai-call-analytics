# ADR 0002: canonical Groq batch speech pipeline

- Status: accepted for implementation; real-provider and release evidence remain separately blocked.
- Date: 2026-07-18

## Context

The canonical pipeline must preserve pyannote speaker IDs, never infer roles from
speaker order, and make one ASR provider call. `pyannote/speaker-diarization-community-1`
is gated: `HF_TOKEN` is permitted only in a controlled provisioning environment, never
in the runtime container. Groq receives normalized raw audio, so that disclosure requires
an approved privacy/legal basis before a live run.

## Decision

`services/speech` is an internal-only FastAPI service. Its only batch production path is:

```text
bytes → magic/MIME/size → ffprobe → deterministic FFmpeg WAV
      → Groq `/openai/v1/audio/transcriptions` exactly once
        model=whisper-large-v3-turbo, language=ru, temperature=0,
        response_format=verbose_json, word+segment timestamps
      → aligned timestamp passthrough (no WhisperX/model)
      → local offline pyannote Community-1 artifact
      → deterministic maximum-time-overlap word-to-speaker merge
      → metadata role map → existing bounded policy resolver
      → complete exact-role TranscriptSnapshot
```

The Groq adapter uses existing `httpx` with `trust_env=False`, redirects disabled,
identity encoding and bounded raw responses. It logs neither audio body nor secret.
The API key is a `SecretStr` supplied only through `MTBANK_SPEECH__GROQ__API_KEY` from
`GROQ_API_KEY`; it is sent only in an Authorization header. Snapshot provenance records
Groq provider/model, endpoint SHA-256 fingerprint, returned request ID when available,
and available usage duration without including a secret.

Groq verbose timestamps are treated as already aligned. There is no faster-whisper,
WhisperX, local ASR fallback, Replicate, Deepgram or pyannoteAI path. The overlap assigner
chooses the maximum positive intersection for every word; ties use stable turn ordering,
consecutive same-speaker words are grouped, and zero-overlap words remain unassigned.
Existing role policy then fails closed if complete role assignment is unavailable; it never
uses a first-speaker heuristic.

## Local artifact and egress boundaries

The local manifest and readiness probe contain only the Community-1 diarization artifact.
Runtime has `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; provisioning validates a
reviewed artifact digest before Hub access and needs `HF_TOKEN` only for that provisioning.
Speech has no host port. It is attached to `application-internal` and its dedicated
`speech-egress` network; no other service receives the Groq key.

Streaming uses the same Groq-only provider for provisional updates. A bounded 16 kHz mono
PCM ring is written to a temporary WAV, then one bounded Groq call runs in a dedicated
rolling semaphore. Fixed cadence, maximum call count and maximum submitted-audio budget
bound the session; timeout/429/provider failure is not retried and merely skips that
provisional update. The first result may be unstable, later results use common-prefix
stabilization, and the public WebSocket reconciles complete audio through canonical batch.
No local faster-whisper, Silero, Replicate, Deepgram or other fallback exists. Compose
remains disabled by default until a controlled rollout; no p95 or live benchmark is claimed.

## Evidence and blockers

No live Groq request, Community-1 provisioning, Docker build/up, deployment, or benchmark
was performed for this decision. Release remains blocked on approved Groq credentials,
remote raw-audio disclosure approval, a reviewed gated Community-1 artifact digest and
manifest, controlled E2E, and measured quality/latency evidence.
