# 0004. Privacy-safe observability, provisional streaming and aggregate trends

## Status

Accepted for the offline implementation slice.

## Decision

The API exposes internal-only Prometheus metrics at `/metrics`. Metrics use fixed, non-content labels; JSON logs and span attributes remove audio, transcripts, prompts, provider bodies, keys, URLs and query strings. The Compose monitoring stack is private and provisioned from files.

`/ws/transcribe` is a bearer-authenticated, origin-checked, bounded protocol. It accepts only sequenced `pcm_s16le` (16 kHz mono) or continuous `ogg_opus` (48 kHz mono) frames; ambiguous `opus` and raw Opus packets are rejected. When explicitly enabled, the public API uses a no-proxy internal WebSocket adapter to reach `/v1/stream`; the internal service preserves a persistent bounded FFmpeg decoder and submits fixed-cadence bounded PCM rings as temporary WAV to Groq only. Each rolling request has timeout, call-count, audio and concurrency bounds; provider errors skip provisional output without retry/fallback. The first partial may be unstable, later common-prefix updates stabilize, and canonical batch reconciliation follows `end`. Defaults stay disabled until controlled runtime evidence exists.

`/trends` reads only `SanitizedAnalysisRecord` through a parameterized repository seam. It requires a bounded window and at least five calls; each rate carries the complete denominator run set and matching evidence run IDs. It has no transcript, SQL, filesystem or arbitrary HTTP tool.

## Release gates

The repository deliberately does not download models or exercise external credentials. A deployment must independently demonstrate controlled Groq rolling streaming p95 below three seconds on the target workload, canonical reconciliation for `pcm_s16le` and `ogg_opus`, approved remote raw-audio disclosure, OTLP export to Tempo, and Grafana browser access after the pinned monitoring images are available.
