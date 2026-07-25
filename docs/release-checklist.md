# Release checklist

Run `uv run python scripts/check_release_gate.py`. A result of `blocked` is not a
passing release result, even when unit tests pass. `--allow-blocked` only returns zero
so CI or an operator can save a diagnostic blocked-gate report; it never approves a
release and does not turn absent evidence into evidence. Store that report as an
external CI/runtime artifact, not in `release-evidence/` or the repository.

`declared_image_digest` и `reviewer_reference_sha256` — только локальные correlation fields. Echo
`/v1/runtime` и hash идентификатора reviewer не доказывают registry provenance или независимую
external проверку. Поэтому `independent_external_attestation` остаётся hard-blocked: approval
выполняется и фиксируется независимой внешней процедурой, а не self-reported evidence в checkout.

## Required immutable evidence

- [ ] Licensed corpus approval for the authored synthetic corpus: 5 calls, 714.802
  seconds, reference text/roles, 8 kHz and WAV/MP3/OGG.
- [ ] Complete local `faster-whisper` + Community-1 WER/DER/role report for all five files, produced by `evaluate_canonical_speech.py`.
- [ ] External independent approval for `models/manifest.json` schema v3, verified ASR/Community-1 artifact tree hashes, file counts, and revisions. `reviewer_reference_sha256` may correlate the external record but cannot attest it.
- [ ] Groq credentials and approved remote raw-audio disclosure only for opt-in WebSocket provisional mode.
- [x] Canonical batch ASR uses local `faster-whisper` medium+ equivalent (`large-v3-turbo`); ASR fallback is absent.
- [x] Development OpenAI-compatible CLIProxyAPI model identity and local four-agent
  live smoke with retrieval and terminal submissions.
- [ ] Final gateway HTTPS nonce-bound release attestation and canonical trace artifact
  with sanitized provider request ID hashes.
- [ ] GPU benchmark from a self-hosted GPU runner, including workload revision, measured limits, and a bearer-fetched app-plane `/v1/benchmark-runtime-binding` response on the same authority as `/ws/transcribe`. It reports the configured remote CUDA `float16` runtime and schema-v3 component revisions without exposing the RunPod bearer. Its `declared_image_digest` echo is not external registry attestation; no local Docker image ID or CPU-derived GPU SLA claim.
- [ ] Live Grafana browser proof, cleaned of PII.
- [ ] WebSocket GPU p95 proof for the approved workload.
- [ ] Canonical Docker application image built from frozen dependencies and its final
  canonical image digest. If PyPI is unavailable, use only the documented trusted wheelhouse path; do
  not remove hashes.
- [ ] Final competitor score on the immutable release SHA.

## Current status

Для project-generated synthetic corpus scope-limited remote-audio разрешение должно
быть проверено отдельно; organizational approval для реальных клиентских/production данных
не предоставлено. Текущий checkout не содержит доказательства controlled canonical speech
run, Community-1 artifact review, real four-agent E2E, Grafana browser proof или GPU run.
CPU WebSocket diagnostic не является GPU evidence. Frozen static competitor benchmark не
заменяет runtime proof, а все comparative scores остаются `unknown`. Final gateway nonce
attestation, canonical metrics, GPU benchmark, GPU WS p95, Grafana artifact, canonical image
digest, signed image bundle и verified competitor score заблокированы до соответствующей
external attestation.

Before an approval, verify source/lock hashes, policy/prompt/dataset/model revisions,
unit/contract and disposable migration results, and the generated release-gate manifest.
SBOM is attached only when the real `syft` tool is available; no synthetic SBOM
placeholder is emitted.
