# Release checklist

Run `uv run python scripts/check_release_gate.py`. A result of `blocked` is not a
passing release result, even when unit tests pass. `--allow-blocked` only returns zero
so CI or an operator can save a diagnostic blocked-gate report; it never approves a
release and does not turn absent evidence into evidence. Store that report as an
external CI/runtime artifact, not in `release-evidence/` or the repository.

## Required immutable evidence

- [ ] Licensed corpus approval for the authored synthetic corpus: 5 calls, 714.802
  seconds, reference text/roles, 8 kHz and WAV/MP3/OGG. The committed provenance and
  external Groq WER baseline do not replace final licence/retention approval.
- [ ] Controlled Groq `whisper-large-v3-turbo` + local Community-1 WER/DER/role report for the same corpus, produced by `evaluate_canonical_speech.py`.
- [ ] Reviewed gated local Community-1 artifact and manifest hash; no local ASR/alignment artifacts exist.
- [ ] Groq credentials and approved remote raw-audio disclosure for the exact controlled corpus/runtime scope.
- [ ] Formal deviation from assignment wording is approved: canonical ASR is Groq, not local `faster-whisper`; no `faster-whisper`, `openai-whisper` or ASR fallback is claimed.
- [x] Development OpenAI-compatible CLIProxyAPI model identity and local four-agent
  live smoke with retrieval and terminal submissions.
- [ ] Final gateway HTTPS nonce-bound release attestation and canonical trace artifact
  with sanitized provider request ID hashes.
- [ ] GPU benchmark from a self-hosted GPU runner, including workload revision and
  measured limits; no CPU-derived GPU SLA claim.
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
