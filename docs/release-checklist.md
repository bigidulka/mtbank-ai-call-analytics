# Release checklist

Этот checklist разделяет обязательный assignment handoff и дополнительные production/organizational gates. Отсутствие внешней registry attestation или разрешения на реальные клиентские данные не блокирует synthetic/no-PII demo, но блокирует любые production claims.

## Assignment handoff — verified

- [x] Пять authored synthetic звонков: 714.802 секунды, WAV/MP3/OGG, 8/16 kHz, reference text/timestamps/roles и SHA-256 provenance.
- [x] Canonical CUDA `float16` runtime: local `faster-whisper` large-v3-turbo + local pyannote Community-1, без ASR fallback.
- [x] Five-file WER/DER/role evaluation: [`../release-evidence/final-115/canonical-speech-evaluation.json`](../release-evidence/final-115/canonical-speech-evaluation.json).
- [x] Public 300-second HTTPS analysis under 60 seconds: [`../release-evidence/final-115/public-five-minute-sla.json`](../release-evidence/final-115/public-five-minute-sla.json).
- [x] App-plane runtime binding to immutable GPU digest: [`../release-evidence/final-115/runtime-binding.json`](../release-evidence/final-115/runtime-binding.json).
- [x] OpenWebUI attachment production flow plus boundary checks: [`../release-evidence/final-115/openwebui-attachment-e2e.json`](../release-evidence/final-115/openwebui-attachment-e2e.json).
- [x] Trends over persisted sanitized calls: [`../release-evidence/final-115/trends-response.json`](../release-evidence/final-115/trends-response.json).
- [x] WebSocket live diagnostic p95 below 3 seconds with canonical reconciliation: [`../release-evidence/final-115/websocket-p95.json`](../release-evidence/final-115/websocket-p95.json).
- [x] Prometheus and provisioned Grafana dashboard for calls, quality, topics, latency, failures and agent tokens.
- [x] Final evidence files linked by SHA-256 and byte count: [`../release-evidence/final-115/manifest.json`](../release-evidence/final-115/manifest.json).
- [x] Offline tests, Ruff, formatting and Pyright green on submission branch.

## External/production gates — intentionally not claimed

- [ ] Independent external registry/model-artifact attestation.
- [ ] Organizational approval for real customer audio, banking secrecy and PII processing.
- [ ] Repeated load/stability study on noisy production calls.
- [ ] Signed production image/SBOM and independent competitor score.
- [ ] Production SLO, incident response and long-lived credential-rotation approval.

`uv run python scripts/check_release_gate.py` remains a conservative production-readiness diagnostic. `blocked` means production/external attestation is incomplete; it does not invalidate assignment-scoped synthetic demo evidence above. `--allow-blocked` never converts missing evidence into evidence.
