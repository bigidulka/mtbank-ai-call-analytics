# Threat model

## Assets

Audio, transcript, prompts, provider credentials, agent outputs, evidence metadata,
model artifacts and evaluation corpus are sensitive assets. Integrity of policy,
prompt, model and code revisions is required for reproducibility.

## Boundaries and mitigations

- Browser traffic terminates only at localhost-bound nginx; backend services have no
  host port.
- OpenWebUI attachment metadata is untrusted: owner, size, MIME, magic bytes and hash
  are checked at the Pipeline boundary.
- API URL fetch uses a bounded SSRF-safe path; arbitrary provider endpoints are not
  accepted from a request.
- Public WebSocket audio is bearer-authenticated, origin-checked, sequenced and bounded;
  only PCM16 or continuous Ogg/Opus enters the no-proxy internal speech transport. Raw
  Opus packets are rejected, while persistent FFmpeg input, output and stderr are bounded.
- Cloud agents receive bounded/redacted inputs through one configured HTTPS gateway;
  retries, budgets, tool allowlists and terminal submission schemas are constrained.
- PostgreSQL persistence and release evidence exclude raw content. Evidence hashes
  provider request IDs rather than retaining them.
- CI does not print secrets and scheduled real E2E fails closed when secrets/config are
  absent.

## Residual risks and release decisions

Model supply-chain provenance, corpus licensing, real provider behavior, GPU capacity,
WebSocket latency and Grafana browser access require live evidence. No static test,
fake agent output or synthetic silence fixture closes these risks. See
[release-checklist.md](release-checklist.md).
