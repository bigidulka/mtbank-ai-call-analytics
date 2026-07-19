# Operations and E2E evidence

## Speech runtime

The sole canonical batch path is Groq `whisper-large-v3-turbo` plus local offline
`pyannote/speaker-diarization-community-1`. `GROQ_API_KEY` is passed only to speech as
`MTBANK_SPEECH__GROQ__API_KEY`; no host port is published for speech, and its dedicated
`speech-egress` network is the only Compose egress for raw-audio ASR. Runtime does not
download Hugging Face artifacts. `HF_TOKEN` is required only in a separately approved
Community-1 provisioning environment.

No live Groq/HF request is part of offline validation. Streaming implementation is Groq-only:
fixed-cadence bounded PCM is temporary WAV input to one Groq call, failures skip provisional
updates without retry/fallback, and canonical batch reconciliation remains authoritative.
Compose default remains disabled until controlled rollout. The exact blockers are approved
Groq credentials and remote-audio disclosure, reviewed Community-1 digest and artifact
manifest, then controlled WebSocket E2E quality/latency evidence including p95.

`uv run --offline --no-sync python scripts/check_release_gate.py --allow-blocked`
is diagnostic only: it permits writing a blocked-gate report without approving a release.
Write the report to an ignored runtime path or attach it in CI; release evidence remains
an external CI/runtime artifact and must not be committed or synthesized locally.

CPU compose validation uses only synthetic `.env` values and never downloads models:

```bash
docker compose --env-file tmp/release-ci.env config --quiet
docker compose --env-file tmp/release-ci.env -f docker-compose.yml -f docker-compose.gpu.yml --profile gpu config --quiet
```

The canonical app image uses frozen `uv.lock`. If PyPI is unavailable, do not weaken
hashes or regenerate locks. Stage only reviewed wheels whose hashes satisfy the frozen
export into the build-context directory, then select the offline branch explicitly:

```bash
install -d docker/wheelhouse
cp /secure/reviewed-wheelhouse/*.whl docker/wheelhouse/
docker build --build-arg USE_WHEELHOUSE=1 -f docker/app.Dockerfile -t mtbank-ai-foundation:offline .
```

`docker/wheelhouse/.gitkeep` is tracked, but wheels are not. The image copies the
directory to `/opt/wheelhouse`, uses `--no-index --find-links /opt/wheelhouse
--require-hashes`, then removes it. An unavailable canonical Docker build remains a
release blocker. This repository has no wheels staged, so this offline path is not
claimed as executed.

## Cloudflare Tunnel preflight

`docker-compose.cloudflare.yml` is an overlay for `docker-compose.yml`: it pins the
linux/amd64 `cloudflared` 2026.7.1 child manifest, waits for a healthy gateway, publishes
no host port and connects only to `gateway-ingress`. The connector receives its remotely
managed tunnel token only as the Compose file secret
`/run/secrets/cloudflare_tunnel_token`; never place that token in `.env`, command-line
arguments, logs or repository files.

The provisioner reads only process environment: `CF_EMAIL`, `CF_GLOBAL_API_KEY`,
`CLOUDFLARE_ZONE`, `CLOUDFLARE_HOSTNAME`,
`CLOUDFLARE_TUNNEL_NAME` and `CLOUDFLARE_TUNNEL_TOKEN_FILE`. Obtain credentials through
the approved credential source at execution time; do not read or copy `vpn-ops/.env`.
The connector token destination must be writable by root because `prepare --apply` sets
its final mode to `0400` and UID/GID to `65532` atomically.

```bash
export CF_EMAIL=...
export CF_GLOBAL_API_KEY=...
export CLOUDFLARE_ZONE=cloud-tunnel-mega-obx1.space
export CLOUDFLARE_HOSTNAME=cloud.cloud-tunnel-mega-obx1.space
export CLOUDFLARE_TUNNEL_NAME=mtbank-ai-gateway
export CLOUDFLARE_TUNNEL_TOKEN_FILE="$PWD/tmp/cloudflared-tunnel-token"
python scripts/provision_cloudflare_tunnel.py prepare
```

Without `--apply`, both `prepare` and `publish` are dry runs with no mutation. After the
runtime is ready and a separate deployment approval is given, `prepare --apply` may create
or reuse only an exactly owned remotely managed tunnel, verifies exact ingress
`hostname → http://gateway:8080` followed by terminal 404, and writes the protected
connector token. Reused configuration drift fails without overwrite. `publish --apply`
starts only the Cloudflare overlay via argv, waits for a bounded healthy control-plane
connection, then creates the exact proxied CNAME only when absent; DNS multiplicity or
drift fails without takeover. Actual external side effects were not run for this release
preparation.

For an approved rollback: remove the owned DNS record first, stop the `cloudflared`
connector second, then optionally delete the dedicated tunnel last. Do not reverse this
order. CI/runtime evidence remains external and is not committed.

`.github/workflows/ci.yml` runs offline checks, disposable PostgreSQL migrations and
the canonical image build. `e2e-real.yml` is scheduled/manual and fails with
`MTBANK_RELEASE_GATE=1` when cloud gateway/model credentials, the approved HTTPS E2E
API endpoint or a nonce-bound live attestation endpoint are absent. The attestation
must bind the same invocation nonce, run endpoint, run ID, provider/model/revision and
canonical trace-artifact hash. Its trace must show four distinct agent trajectories,
retrieval, four terminal submissions and distinct provider request IDs; the exported
artifact stores only typed provenance and ID hashes. A hand-authored local trace is not
accepted as real E2E evidence.

`gpu-benchmark.yml` is manual/self-hosted. It cannot claim a GPU SLA from a CPU runner.
The workflow requires a configured controlled WebSocket target, its approved origin and a
secret API key; it invokes `nvidia-smi -L`, runs the paced client and emits typed evidence
only from that run. Missing NVIDIA, image digest, local manifest or completed WebSocket
observation fails the workflow. CPU output from `run_websocket_benchmark.py` is explicitly
diagnostic-only, never GPU evidence.

WebSocket remains disabled in default Compose. For an approved controlled rollout, require
one exact browser origin and apply the opt-in overlay:

```bash
export MTBANK_WEBSOCKET_ALLOWED_ORIGIN=https://approved.example
# supply the existing required .env secrets separately
docker compose -f docker-compose.yml -f docker-compose.websocket.yml up --build --wait
```

The overlay enables both public API and speech streaming and uses mandatory Compose
interpolation for `MTBANK_WEBSOCKET_ALLOWED_ORIGIN`; it does not publish a new host port.
Do not treat a successful `docker compose config` or CPU diagnostic as runtime proof.
Grafana browser proof and WebSocket GPU p95 proof are separate release gates, not inferred
from compose configuration.
