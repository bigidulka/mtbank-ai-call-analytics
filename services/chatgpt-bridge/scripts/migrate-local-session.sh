#!/usr/bin/env bash
# Moves an already-paired local ChatGPT Web session into a remote chatgpt-bridge
# container without a browser or SSH port-forward, reusing the daemon's own
# /internal/pair endpoint (the same one the browser extension calls).
#
# The credential is piped from the local OS keyring straight through SSH into
# the remote container's python3 process over stdin. It is never written to a
# file on either machine, never passed as a command-line argument (so it never
# appears in `ps`), and never captured by this script's own output.
#
# Usage:
#   services/chatgpt-bridge/scripts/migrate-local-session.sh <ssh-host> <deploy-path>
#
# Requires libsecret's secret-tool locally, and that the remote chatgpt-bridge
# container is already running (deploy/speech-backend bridge-up).
set -euo pipefail

HOST="${1:?usage: migrate-local-session.sh <ssh-host> <deploy-path>}"
DEPLOY_PATH="${2:?usage: migrate-local-session.sh <ssh-host> <deploy-path>}"
COMPOSE_FILES=(-f "$DEPLOY_PATH/docker-compose.yml" -f "$DEPLOY_PATH/docker-compose.custom-speech.yml" -f "$DEPLOY_PATH/docker-compose.chatgpt-bridge.yml")

if ! command -v secret-tool >/dev/null 2>&1; then
  echo "secret-tool not found (package: libsecret-tools / libsecret)" >&2
  exit 1
fi

echo "Issuing one-time pairing code on $HOST ..." >&2
PAIRING_CODE="$(ssh "$HOST" docker compose "${COMPOSE_FILES[@]}" exec -T chatgpt-bridge \
  chatgpt-transcribe-connect pair | sed -n 's/^Pairing code: //p')"
if [[ -z "$PAIRING_CODE" ]]; then
  echo "Failed to obtain a pairing code from $HOST" >&2
  exit 1
fi

echo "Migrating local session credential (never leaves stdin on either side) ..." >&2
secret-tool lookup service chatgpt-transcribe-connect account chatgpt-web-session | \
  ssh "$HOST" docker compose "${COMPOSE_FILES[@]}" exec -T chatgpt-bridge \
    python3 -c '
import json, sys, urllib.error, urllib.request

pairing_code = sys.argv[1]
credentials = json.load(sys.stdin)
credentials["pairing_code"] = pairing_code
body = json.dumps(credentials).encode()
request = urllib.request.Request(
    "http://127.0.0.1:37182/internal/pair",
    data=body,
    headers={"Content-Type": "application/json", "Origin": "chrome-extension://migration"},
)
try:
    with urllib.request.urlopen(request, timeout=10) as response:
        print(response.read().decode())
except urllib.error.HTTPError as error:
    print(error.read().decode(), file=sys.stderr)
    raise SystemExit(1)
' "$PAIRING_CODE"

echo "Done. Verify with: ssh $HOST docker compose ${COMPOSE_FILES[*]} exec -T chatgpt-bridge chatgpt-transcribe-connect status" >&2
