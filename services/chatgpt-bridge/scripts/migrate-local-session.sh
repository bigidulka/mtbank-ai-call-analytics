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
# ssh joins multiple trailing arguments with spaces and re-parses that on the
# remote side, which corrupts any quoting/redirection that only existed in the
# local shell's parse tree. Every remote command here is therefore built and
# passed as exactly one pre-assembled string.
#
# Usage:
#   services/chatgpt-bridge/scripts/migrate-local-session.sh <ssh-host> <deploy-path>
#
# Requires libsecret's secret-tool locally, and that the remote chatgpt-bridge
# container is already running (deploy/speech-backend bridge-up).
set -euo pipefail

HOST="${1:?usage: migrate-local-session.sh <ssh-host> <deploy-path>}"
DEPLOY_PATH="${2:?usage: migrate-local-session.sh <ssh-host> <deploy-path>}"

# Both values are spliced into a string that the remote shell re-parses, so each
# is constrained to characters that carry no meaning there.
if [[ ! "$DEPLOY_PATH" =~ ^/[A-Za-z0-9._/-]*$ ]]; then
  echo "deploy path must be absolute and free of shell metacharacters: $DEPLOY_PATH" >&2
  exit 2
fi

COMPOSE_PREFIX="docker compose -f $DEPLOY_PATH/docker-compose.yml -f $DEPLOY_PATH/docker-compose.custom-speech.yml -f $DEPLOY_PATH/docker-compose.chatgpt-bridge.yml exec -T chatgpt-bridge"

if ! command -v secret-tool >/dev/null 2>&1; then
  echo "secret-tool not found (package: libsecret-tools / libsecret)" >&2
  exit 1
fi

echo "Issuing one-time pairing code on $HOST ..." >&2
PAIRING_CODE="$(ssh "$HOST" "$COMPOSE_PREFIX chatgpt-transcribe-connect pair" | sed -n 's/^Pairing code: //p')"
# PairingState::issue emits base64url-no-pad of 24 random bytes.
if [[ ! "$PAIRING_CODE" =~ ^[A-Za-z0-9_-]{16,128}$ ]]; then
  echo "Failed to obtain a well-formed pairing code from $HOST" >&2
  exit 1
fi

echo "Placing migration helper in the container (contains no secret) ..." >&2
ssh "$HOST" "$COMPOSE_PREFIX tee /tmp/pair-migrate.py >/dev/null" <<'PY'
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
PY

echo "Migrating local session credential (never leaves stdin on either side) ..." >&2
secret-tool lookup service chatgpt-transcribe-connect username chatgpt-web-session | \
  ssh "$HOST" "$COMPOSE_PREFIX python3 /tmp/pair-migrate.py $PAIRING_CODE"

ssh "$HOST" "$COMPOSE_PREFIX rm -f /tmp/pair-migrate.py"

echo "Done. Verify with: ssh $HOST '$COMPOSE_PREFIX chatgpt-transcribe-connect status'" >&2
