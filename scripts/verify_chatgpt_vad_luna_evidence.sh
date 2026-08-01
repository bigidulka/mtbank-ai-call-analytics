#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

export PYTHONHASHSEED=0
EVIDENCE=release-evidence/chatgpt-vad-luna
TMP_DIR=$(mktemp -d "$ROOT/tmp/verify-chatgpt-vad-luna.XXXXXX")
trap 'rm -rf "$TMP_DIR"' EXIT

uv run python scripts/build_vad_anchor_corpus.py \
  --transcriptions "$EVIDENCE/chatgpt-web-transcriptions.json" \
  --output "$TMP_DIR/vad-corpus.json"
uv run python scripts/evaluate_vad_rank_alignment.py \
  --corpus "$TMP_DIR/vad-corpus.json" \
  --roles "$EVIDENCE/semantic-roles.json" \
  --output "$TMP_DIR/candidate-evaluation.json"
uv run python scripts/evaluate_boundaries.py \
  --hypotheses "$TMP_DIR/candidate-evaluation.json" \
  --output "$TMP_DIR/boundary-evaluation.json"
uv run python scripts/evaluate_vad_rank_loco.py \
  --transcriptions "$EVIDENCE/chatgpt-web-transcriptions.json" \
  --roles "$EVIDENCE/semantic-roles.json" \
  --output "$TMP_DIR/loco-validation.json"
uv run python scripts/compare_speech_profiles.py \
  --candidate "$TMP_DIR/candidate-evaluation.json" \
  --output "$TMP_DIR/runpod-comparison.json"

uv run python - "$EVIDENCE" "$TMP_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

evidence = Path(sys.argv[1])
generated = Path(sys.argv[2])

candidate = json.loads((generated / "candidate-evaluation.json").read_text())
boundary = json.loads((generated / "boundary-evaluation.json").read_text())
loco = json.loads((generated / "loco-validation.json").read_text())
comparison = json.loads((generated / "runpod-comparison.json").read_text())

expected_candidate = json.loads((evidence / "candidate-evaluation.json").read_text())
expected_boundary = json.loads((evidence / "boundary-evaluation.json").read_text())
expected_loco = json.loads((evidence / "loco-validation.json").read_text())

checks = {
    "wer": (candidate["micro"]["wer"]["wer"], expected_candidate["micro"]["wer"]["wer"]),
    "speaker_attributed_wer": (
        candidate["micro"]["speaker_attributed_wer"],
        expected_candidate["micro"]["speaker_attributed_wer"],
    ),
    "micro_der": (candidate["micro"]["der"]["der"], expected_candidate["micro"]["der"]["der"]),
    "role_accuracy": (
        candidate["micro"]["time_weighted_role_accuracy"],
        expected_candidate["micro"]["time_weighted_role_accuracy"],
    ),
    "boundary_median": (
        boundary["micro"]["median_absolute_error_seconds"],
        expected_boundary["micro"]["median_absolute_error_seconds"],
    ),
    "boundary_p95": (
        boundary["micro"]["p95_absolute_error_seconds"],
        expected_boundary["micro"]["p95_absolute_error_seconds"],
    ),
    "loco_macro_der": (loco["macro_held_out_der"], expected_loco["macro_held_out_der"]),
}
for name, (actual, expected) in checks.items():
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(f"{name} mismatch: {actual} != {expected}")
if comparison.get("status") != "completed":
    raise SystemExit("profile comparison did not complete")
print(json.dumps({name: actual for name, (actual, _) in checks.items()}, sort_keys=True))
PY
