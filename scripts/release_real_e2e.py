#!/usr/bin/env python3
"""Collects a nonce-bound live E2E attestation without retaining trace content."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

import httpx

from mtbank_ai.release.evidence import export_evidence, sha256, sha256_text
from mtbank_ai.release.gates import require_real_llm_environment

_REQUIRED_AGENTS = frozenset({"classifier", "quality", "compliance", "summarizer"})


def validate_trace(trace: object) -> dict[str, object]:
    """Accepts only a four-agent terminal trace with retrieval and provider IDs."""

    if not isinstance(trace, Mapping):
        raise ValueError("real E2E trace должен быть JSON object")
    trajectories = trace.get("trajectories")
    if not isinstance(trajectories, list) or len(trajectories) != 4:
        raise ValueError("real E2E trace должен содержать ровно четыре trajectories")
    agent_ids: set[str] = set()
    provider_ids: list[str] = []
    retrieval_calls = 0
    terminal_submissions = 0
    for trajectory in trajectories:
        if not isinstance(trajectory, Mapping):
            raise ValueError("trajectory должен быть object")
        agent_id = trajectory.get("agent_id")
        if not isinstance(agent_id, str):
            raise ValueError("trajectory должен иметь agent_id")
        agent_ids.add(agent_id)
        retrieval_calls += _nonnegative_int(trajectory.get("retrieval_calls"), "retrieval_calls")
        terminal_submissions += _nonnegative_int(trajectory.get("terminal_submissions"), "terminal_submissions")
        provider_request_id = trajectory.get("provider_request_id")
        if not isinstance(provider_request_id, str) or not provider_request_id:
            raise ValueError("trajectory должен подтверждать provider_request_id")
        provider_ids.append(provider_request_id)
    if agent_ids != _REQUIRED_AGENTS:
        raise ValueError("real E2E trace должен покрывать classifier, quality, compliance и summarizer")
    if retrieval_calls < 1 or terminal_submissions != 4 or len(set(provider_ids)) != 4:
        raise ValueError("real E2E trace не подтверждает retrieval, terminal submissions или distinct provider IDs")
    return {
        "agent_ids": sorted(agent_ids),
        "retrieval_calls": retrieval_calls,
        "terminal_submissions": terminal_submissions,
        "provider_request_ids_sha256": [sha256_text(value) for value in provider_ids],
    }


def validate_live_attestation(
    attestation: object,
    *,
    invocation_nonce: str,
    run_endpoint: str,
    code_sha: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Rejects anything except an attestation produced for this nonce-bound live invocation."""

    if not isinstance(attestation, Mapping):
        raise ValueError("real E2E attestation должен быть JSON object")
    required = {
        "schema_version",
        "kind",
        "code_sha",
        "invocation_nonce",
        "run_endpoint",
        "run_id",
        "provider",
        "trace",
    }
    if set(attestation) != required:
        raise ValueError("real E2E attestation содержит недопустимые поля")
    if (
        attestation.get("schema_version") != "1"
        or attestation.get("kind") != "real-agent-run-attestation"
        or attestation.get("code_sha") != code_sha
        or attestation.get("invocation_nonce") != invocation_nonce
        or attestation.get("run_endpoint") != run_endpoint
    ):
        raise ValueError("real E2E attestation не привязан к текущему invocation, endpoint или code SHA")
    run_id = attestation.get("run_id")
    provider = attestation.get("provider")
    trace = attestation.get("trace")
    if not isinstance(run_id, str) or not run_id or not isinstance(provider, Mapping) or not isinstance(trace, Mapping):
        raise ValueError("real E2E attestation не содержит run/provider/trace provenance")
    if set(provider) != {"provider_id", "model_id", "model_revision"} or not all(
        isinstance(value, str) and value for value in provider.values()
    ):
        raise ValueError("real E2E attestation не содержит provider/model revision provenance")
    if set(trace) != {"artifact_sha256", "trajectories"}:
        raise ValueError("real E2E attestation не содержит trace artifact provenance")
    trajectories = trace["trajectories"]
    artifact_sha256 = trace["artifact_sha256"]
    if not isinstance(artifact_sha256, str) or artifact_sha256 != sha256(trajectories):
        raise ValueError("real E2E trace artifact hash не совпадает с canonical trace")
    validated_trace = validate_trace({"trajectories": trajectories})
    provenance = {
        "invocation_nonce_sha256": sha256_text(invocation_nonce),
        "model_id": provider["model_id"],
        "model_revision": provider["model_revision"],
        "provider_id": provider["provider_id"],
        "run_endpoint_sha256": sha256_text(run_endpoint),
        "run_id_sha256": sha256_text(run_id),
        "trace_artifact_sha256": artifact_sha256,
    }
    return validated_trace, provenance


def _nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} должен быть неотрицательным integer")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attestation-url", required=True)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--code-sha", required=True)
    arguments = parser.parse_args()

    missing = require_real_llm_environment(os.environ)
    if missing:
        raise SystemExit("release real_llm blocked: не заданы " + ", ".join(missing))
    run_endpoint = os.environ.get("MTBANK_REAL_E2E_API_BASE_URL", "").strip()
    if not run_endpoint.startswith("https://"):
        raise SystemExit("release real_llm blocked: MTBANK_REAL_E2E_API_BASE_URL должен быть HTTPS URL")
    if not arguments.attestation_url.startswith("https://"):
        raise SystemExit("release real_llm blocked: attestation URL должен быть HTTPS URL")

    invocation_nonce = uuid4().hex
    response = httpx.post(
        arguments.attestation_url,
        json={"invocation_nonce": invocation_nonce, "run_endpoint": run_endpoint},
        headers={"X-MTBank-Release-Nonce": invocation_nonce},
        timeout=30.0,
    )
    response.raise_for_status()
    trace, provenance = validate_live_attestation(
        response.json(),
        invocation_nonce=invocation_nonce,
        run_endpoint=run_endpoint,
        code_sha=arguments.code_sha,
    )
    evidence = export_evidence(
        kind="real-agent-traces",
        code_sha=arguments.code_sha,
        provenance=provenance,
        metrics={
            "retrieval_calls": trace["retrieval_calls"],
            "terminal_submissions": trace["terminal_submissions"],
        },
        observations={
            "agent_ids": trace["agent_ids"],
            "provider_request_ids_sha256": trace["provider_request_ids_sha256"],
        },
    )
    arguments.evidence.parent.mkdir(parents=True, exist_ok=True)
    arguments.evidence.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
