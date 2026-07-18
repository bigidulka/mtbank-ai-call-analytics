from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from mtbank_ai.agent_runtime import (
    CapabilityProbeRunner,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ModelToolCall,
    ModelUsage,
    ProbeMode,
    PromptRegistry,
    PromptRegistryError,
)
from mtbank_ai.agent_runtime.capabilities import CapabilityName, CapabilityProbeError, StreamingProbeResult
from mtbank_ai.agent_runtime.contracts import FunctionToolSchema

NOW = datetime(2026, 7, 16, tzinfo=UTC)


class ScriptedProbeClient:
    def __init__(self) -> None:
        self.calls: list[ModelRequest] = []
        self._responses = [
            _response(_call("probe_echo", "one")),
            _response(_call("probe_echo", "two")),
            _response(_call("probe_echo", "three"), _call("probe_second", "four")),
            _response(_call("probe_system", "system", value="system-nonce")),
            _response(_call("probe_echo", "five"), output_tokens=8),
        ]

    async def complete(self, request: ModelRequest, *, deadline_at: datetime) -> ModelResponse:
        del deadline_at
        self.calls.append(request)
        return self._responses.pop(0)

    async def probe_streaming(self, request: ModelRequest, *, deadline_at: datetime) -> StreamingProbeResult:
        del request, deadline_at
        return StreamingProbeResult(
            model_id="configured-model",
            cancelled=True,
            usage=ModelUsage(input_tokens=1, output_tokens=8, total_tokens=9),
            limit_enforced=True,
        )


def _call(name: str, call_id: str, *, value: str = "probe") -> ModelToolCall:
    return ModelToolCall(id=call_id, name=name, arguments_json=f'{{"value":"{value}"}}')


def _response(
    *calls: ModelToolCall,
    output_tokens: int = 1,
    model_id: str = "configured-model",
    text: bool | None = None,
) -> ModelResponse:
    return ModelResponse(
        request_id="probe-request",
        model_id=model_id,
        finish_reason="tool_calls" if calls else "stop",
        tool_calls=calls,
        usage=ModelUsage(input_tokens=1, output_tokens=output_tokens, total_tokens=1 + output_tokens),
        latency_ms=1,
        has_text_content=not calls if text is None else text,
    )


def test_prompt_registry_hashes_canonical_inputs_and_blocks_escape(tmp_path: Path) -> None:
    root = tmp_path / "prompts"
    prompt_dir = root / "quality"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "v1.md").write_text("Reviewed prompt\r\n", encoding="utf-8")
    registry = PromptRegistry(root)
    root_link = tmp_path / "prompt-root-link"
    root_link.symlink_to(root, target_is_directory=True)
    with pytest.raises(PromptRegistryError, match="root"):
        PromptRegistry(root_link)
    tools = (
        FunctionToolSchema(
            name="lookup",
            description="Lookup evidence.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        ),
    )

    bundle = registry.load("quality", "v1", policy_inputs={"version": "policy/v1"}, tool_schemas=tools)

    assert bundle.text == "Reviewed prompt\n"
    assert bundle.reference.content_hash == hashlib.sha256(b"Reviewed prompt\n").hexdigest()
    assert bundle.reference.bundle_hash != bundle.reference.content_hash
    assert len(bundle.policy_hash) == 64
    assert len(bundle.tool_schema_hash) == 64
    with pytest.raises(PromptRegistryError):
        registry.load("../outside", "v1", policy_inputs={}, tool_schemas=tools)

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "v1.md").write_text("outside", encoding="utf-8")
    (root / "escaped").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PromptRegistryError, match="symlink"):
        registry.load("escaped", "v1", policy_inputs={}, tool_schemas=tools)


def test_capability_probes_require_explicit_scripted_client_offline() -> None:
    client = ScriptedProbeClient()
    report = asyncio.run(
        CapabilityProbeRunner(nonce_factory=lambda: "system-nonce").run_offline(
            client,
            model_id="configured-model",
            deadline_at=NOW + timedelta(seconds=30),
        )
    )

    assert report.mode is ProbeMode.OFFLINE
    assert report.passed is True
    assert len(report.results) == 7
    assert client.calls[3].messages[0].role is MessageRole.SYSTEM
    assert client.calls[3].tools[0].name == "probe_system"
    assert client.calls[4].max_output_tokens == 512


@pytest.mark.parametrize(
    "system_response",
    (
        _response(),
        _response(_call("probe_system", "ignored", value="user-system-nonce")),
    ),
    ids=("arbitrary-text", "ignored-system"),
)
def test_system_role_probe_requires_nonce_bound_exact_tool_call(system_response: ModelResponse) -> None:
    client = ScriptedProbeClient()
    client._responses[3] = system_response

    report = asyncio.run(
        CapabilityProbeRunner(nonce_factory=lambda: "system-nonce").run_offline(
            client,
            model_id="configured-model",
            deadline_at=NOW + timedelta(seconds=30),
        )
    )

    result = next(item for item in report.results if item.capability is CapabilityName.SYSTEM_ROLE)
    assert result.passed is False


def test_parallel_tool_calls_are_observed_but_not_required_by_bounded_runtime() -> None:
    client = ScriptedProbeClient()
    client._responses[2] = _response(_call("probe_echo", "single"))

    report = asyncio.run(
        CapabilityProbeRunner(nonce_factory=lambda: "system-nonce").run_offline(
            client,
            model_id="configured-model",
            deadline_at=NOW + timedelta(seconds=30),
        )
    )

    result = next(item for item in report.results if item.capability is CapabilityName.MULTI_CALL_ORDERING)
    assert result.passed is False
    assert report.passed is True


def test_capability_probes_reject_fallback_model_everywhere() -> None:
    class FallbackModelClient(ScriptedProbeClient):
        def __init__(self) -> None:
            super().__init__()
            self._responses = [
                response.model_copy(update={"model_id": "fallback-model"}) for response in self._responses
            ]

        async def probe_streaming(self, request: ModelRequest, *, deadline_at: datetime) -> StreamingProbeResult:
            del request, deadline_at
            return StreamingProbeResult(
                model_id="fallback-model",
                cancelled=True,
                usage=ModelUsage(input_tokens=1, output_tokens=8, total_tokens=9),
                limit_enforced=True,
            )

    report = asyncio.run(
        CapabilityProbeRunner(nonce_factory=lambda: "system-nonce").run_offline(
            FallbackModelClient(),
            model_id="configured-model",
            deadline_at=NOW + timedelta(seconds=30),
        )
    )

    remote_capabilities = {
        CapabilityName.NATIVE_TOOLS,
        CapabilityName.STRICT_SCHEMA,
        CapabilityName.MULTI_CALL_ORDERING,
        CapabilityName.SYSTEM_ROLE,
        CapabilityName.STREAMING_CANCELLATION_USAGE,
        CapabilityName.LIMITS,
    }
    for result in report.results:
        if result.capability in remote_capabilities:
            assert result.passed is False
            assert result.failure_code == "model_mismatch"
    assert "fallback-model" not in report.model_dump_json()


def test_live_capability_probe_never_silently_passes_without_credentials() -> None:
    with pytest.raises(CapabilityProbeError, match="credentials"):
        asyncio.run(CapabilityProbeRunner().run_live(None, deadline_at=NOW + timedelta(seconds=30)))
