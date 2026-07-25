from __future__ import annotations

import asyncio
from typing import cast
from uuid import UUID

import pytest

from mtbank_ai.speech.contracts import RoleResolutionCandidate, SpeechFile
from mtbank_ai.speech.roles import RoleResolutionRequiredError
from services.speech.engine import CanonicalBatchEngine
from services.speech.errors import SpeechConfigurationError
from services.speech.role_agent import LlmRoleResolver
from services.speech.runtime import LazySpeechRuntime
from tests.unit.speech._helpers import make_registry


def test_runtime_loads_llm_role_agent_by_default_and_preserves_explicit_none(tmp_path) -> None:
    _, settings = make_registry(tmp_path)
    received: list[object] = []

    def factory(registry, runtime, faster_whisper, resolver):
        del registry, runtime, faster_whisper
        received.append(resolver)
        return cast(CanonicalBatchEngine, object())

    async def scenario() -> None:
        default_runtime = LazySpeechRuntime(settings, engine_factory=factory)
        assert await default_runtime.ready()
        await default_runtime._get_engine()
        assert isinstance(received[-1], LlmRoleResolver)

        explicit_none_runtime = LazySpeechRuntime(settings, engine_factory=factory, role_resolver=None)
        assert await explicit_none_runtime.ready()
        await explicit_none_runtime._get_engine()
        assert received[-1] is None

    asyncio.run(scenario())


def test_runtime_fails_closed_when_role_agent_configuration_is_unavailable(tmp_path) -> None:
    _, settings = make_registry(tmp_path)
    invalid_settings = settings.model_copy(update={"role_agent": None})

    with pytest.raises(SpeechConfigurationError, match="role agent configuration"):
        LazySpeechRuntime(invalid_settings)


def test_runtime_explicit_none_keeps_role_resolution_fail_closed_and_ready(tmp_path) -> None:
    _, settings = make_registry(tmp_path)

    class FailClosedEngine:
        def transcribe(self, source: SpeechFile) -> object:
            del source
            raise RoleResolutionRequiredError(
                (
                    RoleResolutionCandidate(
                        original_speaker_id="speaker-unknown",
                        evidence_segment_ids=(UUID("11111111-1111-4111-8111-111111111111"),),
                    ),
                )
            )

    def factory(registry, runtime, faster_whisper, resolver):
        del registry, runtime, faster_whisper
        assert resolver is None
        return cast(CanonicalBatchEngine, FailClosedEngine())

    async def scenario() -> None:
        runtime = LazySpeechRuntime(settings, engine_factory=factory, role_resolver=None)
        assert await runtime.ready()
        with pytest.raises(RoleResolutionRequiredError):
            await runtime.transcribe(SpeechFile("call.wav", "audio/wav", b"RIFF"))
        assert await runtime.ready()

    asyncio.run(scenario())
