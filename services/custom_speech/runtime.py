"""Bounded OpenAI-compatible ASR + semantic-role + VAD runtime."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self
from uuid import NAMESPACE_URL, uuid5

import httpx
from pydantic import Field, ValidationError, model_validator

from mtbank_ai.domain.base import StrictFrozenModel
from mtbank_ai.domain.provenance import ComponentRevision
from mtbank_ai.domain.transcript import (
    ASRMetadata,
    ASRProviderMetadata,
    RoleAgentProvenance,
    RoleAssignment,
    RoleResolution,
    RoleResolutionSource,
    SpeakerRole,
    TranscriptSegment,
    TranscriptSnapshot,
)
from mtbank_ai.speech.contracts import SpeechFile, SpeechTranscriptionResponse
from services.custom_speech.settings import CustomSpeechSettings
from services.speech.errors import NoSpeechError, SpeechOverloadedError, SpeechProviderError
from services.speech.media import MediaLimits, MediaNormalizer, NormalizedAudio

_PROMPT_ROOT = Path(__file__).resolve().parents[2] / "src" / "mtbank_ai" / "agents"
_TOOL_NAME = "submit_flat_turns"
_EVENT = re.compile(r"silence_(start|end):\s*([0-9.]+)")
_MAX_VAD_OUTPUT_BYTES = 64 * 1024
_MAX_ASR_RESPONSE_BYTES = 4 * 1024 * 1024
_MAX_ROLE_RESPONSE_BYTES = 2 * 1024 * 1024
# Attestation provenance must name the backend that actually produced the text; anything
# the endpoint allowlist admits beyond the two official providers is the internal bridge.
_PROVIDER_BY_HOST = {"api.openai.com": "openai", "api.groq.com": "groq"}


class FlatAsrResult(StrictFrozenModel):
    text: str = Field(min_length=1, max_length=20_000)
    provider_metadata: ASRProviderMetadata


class OpenAiFlatTranscriber:
    """Exactly one bounded allowlisted audio transcription request."""

    def __init__(
        self,
        settings: CustomSpeechSettings,
        *,
        client_factory: Callable[..., httpx.Client] = httpx.Client,
    ) -> None:
        self._settings = settings
        self._client_factory = client_factory

    def transcribe(self, audio: NormalizedAudio, *, language: str) -> FlatAsrResult:
        if language != "ru":
            raise SpeechProviderError("custom ASR language must be Russian")
        try:
            with (
                audio.path.open("rb") as source,
                self._client_factory(
                    timeout=httpx.Timeout(120.0, connect=10.0),
                    trust_env=False,
                    follow_redirects=False,
                ) as client,
                client.stream(
                    "POST",
                    str(self._settings.asr_endpoint),
                    files=(
                        ("file", (audio.path.name, source, "audio/wav")),
                        ("model", (None, self._settings.asr_model)),
                        ("language", (None, language)),
                        ("temperature", (None, "0")),
                        ("response_format", (None, "json")),
                    ),
                    headers={
                        "Authorization": f"Bearer {self._settings.asr_api_key.get_secret_value()}",
                        "Accept-Encoding": "identity",
                    },
                ) as response,
            ):
                content = _bounded_response(response, _MAX_ASR_RESPONSE_BYTES)
                if not 200 <= response.status_code < 300:
                    raise SpeechProviderError("custom ASR request failed")
                payload = json.loads(content)
                text = payload.get("text") if isinstance(payload, dict) else None
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("custom ASR response has no text")
                provider = _PROVIDER_BY_HOST.get(self._settings.asr_endpoint.host or "", "chatgpt-bridge")
                return FlatAsrResult(
                    text=text.strip(),
                    provider_metadata=ASRProviderMetadata(
                        provider=provider,
                        model=self._settings.asr_model,
                        endpoint_fingerprint=self._settings.asr_endpoint_fingerprint,
                        request_id=_request_id(response),
                        usage_seconds=None,
                    ),
                )
        except SpeechProviderError:
            raise
        except (OSError, ValueError, json.JSONDecodeError, httpx.HTTPError) as error:
            raise SpeechProviderError("custom ASR request failed") from error


class FlatTurn(StrictFrozenModel):
    start_word_index: int = Field(ge=0, le=1_200)
    end_word_index: int = Field(ge=0, le=1_200)
    role: Literal["Оператор", "Клиент"]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.start_word_index > self.end_word_index:
            raise ValueError("semantic turn range is reversed")
        return self


class FlatDecision(StrictFrozenModel):
    turns: tuple[FlatTurn, ...] = Field(min_length=1, max_length=1_200)


class VadAnchor(StrictFrozenModel):
    start: float = Field(ge=0.0)
    end: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.start >= self.end:
            raise ValueError("VAD anchor range is reversed")
        return self


class SemanticTurnResolver:
    """One bounded typed call that partitions every ASR word exactly once."""

    def __init__(
        self,
        settings: CustomSpeechSettings,
        *,
        client_factory: Callable[..., httpx.Client] = httpx.Client,
    ) -> None:
        self._settings = settings
        self._client_factory = client_factory
        self._tool_schema = FlatDecision.model_json_schema()
        prompt_path = _PROMPT_ROOT / "flat_transcript_reconstructor" / "v1.md"
        self._prompt_text = prompt_path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        if not self._prompt_text.strip():
            raise SpeechProviderError("reviewed semantic prompt is unavailable")
        self._prompt_hash = hashlib.sha256(
            json.dumps(
                {"prompt": self._prompt_text, "tool_schema": self._tool_schema},
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()

    @property
    def provenance(self) -> RoleAgentProvenance:
        return RoleAgentProvenance(
            policy_id="flat_transcript_reconstructor",
            version="v1",
            owner="MTBank AI Engineering",
            effective_date="2026-08-04",
            sha256=self._prompt_hash,
        )

    def resolve(self, words: Sequence[str]) -> tuple[FlatTurn, ...]:
        if not words or len(words) > self._settings.max_words:
            raise SpeechProviderError("flat transcript exceeds semantic turn bound")
        payload = json.dumps(
            {"words": [{"index": index, "word": word} for index, word in enumerate(words)]},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        request_payload = {
            "model": self._settings.role_model,
            "messages": (
                {"role": "system", "content": self._prompt_text},
                {"role": "user", "content": payload},
            ),
            "tools": (
                {
                    "type": "function",
                    "function": {
                        "name": _TOOL_NAME,
                        "description": "Submit contiguous role turns covering every flat transcript word exactly once.",
                        "parameters": self._tool_schema,
                        "strict": True,
                    },
                },
            ),
            "tool_choice": {"type": "function", "function": {"name": _TOOL_NAME}},
            "temperature": 0,
            "max_tokens": 8_000,
        }
        # This leg is the only one that can fail without the audio being at fault: gateways
        # return transient 5xx, and even models that honour the strict schema most of the time
        # occasionally emit a malformed or partial turn list. A single attempt turns each of
        # those into a failed transcription, so retry a bounded number of times. The request is
        # deterministic (temperature 0) and has no side effects, so replaying it is safe.
        last_error: Exception | None = None
        for attempt in range(self._settings.role_max_attempts):
            try:
                return self._resolve_once(request_payload, word_count=len(words))
            except (
                SpeechProviderError,
                httpx.HTTPError,
                TypeError,
                ValueError,
                ValidationError,
                json.JSONDecodeError,
            ) as error:
                last_error = error
                if attempt + 1 < self._settings.role_max_attempts:
                    time.sleep(min(2.0, 0.25 * (2**attempt)))
        raise SpeechProviderError("semantic role request failed") from last_error

    def _resolve_once(self, request_payload: dict[str, object], *, word_count: int) -> tuple[FlatTurn, ...]:
        with self._client_factory(
            timeout=httpx.Timeout(
                self._settings.role_timeout_seconds,
                connect=min(5.0, self._settings.role_timeout_seconds),
            ),
            trust_env=False,
            follow_redirects=False,
        ) as client:
            with client.stream(
                "POST",
                f"{str(self._settings.role_base_url).rstrip('/')}/chat/completions",
                json=request_payload,
                headers={
                    "Authorization": f"Bearer {self._settings.role_api_key.get_secret_value()}",
                    "Accept-Encoding": "identity",
                },
            ) as response:
                content = _bounded_response(response, _MAX_ROLE_RESPONSE_BYTES)
                if not 200 <= response.status_code < 300:
                    raise SpeechProviderError("semantic role request failed")
        parsed = json.loads(content)
        decision = _flat_decision(parsed, expected_model=self._settings.role_model)
        _validate_turn_coverage(decision.turns, word_count)
        return _merge_same_role(decision.turns)

    def close(self) -> None:
        return None


class SemanticVadSpeechRuntime:
    """Experimental no-GPU runtime preserving canonical speech response contract."""

    def __init__(
        self,
        settings: CustomSpeechSettings,
        *,
        transcriber: OpenAiFlatTranscriber | None = None,
        resolver: SemanticTurnResolver | None = None,
    ) -> None:
        self._settings = settings
        runtime = settings.speech_runtime()
        self._normalizer = MediaNormalizer(
            MediaLimits(
                max_upload_bytes=runtime.max_upload_bytes,
                max_duration_seconds=runtime.max_duration_seconds,
                process_timeout_seconds=runtime.ffmpeg_timeout_seconds,
                temp_root=Path(runtime.temp_root),
                sample_rate_hz=runtime.normalization_sample_rate_hz,
                channels=runtime.normalization_channels,
                codec=runtime.normalization_codec,
            )
        )
        self._transcriber = transcriber or OpenAiFlatTranscriber(settings)
        self._resolver = resolver or SemanticTurnResolver(settings)
        self._slot = asyncio.Semaphore(1)
        self._admission_lock = asyncio.Lock()
        self._outstanding = 0

    async def ready(self) -> bool:
        return True

    def model_revisions(self) -> tuple[ComponentRevision, ComponentRevision]:
        return _asr_revision(self._settings), _diarization_revision(self._settings)

    def runtime_attestation(self) -> dict[str, object]:
        return {
            "runtime": {
                "profile": "experimental_no_gpu",
                "device": "cpu",
                "asr": _component_json(_asr_revision(self._settings)),
                "alignment": _component_json(_alignment_revision()),
                "diarization": _component_json(_diarization_revision(self._settings)),
                "role_model": self._settings.role_model,
            }
        }

    async def transcribe(self, source: SpeechFile) -> SpeechTranscriptionResponse:
        await self._reserve()
        try:
            async with self._slot:
                async with asyncio.timeout(180.0):
                    return await asyncio.to_thread(self._transcribe_sync, source)
        except TimeoutError as error:
            raise SpeechProviderError("custom speech request exceeded deadline") from error
        finally:
            await self._release()

    async def close(self) -> None:
        self._resolver.close()

    def _transcribe_sync(self, source: SpeechFile) -> SpeechTranscriptionResponse:
        started = time.perf_counter()
        with self._normalizer.normalize(source) as audio:
            transcription = self._transcriber.transcribe(audio, language="ru")
            words = tuple(transcription.text.split())
            if not words:
                raise NoSpeechError("custom ASR returned no words")
            turns = self._resolver.resolve(words)
            anchors = _vad_anchors(audio, self._settings)
            aligned = _align_turns(turns, words, anchors, audio.duration_seconds)
            return SpeechTranscriptionResponse(
                transcript=_snapshot(
                    source=source,
                    audio=audio,
                    turns=turns,
                    aligned=aligned,
                    provider=transcription.provider_metadata,
                    role_provenance=self._resolver.provenance,
                    pipeline_revision=self._settings.pipeline_revision,
                    processing_ms=max(0, round((time.perf_counter() - started) * 1_000)),
                    settings=self._settings,
                )
            )

    async def _reserve(self) -> None:
        async with self._admission_lock:
            if self._outstanding >= 3:
                raise SpeechOverloadedError("custom speech queue is full")
            self._outstanding += 1

    async def _release(self) -> None:
        async with self._admission_lock:
            self._outstanding -= 1


class _AlignedTurn(StrictFrozenModel):
    start: float = Field(ge=0.0)
    end: float = Field(gt=0.0)
    role: Literal["Оператор", "Клиент"]
    confidence: float = Field(ge=0.0, le=1.0)
    text: str = Field(min_length=1, max_length=20_000)


def _validate_turn_coverage(turns: Sequence[FlatTurn], word_count: int) -> None:
    expected = 0
    for turn in turns:
        if turn.start_word_index != expected:
            raise ValueError("semantic turns have gap, overlap, or reorder")
        expected = turn.end_word_index + 1
    if expected != word_count:
        raise ValueError("semantic turns do not cover all words")


def _merge_same_role(turns: Sequence[FlatTurn]) -> tuple[FlatTurn, ...]:
    merged: list[FlatTurn] = []
    for turn in turns:
        if merged and merged[-1].role == turn.role:
            previous = merged[-1]
            merged[-1] = FlatTurn(
                start_word_index=previous.start_word_index,
                end_word_index=turn.end_word_index,
                role=previous.role,
                confidence=min(previous.confidence, turn.confidence),
                evidence="adjacent same-role semantic turns merged",
            )
        else:
            merged.append(turn)
    return tuple(merged)


def _vad_anchors(audio: NormalizedAudio, settings: CustomSpeechSettings) -> tuple[VadAnchor, ...]:
    stderr_path = audio.path.parent / ".vad.stderr"
    with stderr_path.open("xb") as stderr:
        process = subprocess.Popen(
            (
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-nostats",
                "-i",
                str(audio.path),
                "-af",
                f"silencedetect=noise={settings.vad_noise_db}dB:d={settings.vad_minimum_silence_seconds}",
                "-f",
                "null",
                "-",
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=stderr,
            close_fds=True,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=30.0)
        except subprocess.TimeoutExpired as error:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise SpeechProviderError("VAD process exceeded deadline") from error
    if return_code != 0 or stderr_path.stat().st_size > _MAX_VAD_OUTPUT_BYTES:
        raise SpeechProviderError("VAD process failed")
    events = _EVENT.findall(stderr_path.read_text(encoding="utf-8", errors="replace"))
    silences: list[tuple[float, float]] = []
    pending: float | None = None
    for kind, raw_value in events:
        value = min(max(float(raw_value), 0.0), audio.duration_seconds)
        if kind == "start":
            pending = value
        elif pending is not None:
            silences.append((pending, value))
            pending = None
    if pending is not None:
        silences.append((pending, audio.duration_seconds))
    anchors: list[VadAnchor] = []
    cursor = 0.0
    for start, end in silences:
        start = min(max(start, cursor), audio.duration_seconds)
        end = min(max(end, start), audio.duration_seconds)
        if start - cursor >= 0.03:
            anchors.append(VadAnchor(start=cursor, end=start))
        cursor = max(cursor, end)
    if audio.duration_seconds - cursor >= 0.03:
        anchors.append(VadAnchor(start=cursor, end=audio.duration_seconds))
    if not anchors:
        raise NoSpeechError("VAD produced no speech anchors")
    return tuple(anchors)


def _align_turns(
    turns: tuple[FlatTurn, ...],
    words: tuple[str, ...],
    anchors: tuple[VadAnchor, ...],
    duration: float,
) -> tuple[_AlignedTurn, ...]:
    if len(turns) > len(anchors):
        raise SpeechProviderError("semantic turns exceed VAD anchor topology")
    gaps = tuple(right.start - left.end for left, right in zip(anchors, anchors[1:]))
    boundaries = sorted(sorted(range(len(gaps)), key=lambda index: (-gaps[index], index))[: len(turns) - 1])
    groups: list[tuple[int, int]] = []
    left = 0
    for boundary in boundaries:
        groups.append((left, boundary))
        left = boundary + 1
    groups.append((left, len(anchors) - 1))
    for start_padding, end_padding in ((0.2, 0.9), (0.1, 0.2), (0.0, 0.0)):
        aligned = tuple(
            _AlignedTurn(
                start=max(0.0, anchors[first].start - start_padding),
                end=min(duration, anchors[last].end + end_padding),
                role=turn.role,
                confidence=turn.confidence,
                text=" ".join(words[turn.start_word_index : turn.end_word_index + 1]),
            )
            for turn, (first, last) in zip(turns, groups, strict=True)
        )
        if not any(current.start < previous.end for previous, current in zip(aligned, aligned[1:])):
            return aligned
    raise SpeechProviderError("VAD semantic alignment overlaps after bounded padding reduction")


def _snapshot(
    *,
    source: SpeechFile,
    audio: NormalizedAudio,
    turns: tuple[FlatTurn, ...],
    aligned: tuple[_AlignedTurn, ...],
    provider: ASRProviderMetadata | None,
    role_provenance: RoleAgentProvenance,
    pipeline_revision: str,
    processing_ms: int,
    settings: CustomSpeechSettings,
) -> TranscriptSnapshot:
    segments = tuple(
        TranscriptSegment(
            id=uuid5(NAMESPACE_URL, f"{audio.audio_sha256}/{pipeline_revision}/{index}"),
            original_speaker_id=_speaker_id(item.role),
            speaker=SpeakerRole(item.role),
            role_confidence=item.confidence,
            speaker_confidence=item.confidence,
            start=item.start,
            end=item.end,
            text=item.text,
            redacted_text=item.text,
            word_timestamps=(),
        )
        for index, item in enumerate(aligned)
    )
    assignments = tuple(
        RoleAssignment(
            original_speaker_id=_speaker_id(role.value),
            role=role,
            confidence=min(turn.confidence for turn in turns if turn.role == role.value),
            evidence_segment_ids=tuple(segment.id for segment in segments if segment.speaker is role),
            source=RoleResolutionSource.AGENT,
            resolution_evidence="bounded semantic turn reconstruction",
        )
        for role in (SpeakerRole.OPERATOR, SpeakerRole.CLIENT)
        if any(turn.role == role.value for turn in turns)
    )
    return TranscriptSnapshot(
        transcript_id=uuid5(NAMESPACE_URL, f"mtbank-ai/transcript/{audio.audio_sha256}/{pipeline_revision}"),
        audio_sha256=audio.audio_sha256,
        revision=pipeline_revision,
        language="ru",
        duration_seconds=audio.duration_seconds,
        segments=segments,
        role_resolution=RoleResolution(
            assignments=assignments,
            needs_review=any(assignment.confidence < 0.75 for assignment in assignments),
            agent_provenance=role_provenance,
        ),
        asr_metadata=ASRMetadata(
            asr=_asr_revision(settings),
            alignment=_alignment_revision(),
            diarization=_diarization_revision(settings),
            language="ru",
            processing_ms=processing_ms,
            provider=provider,
        ),
        created_at=datetime.now(UTC),
    )


def _speaker_id(role: str) -> str:
    return "semantic-operator" if role == SpeakerRole.OPERATOR.value else "semantic-client"


def _asr_revision(settings: CustomSpeechSettings) -> ComponentRevision:
    return ComponentRevision(
        package="openai-compatible-audio-api",
        package_version="v1",
        model_id=settings.asr_model,
        model_revision="provider-managed",
    )


def _alignment_revision() -> ComponentRevision:
    return ComponentRevision(
        package="mtbank-ai",
        package_version="1",
        model_id="ffmpeg-silencedetect-ranked-gap",
        model_revision="adaptive-padding-v1",
    )


def _diarization_revision(settings: CustomSpeechSettings) -> ComponentRevision:
    return ComponentRevision(
        package="mtbank-ai",
        package_version="1",
        model_id=f"semantic-vad/{settings.role_model}",
        model_revision="flat-transcript-reconstructor-v1",
    )


def _flat_decision(payload: object, *, expected_model: str) -> FlatDecision:
    if not isinstance(payload, dict) or payload.get("model") != expected_model:
        raise ValueError("semantic role response model drift")
    choices = payload.get("choices")
    usage = payload.get("usage")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(usage, dict):
        raise ValueError("semantic role response envelope is invalid")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict) or message.get("content") not in {None, ""}:
        raise ValueError("semantic role response contains unauthorized text")
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise ValueError("semantic role response requires one tool call")
    call = calls[0]
    function = call.get("function") if isinstance(call, dict) else None
    if not isinstance(function, dict) or function.get("name") != _TOOL_NAME:
        raise ValueError("semantic role response called unauthorized tool")
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        raise ValueError("semantic role tool arguments are invalid")
    return FlatDecision.model_validate_json(arguments, strict=True)


def _bounded_response(response: httpx.Response, maximum_bytes: int) -> bytes:
    if response.headers.get("content-encoding", "identity").strip().casefold() != "identity":
        raise SpeechProviderError("custom ASR response uses unsupported content encoding")
    content_length = response.headers.get("content-length")
    if content_length is not None and (not content_length.isdecimal() or int(content_length) > maximum_bytes):
        raise SpeechProviderError("custom ASR response exceeded bound")
    content = bytearray()
    for chunk in response.iter_raw():
        if len(chunk) > maximum_bytes - len(content):
            raise SpeechProviderError("custom ASR response exceeded bound")
        content.extend(chunk)
    return bytes(content)


def _request_id(response: httpx.Response) -> str | None:
    for header in ("x-request-id", "request-id"):
        value = response.headers.get(header)
        if value is not None and value.strip():
            return value.strip()[:256]
    return None


def _component_json(component: ComponentRevision) -> dict[str, str]:
    return {
        "package": component.package,
        "package_version": component.package_version,
        "model_id": component.model_id,
        "model_revision": component.model_revision,
    }
