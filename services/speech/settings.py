"""Typed runtime configuration for the isolated speech service."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Literal, Self

from pydantic import HttpUrl, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from mtbank_ai.domain.base import Confidence, NonEmptyId, NonNegativeInt, PositiveFloat, PositiveInt, StrictFrozenModel
from mtbank_ai.runtime_secrets import SecretConfigurationError, require_runtime_secret


class FasterWhisperSettings(StrictFrozenModel):
    """Local CTranslate2 faster-whisper settings for the canonical batch path."""

    provider: Literal["faster-whisper"] = "faster-whisper"
    model_id: Literal["dropbox-dash/faster-whisper-large-v3-turbo"] = "dropbox-dash/faster-whisper-large-v3-turbo"
    language: Literal["ru"] = "ru"
    beam_size: PositiveInt = 5
    cpu_compute_type: Literal["int8"] = "int8"
    cuda_compute_type: Literal["float16"] = "float16"
    cpu_threads: PositiveInt = 8
    vad_filter: bool = False

    def compute_type(self, *, device: str) -> str:
        if device == "cpu":
            return self.cpu_compute_type
        if device == "cuda":
            return self.cuda_compute_type
        raise ValueError("unsupported faster-whisper device")


class GroqTranscriptionSettings(StrictFrozenModel):
    """Opt-in Groq provider used only by provisional rolling WebSocket updates."""

    provider: Literal["groq"] = "groq"
    api_key: SecretStr
    endpoint: HttpUrl = HttpUrl("https://api.groq.com/openai/v1/audio/transcriptions")
    model: Literal["whisper-large-v3-turbo"] = "whisper-large-v3-turbo"
    language: Literal["ru"] = "ru"
    temperature: Literal[0] = 0
    response_format: Literal["verbose_json"] = "verbose_json"
    request_timeout_seconds: PositiveFloat = 180.0
    connect_timeout_seconds: PositiveFloat = 10.0
    max_response_bytes: PositiveInt = 4 * 1024 * 1024

    @field_validator("api_key")
    @classmethod
    def require_nonempty_secret(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("Groq API key is required")
        return value

    @model_validator(mode="after")
    def validate_endpoint(self) -> Self:
        if (
            self.endpoint.scheme != "https"
            or self.endpoint.host != "api.groq.com"
            or self.endpoint.path != "/openai/v1/audio/transcriptions"
            or self.endpoint.query is not None
            or self.endpoint.fragment is not None
            or self.endpoint.username is not None
            or self.endpoint.password is not None
        ):
            raise ValueError("Groq endpoint must be the configured HTTPS transcription endpoint")
        if self.connect_timeout_seconds > self.request_timeout_seconds:
            raise ValueError("Groq connect timeout cannot exceed request timeout")
        return self

    @property
    def endpoint_fingerprint(self) -> str:
        return hashlib.sha256(str(self.endpoint).encode("utf-8")).hexdigest()


class SpeechRuntimeSettings(StrictFrozenModel):
    device: Literal["cpu", "cuda"] = "cpu"
    image_digest: str | None = None
    language: Literal["ru"] = "ru"
    pipeline_revision: NonEmptyId = "speech/local-faster-whisper-v1"
    role_review_confidence_threshold: Confidence = 0.75
    normalization_sample_rate_hz: PositiveInt = 16_000
    normalization_channels: Literal[1] = 1
    normalization_codec: Literal["pcm_s16le"] = "pcm_s16le"
    max_upload_bytes: PositiveInt = 25 * 1024 * 1024
    max_duration_seconds: PositiveFloat = 300.0
    ffmpeg_timeout_seconds: PositiveFloat = 30.0
    request_timeout_seconds: PositiveFloat = 180.0
    max_concurrency: PositiveInt = 1
    queue_capacity: NonNegativeInt = 2
    temp_root: NonEmptyId = "/tmp/mtbank-speech"

    @field_validator("image_digest")
    @classmethod
    def validate_image_digest(cls, value: str | None) -> str | None:
        if value is not None and (not re.fullmatch(r"sha256:[0-9a-f]{64}", value) or value != value.strip()):
            raise ValueError("image_digest должен быть immutable sha256 digest")
        return value


class SpeechStreamingSettings(StrictFrozenModel):
    """Bounded opt-in Groq rolling transcription settings for provisional WebSocket updates."""

    enabled: bool = False
    max_frame_bytes: PositiveInt = 64 * 1024
    max_session_bytes: PositiveInt = 10 * 1024 * 1024
    max_update_text_bytes: PositiveInt = 48 * 1024
    max_duration_seconds: PositiveFloat = 300.0
    processing_timeout_seconds: PositiveFloat = 2.0
    ffmpeg_timeout_seconds: PositiveFloat = 5.0
    max_decoder_output_bytes: PositiveInt = 10 * 1024 * 1024
    max_decoder_stderr_bytes: PositiveInt = 64 * 1024
    rolling_window_seconds: PositiveFloat = 8.0
    rolling_step_seconds: PositiveFloat = 1.5
    rolling_call_timeout_seconds: PositiveFloat = 1.5
    max_rolling_calls_per_session: PositiveInt = 100
    max_rolling_audio_seconds_per_session: PositiveFloat = 180.0
    max_concurrent_rolling_calls: PositiveInt = 1
    pcm_energy_threshold: NonNegativeInt = 16

    @model_validator(mode="after")
    def validate_limits(self) -> Self:
        required_pcm_bytes = int(self.max_duration_seconds * 16_000 * 2)
        if self.max_session_bytes < self.max_frame_bytes:
            raise ValueError("streaming max_session_bytes не может быть меньше max_frame_bytes")
        if self.max_session_bytes < required_pcm_bytes:
            raise ValueError("streaming max_session_bytes не покрывает максимальную PCM duration")
        if self.max_update_text_bytes > self.max_frame_bytes:
            raise ValueError("streaming max_update_text_bytes не может превышать max_frame_bytes")
        if self.max_decoder_output_bytes < required_pcm_bytes:
            raise ValueError("streaming max_decoder_output_bytes не покрывает максимальную PCM duration")
        if self.rolling_step_seconds > self.rolling_window_seconds:
            raise ValueError("streaming rolling_step_seconds не может превышать streaming rolling_window_seconds")
        if self.rolling_call_timeout_seconds > self.processing_timeout_seconds:
            raise ValueError("rolling call timeout не может превышать streaming processing timeout")
        if self.max_rolling_audio_seconds_per_session > self.max_duration_seconds:
            raise ValueError("streaming rolling audio budget не может превышать max duration")
        return self


class SpeechModelSettings(StrictFrozenModel):
    manifest_path: NonEmptyId = "/models/manifest.json"
    artifact_root: NonEmptyId = "/models/artifacts"


class SpeechAccessSettings(StrictFrozenModel):
    """Optional bearer boundary for a remotely exposed single speech container."""

    mode: Literal["internal", "bearer"]
    bearer_key: SecretStr | None = None

    @model_validator(mode="after")
    def validate_access_mode(self) -> Self:
        if self.mode == "internal":
            if self.bearer_key is not None:
                raise ValueError("internal speech access must not configure a bearer key")
            return self
        if self.bearer_key is None:
            raise ValueError("bearer speech access requires a bearer key")
        bearer_key = self.bearer_key.get_secret_value()
        if not bearer_key.isascii():
            raise ValueError("bearer speech access key must contain only ASCII characters")
        try:
            require_runtime_secret("MTBANK_SPEECH__ACCESS__BEARER_KEY", bearer_key)
        except SecretConfigurationError as error:
            raise ValueError("bearer speech access key is unsafe") from error
        return self


class SpeechSettings(BaseSettings):
    """Canonical batch needs only local artifacts; Groq is optional streaming configuration."""

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=None,
        env_ignore_empty=True,
        env_nested_delimiter="__",
        env_prefix="MTBANK_SPEECH__",
        extra="ignore",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )

    runtime: SpeechRuntimeSettings = SpeechRuntimeSettings()
    faster_whisper: FasterWhisperSettings = FasterWhisperSettings()
    groq: GroqTranscriptionSettings | None = None
    streaming: SpeechStreamingSettings = SpeechStreamingSettings()
    models: SpeechModelSettings = SpeechModelSettings()
    access: SpeechAccessSettings

    @model_validator(mode="after")
    def validate_profile(self) -> Self:
        if self.runtime.max_concurrency != 1:
            raise ValueError("один speech process обслуживает ровно один canonical worker")
        if not Path(self.runtime.temp_root).is_absolute():
            raise ValueError("temp_root должен быть абсолютным путём")
        if not Path(self.models.manifest_path).is_absolute() or not Path(self.models.artifact_root).is_absolute():
            raise ValueError("пути model manifest и artifacts должны быть абсолютными")
        if self.runtime.language != self.faster_whisper.language:
            raise ValueError("canonical speech language must match local faster-whisper language")
        if self.streaming.enabled:
            if self.groq is None:
                raise ValueError("Groq configuration is required only when streaming is enabled")
            if self.runtime.language != self.groq.language:
                raise ValueError("streaming Groq language must match canonical language")
        return self

    @property
    def temp_root(self) -> Path:
        return Path(self.runtime.temp_root)

    @property
    def manifest_path(self) -> Path:
        return Path(self.models.manifest_path)

    @property
    def artifact_root(self) -> Path:
        return Path(self.models.artifact_root)
