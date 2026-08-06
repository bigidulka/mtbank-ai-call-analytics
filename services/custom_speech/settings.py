"""Typed settings for experimental OpenAI ASR + semantic VAD backend."""

from __future__ import annotations

import hashlib
from typing import Literal, Self

from pydantic import HttpUrl, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from mtbank_ai.config import _is_internal_service_name
from mtbank_ai.domain.base import NonEmptyId, PositiveFloat, PositiveInt
from mtbank_ai.runtime_secrets import SecretConfigurationError, require_runtime_secret
from services.speech.settings import RoleAgentSettings, SpeechRuntimeSettings

_OFFICIAL_ASR_ENDPOINTS = {
    ("api.openai.com", "/v1/audio/transcriptions"),
    ("api.groq.com", "/openai/v1/audio/transcriptions"),
}


def _is_allowed_asr_endpoint(url: HttpUrl) -> bool:
    """Official providers require HTTPS; an internal bridge may use HTTP only on the Docker network."""

    if (url.host, url.path) in _OFFICIAL_ASR_ENDPOINTS:
        return url.scheme == "https"
    return url.scheme == "http" and url.path == "/v1/audio/transcriptions" and _is_internal_service_name(url.host or "")


class CustomSpeechSettings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=None,
        env_ignore_empty=True,
        env_nested_delimiter="__",
        env_prefix="MTBANK_CUSTOM_SPEECH__",
        extra="ignore",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )

    asr_api_key: SecretStr
    asr_endpoint: HttpUrl = HttpUrl("https://api.openai.com/v1/audio/transcriptions")
    asr_model: NonEmptyId = "gpt-4o-transcribe"
    role_base_url: HttpUrl
    role_api_key: SecretStr
    role_model: NonEmptyId = "gpt-5.6-sol"
    role_timeout_seconds: PositiveFloat = 30.0
    role_max_attempts: PositiveInt = 3
    max_words: PositiveInt = 1_200
    # Swept against the real-call references: -50 dB scored 73.16% mean role accuracy with a
    # 1.92 pp spread across min-silence values, where the previous -45 dB scored 69.59% and
    # swung 5.22 pp. A single -40 dB/0.40 s point scored higher still (76.16%) but its
    # neighbours sat near 68%, so it is a spike on a two-call sample, not a setting to adopt.
    vad_noise_db: float = -50.0
    vad_minimum_silence_seconds: PositiveFloat = 0.25
    # v2 marks word-proportional alignment. Transcripts from the two revisions place turn
    # boundaries differently, so the revision has to distinguish them for provenance.
    pipeline_revision: Literal[
        "speech/openai-semantic-vad-v1",
        "speech/chatgpt-bridge-semantic-vad-v1",
        "speech/openai-semantic-vad-v2",
        "speech/chatgpt-bridge-semantic-vad-v2",
    ] = "speech/openai-semantic-vad-v2"

    @field_validator("asr_api_key", "role_api_key")
    @classmethod
    def validate_secret(cls, value: SecretStr) -> SecretStr:
        secret = value.get_secret_value()
        if not secret.isascii() or any(character.isspace() for character in secret):
            raise ValueError("provider key is invalid")
        try:
            require_runtime_secret("MTBANK_CUSTOM_SPEECH__PROVIDER_KEY", secret)
        except SecretConfigurationError as error:
            raise ValueError("provider key is invalid") from error
        return value

    @model_validator(mode="after")
    def validate_profile(self) -> Self:
        if (
            not _is_allowed_asr_endpoint(self.asr_endpoint)
            or self.asr_endpoint.query is not None
            or self.asr_endpoint.fragment is not None
            or self.asr_endpoint.username is not None
            or self.asr_endpoint.password is not None
        ):
            raise ValueError("ASR endpoint must be an allowlisted HTTPS provider or internal HTTP bridge")
        if (
            self.role_base_url.scheme != "https"
            or self.role_base_url.query is not None
            or self.role_base_url.fragment is not None
        ):
            raise ValueError("role gateway must use HTTPS without query or fragment")
        if self.max_words > 1_200:
            raise ValueError("semantic word bound cannot exceed tool schema")
        # Retries are bounded so a failing gateway cannot hold a transcription request open
        # for role_max_attempts * role_timeout_seconds beyond the caller's own deadline.
        if self.role_max_attempts > 5:
            raise ValueError("role attempts cannot exceed 5")
        if not -100.0 <= self.vad_noise_db <= 0.0:
            raise ValueError("VAD noise threshold must be between -100 and 0 dB")
        return self

    def speech_runtime(self) -> SpeechRuntimeSettings:
        return SpeechRuntimeSettings(
            device="cpu",
            language="ru",
            pipeline_revision=self.pipeline_revision,
            request_timeout_seconds=180.0,
            max_concurrency=1,
            queue_capacity=2,
            temp_root="/tmp/mtbank-custom-speech",
        )

    def role_agent(self) -> RoleAgentSettings:
        return RoleAgentSettings(
            base_url=self.role_base_url,
            api_key=self.role_api_key,
            model=self.role_model,
            timeout_seconds=self.role_timeout_seconds,
            connect_timeout_seconds=min(5.0, self.role_timeout_seconds),
            max_output_tokens=8_000,
            max_input_chars=60_000,
            max_candidates=2,
        )

    @property
    def asr_endpoint_fingerprint(self) -> str:
        return hashlib.sha256(str(self.asr_endpoint).encode()).hexdigest()
