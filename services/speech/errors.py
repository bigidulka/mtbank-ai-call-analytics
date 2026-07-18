"""Internal typed failures; HTTP layer maps them to stable DomainError codes."""

from __future__ import annotations


class SpeechServiceError(Exception):
    """Base class intentionally carries no public provider detail."""


class SpeechConfigurationError(SpeechServiceError):
    pass


class MediaValidationError(SpeechServiceError):
    pass


class UnsupportedMediaError(MediaValidationError):
    pass


class MediaTimeoutError(MediaValidationError):
    pass


class NoSpeechError(SpeechServiceError):
    pass


class SpeechDeadlineExceededError(SpeechServiceError):
    pass


class SpeechProviderError(SpeechServiceError):
    pass


class SpeechOverloadedError(SpeechServiceError):
    pass
