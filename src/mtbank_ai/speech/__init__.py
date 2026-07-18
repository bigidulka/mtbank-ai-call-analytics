"""Контракты internal-only canonical speech pipeline."""

from mtbank_ai.speech.client import HttpSpeechServiceClient, SpeechServiceClientSettings, SpeechTranscriptionPort
from mtbank_ai.speech.contracts import (
    DiarizedSegment,
    RoleResolutionDecision,
    RoleSegmentEvidence,
    SpeechFile,
    SpeechMetadata,
    SpeechTranscriptionResponse,
)
from mtbank_ai.speech.roles import PolicyRoleResolver, RoleResolutionRequiredError, RoleResolverPort

__all__ = [
    "DiarizedSegment",
    "HttpSpeechServiceClient",
    "PolicyRoleResolver",
    "RoleResolutionDecision",
    "RoleResolutionRequiredError",
    "RoleSegmentEvidence",
    "RoleResolverPort",
    "SpeechFile",
    "SpeechMetadata",
    "SpeechServiceClientSettings",
    "SpeechTranscriptionPort",
    "SpeechTranscriptionResponse",
]
