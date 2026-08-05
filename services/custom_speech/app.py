"""FastAPI factory for experimental no-GPU semantic VAD speech service."""

from __future__ import annotations

from fastapi import FastAPI

from services.custom_speech.runtime import SemanticVadSpeechRuntime
from services.custom_speech.settings import CustomSpeechSettings
from services.speech.app import create_app as create_speech_app
from services.speech.settings import FasterWhisperSettings, SpeechAccessSettings, SpeechSettings


def create_app() -> FastAPI:
    custom = CustomSpeechSettings.model_validate({})
    envelope = SpeechSettings(
        runtime=custom.speech_runtime(),
        faster_whisper=FasterWhisperSettings(),
        groq=None,
        role_agent=None,
        access=SpeechAccessSettings(mode="internal"),
    )
    app = create_speech_app(envelope, SemanticVadSpeechRuntime(custom))
    app.title = "MTBank Experimental No-GPU Speech Service"
    return app
