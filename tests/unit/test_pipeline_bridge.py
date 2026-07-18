from __future__ import annotations

import asyncio
import hashlib
import io
import threading
import time
import wave

import pytest

from mtbank_ai.pipeline_bridge import (
    MAIN_PIPELINE_ID,
    AttachmentDescriptor,
    AttachmentMetadataError,
    AttachmentReferenceError,
    create_signed_reference,
    normalise_file_descriptors,
    verify_signed_reference,
)
from pipeline import (
    _TRUSTED_OPENWEBUI_INTERNAL_URL,
    AuthoritativeFile,
    DownloadedFile,
    FileFetchError,
    OpenWebUIFileClient,
    _require_exact_openwebui_internal_url,
)
from pipeline import Pipeline as MainPipeline

SIGNING_KEY = "a" * 32
PIPELINES_API_KEY = "p" * 32
API_KEY = "N7!qR2@vL9#sX4$kM8%tY1^cD6&hJ3*F"
USER_ID = "user-123"
OTHER_USER_ID = "user-456"
FILE_ID = "f1d8f938-3c38-4f5f-a6d1-3c54e7cb5fc0"


def _wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8_000)
        output.writeframes(b"\x00\x00\x10\x00\xf0\xff\x00\x00" * 16)
    return buffer.getvalue()


WAV_BYTES = _wav_bytes()
WAV_HASH = hashlib.sha256(WAV_BYTES).hexdigest()


def _browser_file_item(*, file_id: str = FILE_ID, file_type: str = "file") -> dict[str, object]:
    return {
        "type": file_type,
        "id": file_id,
        "file": {
            "id": file_id,
            "filename": "client-controlled.wav",
            "meta": {
                "name": "client-controlled.wav",
                "content_type": "audio/wav",
                "size": len(WAV_BYTES),
                "file_hash": WAV_HASH,
                "url": "https://attacker.invalid/file.wav",
            },
        },
    }


def _signed_reference(
    *,
    subject: str = USER_ID,
    audience: str = MAIN_PIPELINE_ID,
    issued_at: int = 1_000,
    ttl_seconds: int = 300,
) -> dict[str, object]:
    return create_signed_reference(
        [_browser_file_item()],
        subject=subject,
        audience=audience,
        signing_key=SIGNING_KEY,
        issued_at=issued_at,
        ttl_seconds=ttl_seconds,
    )


def _fresh_signed_reference(*, subject: str = USER_ID) -> dict[str, object]:
    return _signed_reference(subject=subject, issued_at=int(time.time()))


def _pipeline_body(reference: dict[str, object], *, request_user_id: str = USER_ID) -> dict[str, object]:
    return {
        "user": {"id": request_user_id},
        "mtbank_attachment_user_id": USER_ID,
        "mtbank_attachment_ref": reference,
    }


def _set_pipeline_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MTBANK_ATTACHMENT_SIGNING_KEY", SIGNING_KEY)
    monkeypatch.setenv("MTBANK_API_KEY", API_KEY)
    monkeypatch.setenv("PIPELINES_API_KEY", PIPELINES_API_KEY)
    monkeypatch.setenv("WEBUI_ADMIN_EMAIL", "admin@example.test")
    monkeypatch.setenv("WEBUI_ADMIN_PASSWORD", "not-a-real-password")
    monkeypatch.delenv("MTBANK_PIPELINE_PROBE_MODE", raising=False)


def _authoritative_file(
    *,
    user_id: str = USER_ID,
    name: str = "authoritative-probe.wav",
    content_type: str = "audio/wav",
    size: int = len(WAV_BYTES),
    file_hash: str = WAV_HASH,
) -> AuthoritativeFile:
    return AuthoritativeFile(
        file_id=FILE_ID,
        user_id=user_id,
        name=name,
        content_type=content_type,
        size=size,
        file_hash=file_hash,
    )


def test_signed_reference_has_only_canonical_transport_claims() -> None:
    reference = _signed_reference()

    verified = verify_signed_reference(
        reference,
        expected_subject=USER_ID,
        expected_audience=MAIN_PIPELINE_ID,
        signing_key=SIGNING_KEY,
        now=1_001,
        max_ttl_seconds=300,
    )

    assert set(reference) == {"v", "aud", "sub", "file_id", "iat", "exp", "signature"}
    assert verified.file_id == FILE_ID
    assert verified.subject == USER_ID
    assert verified.audience == MAIN_PIPELINE_ID
    assert verified.issued_at == 1_000
    assert verified.expires_at == 1_300
    assert "name" not in reference
    assert "content_type" not in reference
    assert "file_hash" not in reference


def test_signed_reference_rejects_tampering() -> None:
    reference = _signed_reference()
    reference["file_id"] = "779d55ba-4766-4529-8267-3111d9a28619"

    with pytest.raises(AttachmentReferenceError, match="подпись"):
        verify_signed_reference(
            reference,
            expected_subject=USER_ID,
            expected_audience=MAIN_PIPELINE_ID,
            signing_key=SIGNING_KEY,
            now=1_001,
            max_ttl_seconds=300,
        )


def test_signed_reference_rejects_extra_client_hints() -> None:
    reference = _signed_reference()
    reference["name"] = "attacker.wav"

    with pytest.raises(AttachmentReferenceError, match="неподдерживаемые поля"):
        verify_signed_reference(
            reference,
            expected_subject=USER_ID,
            expected_audience=MAIN_PIPELINE_ID,
            signing_key=SIGNING_KEY,
            now=1_001,
            max_ttl_seconds=300,
        )


def test_signed_reference_rejects_wrong_audience() -> None:
    reference = _signed_reference(audience="another-pipeline")

    with pytest.raises(AttachmentReferenceError, match="audience"):
        verify_signed_reference(
            reference,
            expected_subject=USER_ID,
            expected_audience=MAIN_PIPELINE_ID,
            signing_key=SIGNING_KEY,
            now=1_001,
            max_ttl_seconds=300,
        )


@pytest.mark.parametrize(
    ("now", "future_skew", "expected_error"),
    [
        (1_300, 5, "просрочен"),
        (999, 0, "будущего"),
    ],
)
def test_signed_reference_rejects_expiry_and_future_skew(
    now: int,
    future_skew: int,
    expected_error: str,
) -> None:
    reference = _signed_reference()

    with pytest.raises(AttachmentReferenceError, match=expected_error):
        verify_signed_reference(
            reference,
            expected_subject=USER_ID,
            expected_audience=MAIN_PIPELINE_ID,
            signing_key=SIGNING_KEY,
            now=now,
            max_ttl_seconds=300,
            max_future_skew_seconds=future_skew,
        )


def test_normalises_real_browser_shape_without_copying_hints() -> None:
    descriptors = normalise_file_descriptors([_browser_file_item()])

    assert descriptors == [AttachmentDescriptor(file_id=FILE_ID)]


@pytest.mark.parametrize(
    "files",
    [
        [{"type": "file", "id": "../outside"}],
        [_browser_file_item(file_type="collection")],
        [_browser_file_item(), _browser_file_item()],
    ],
)
def test_rejects_noncanonical_or_non_direct_browser_items(files: object) -> None:
    with pytest.raises(AttachmentMetadataError):
        normalise_file_descriptors(files)


def test_single_pipeline_inlet_overwrites_client_mtbank_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MTBANK_ATTACHMENT_SIGNING_KEY", SIGNING_KEY)
    pipeline = MainPipeline()
    body = {
        "model": MAIN_PIPELINE_ID,
        "user": {"id": "client-controlled"},
        "mtbank_attachment_ref": {"client": "controlled"},
        "mtbank_attachment_user_id": "client-controlled",
        "mtbank_attachment_error": "client-controlled",
        "mtbank_extra": "client-controlled",
        "metadata": {"files": [_browser_file_item()]},
    }

    forwarded = asyncio.run(pipeline.inlet(body, {"id": USER_ID}))

    assert forwarded["metadata"] == body["metadata"]
    assert forwarded["user"] == {"id": "client-controlled"}
    assert forwarded["mtbank_attachment_user_id"] == USER_ID
    assert set(forwarded["mtbank_attachment_ref"]) == {"v", "aud", "sub", "file_id", "iat", "exp", "signature"}
    assert "mtbank_attachment_error" not in forwarded
    assert "mtbank_extra" not in forwarded
    verified = verify_signed_reference(
        forwarded["mtbank_attachment_ref"],
        expected_subject=USER_ID,
        expected_audience=MAIN_PIPELINE_ID,
        signing_key=SIGNING_KEY,
        now=forwarded["mtbank_attachment_ref"]["iat"],
        max_ttl_seconds=300,
    )
    assert verified.file_id == FILE_ID


def test_inlet_skips_auxiliary_tasks_and_marks_invalid_attachments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MTBANK_ATTACHMENT_SIGNING_KEY", SIGNING_KEY)
    pipeline = MainPipeline()

    skipped = asyncio.run(
        pipeline.inlet(
            {"metadata": {"task": "title_generation", "files": [_browser_file_item()]}},
            {"id": USER_ID},
        )
    )
    invalid = asyncio.run(
        pipeline.inlet(
            {"metadata": {"files": [{"type": "file", "id": "../bad"}]}},
            {"id": USER_ID},
        )
    )

    assert "mtbank_attachment_ref" not in skipped
    assert "mtbank_attachment_error" not in skipped
    assert invalid["mtbank_attachment_error"] == "invalid_attachment"


class _FakeFileClient:
    instances: list[_FakeFileClient] = []
    authoritative_file = _authoritative_file()
    downloaded_content = WAV_BYTES

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.get_file_calls: list[str] = []
        self.download_calls: list[tuple[str, int]] = []
        self.instances.append(self)

    @classmethod
    def reset(cls, authoritative_file: AuthoritativeFile | None = None, content: bytes = WAV_BYTES) -> None:
        cls.instances.clear()
        cls.authoritative_file = authoritative_file or _authoritative_file()
        cls.downloaded_content = content

    def get_file(self, file_id: str) -> AuthoritativeFile:
        self.get_file_calls.append(file_id)
        return self.authoritative_file

    def download(self, file_id: str, *, max_bytes: int) -> DownloadedFile:
        self.download_calls.append((file_id, max_bytes))
        return DownloadedFile(content=self.downloaded_content)


def test_main_pipeline_lifecycle_and_authoritative_audio_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_pipeline_environment(monkeypatch)
    monkeypatch.setenv("MTBANK_PIPELINE_PROBE_MODE", "true")
    _FakeFileClient.reset()
    pipeline = MainPipeline(client_factory=_FakeFileClient)

    asyncio.run(pipeline.on_startup())
    result = pipeline.pipe(
        user_message="",
        model_id=MAIN_PIPELINE_ID,
        messages=[],
        body=_pipeline_body(_fresh_signed_reference()),
    )

    assert "authoritative-probe.wav" in result
    assert "client-controlled.wav" not in result
    assert f"`{len(WAV_BYTES)}`" in result
    assert WAV_HASH in result
    assert _FakeFileClient.instances[-1].get_file_calls == [FILE_ID]
    assert _FakeFileClient.instances[-1].download_calls == [(FILE_ID, pipeline.valves.MAX_ATTACHMENT_BYTES)]

    pipeline.valves.DISPLAY_NAME = "Обновлённый Attachment Probe"
    asyncio.run(pipeline.on_valves_updated())
    assert pipeline.name == "Обновлённый Attachment Probe"
    assert len(_FakeFileClient.instances) == 2
    asyncio.run(pipeline.on_shutdown())
    assert pipeline._file_client is None


def test_pipeline_does_not_return_probe_output_without_explicit_test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_pipeline_environment(monkeypatch)
    _FakeFileClient.reset()
    pipeline = MainPipeline(client_factory=_FakeFileClient)
    asyncio.run(pipeline.on_startup())
    pipeline._analysis_adapter = None

    result = pipeline.pipe(
        user_message="",
        model_id=MAIN_PIPELINE_ID,
        messages=[],
        body=_pipeline_body(_fresh_signed_reference()),
    )

    assert "Анализ временно недоступен" in result
    assert WAV_HASH not in result
    assert "OpenWebUI API файла вернул" not in result


def test_pipeline_startup_requires_api_key_outside_explicit_probe_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_pipeline_environment(monkeypatch)
    monkeypatch.delenv("MTBANK_API_KEY")

    with pytest.raises(ValueError, match="MTBANK_API_KEY"):
        asyncio.run(MainPipeline(client_factory=_FakeFileClient).on_startup())


def test_explicit_probe_mode_is_the_only_mode_that_allows_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_pipeline_environment(monkeypatch)
    monkeypatch.setenv("MTBANK_PIPELINE_PROBE_MODE", "true")
    monkeypatch.delenv("MTBANK_API_KEY")
    _FakeFileClient.reset()
    pipeline = MainPipeline(client_factory=_FakeFileClient)

    asyncio.run(pipeline.on_startup())

    assert pipeline._analysis_adapter is None


def test_startup_applies_hydrated_persisted_display_name(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_pipeline_environment(monkeypatch)
    _FakeFileClient.reset()
    pipeline = MainPipeline(client_factory=_FakeFileClient)
    persisted_display_name = "Persisted Attachment Probe"

    pipeline.valves = pipeline.Valves(DISPLAY_NAME=persisted_display_name)
    assert pipeline.name != persisted_display_name

    asyncio.run(pipeline.on_startup())

    assert pipeline.name == persisted_display_name


def test_escapes_authoritative_filename_before_markdown_rendering(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_pipeline_environment(monkeypatch)
    monkeypatch.setenv("MTBANK_PIPELINE_PROBE_MODE", "true")
    malicious_name = '`</code>![x](https:attacker.invalid)&".wav'
    _FakeFileClient.reset(_authoritative_file(name=malicious_name))
    pipeline = MainPipeline(client_factory=_FakeFileClient)
    asyncio.run(pipeline.on_startup())

    result = pipeline.pipe(
        user_message="",
        model_id=MAIN_PIPELINE_ID,
        messages=[],
        body=_pipeline_body(_fresh_signed_reference()),
    )

    assert malicious_name not in result
    assert "![x]" not in result
    assert "<code>&#96;&lt;/code&gt;&#33;&#91;x&#93;&#40;https:attacker.invalid&#41;&amp;&quot;.wav</code>" in result


def test_owner_mismatch_rejects_without_content_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_pipeline_environment(monkeypatch)
    _FakeFileClient.reset(_authoritative_file(user_id=OTHER_USER_ID))
    pipeline = MainPipeline(client_factory=_FakeFileClient)
    asyncio.run(pipeline.on_startup())

    result = pipeline.pipe(
        user_message="",
        model_id=MAIN_PIPELINE_ID,
        messages=[],
        body=_pipeline_body(_fresh_signed_reference()),
    )

    assert "Вложение недоступно" in result
    assert _FakeFileClient.instances[-1].get_file_calls == [FILE_ID]
    assert _FakeFileClient.instances[-1].download_calls == []


def test_metadata_fetch_failure_matches_owner_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_pipeline_environment(monkeypatch)

    _FakeFileClient.reset(_authoritative_file(user_id=OTHER_USER_ID))
    owner_pipeline = MainPipeline(client_factory=_FakeFileClient)
    asyncio.run(owner_pipeline.on_startup())
    owner_result = owner_pipeline.pipe(
        user_message="",
        model_id=MAIN_PIPELINE_ID,
        messages=[],
        body=_pipeline_body(_fresh_signed_reference()),
    )

    class MissingFileClient(_FakeFileClient):
        def get_file(self, file_id: str) -> AuthoritativeFile:
            self.get_file_calls.append(file_id)
            raise FileFetchError("metadata недоступна")

    missing_pipeline = MainPipeline(client_factory=MissingFileClient)
    asyncio.run(missing_pipeline.on_startup())
    missing_result = missing_pipeline.pipe(
        user_message="",
        model_id=MAIN_PIPELINE_ID,
        messages=[],
        body=_pipeline_body(_fresh_signed_reference()),
    )

    assert missing_result == owner_result
    assert "Вложение недоступно" in missing_result


def test_forwarded_user_mismatch_rejects_before_metadata_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_pipeline_environment(monkeypatch)
    _FakeFileClient.reset()
    pipeline = MainPipeline(client_factory=_FakeFileClient)
    asyncio.run(pipeline.on_startup())

    result = pipeline.pipe(
        user_message="",
        model_id=MAIN_PIPELINE_ID,
        messages=[],
        body=_pipeline_body(_fresh_signed_reference(), request_user_id=OTHER_USER_ID),
    )

    assert "Вложение недоступно" in result
    assert _FakeFileClient.instances[-1].get_file_calls == []
    assert _FakeFileClient.instances[-1].download_calls == []


@pytest.mark.parametrize(
    ("authoritative_file", "downloaded_content"),
    [
        (_authoritative_file(), WAV_BYTES[:-1]),
        (_authoritative_file(file_hash="0" * 64), WAV_BYTES),
    ],
)
def test_authoritative_size_or_hash_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    authoritative_file: AuthoritativeFile,
    downloaded_content: bytes,
) -> None:
    _set_pipeline_environment(monkeypatch)
    _FakeFileClient.reset(authoritative_file, content=downloaded_content)
    pipeline = MainPipeline(client_factory=_FakeFileClient)
    asyncio.run(pipeline.on_startup())

    result = pipeline.pipe(
        user_message="",
        model_id=MAIN_PIPELINE_ID,
        messages=[],
        body=_pipeline_body(_fresh_signed_reference()),
    )

    assert "Не удалось проверить целостность" in result
    assert WAV_HASH not in result
    assert _FakeFileClient.instances[-1].download_calls == [(FILE_ID, pipeline.valves.MAX_ATTACHMENT_BYTES)]


def test_authoritative_audio_mime_and_magic_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_pipeline_environment(monkeypatch)
    pipeline = MainPipeline(client_factory=_FakeFileClient)

    _FakeFileClient.reset(_authoritative_file(content_type="application/octet-stream"))
    asyncio.run(pipeline.on_startup())
    unsupported_mime = pipeline.pipe(
        user_message="",
        model_id=MAIN_PIPELINE_ID,
        messages=[],
        body=_pipeline_body(_fresh_signed_reference()),
    )
    assert "не является поддерживаемым аудиофайлом" in unsupported_mime
    assert _FakeFileClient.instances[-1].download_calls == []

    _FakeFileClient.reset(
        _authoritative_file(size=len(b"not-a-wav"), file_hash=hashlib.sha256(b"not-a-wav").hexdigest()),
        content=b"not-a-wav",
    )
    asyncio.run(pipeline.on_valves_updated())
    invalid_magic = pipeline.pipe(
        user_message="",
        model_id=MAIN_PIPELINE_ID,
        messages=[],
        body=_pipeline_body(_fresh_signed_reference()),
    )
    assert "не является поддерживаемым аудиофайлом" in invalid_magic
    assert WAV_HASH not in invalid_magic


def test_hard_size_limit_rejects_before_content_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_pipeline_environment(monkeypatch)
    pipeline = MainPipeline(client_factory=_FakeFileClient)
    _FakeFileClient.reset(_authoritative_file(size=pipeline.valves.MAX_ATTACHMENT_BYTES + 1))
    asyncio.run(pipeline.on_startup())

    result = pipeline.pipe(
        user_message="",
        model_id=MAIN_PIPELINE_ID,
        messages=[],
        body=_pipeline_body(_fresh_signed_reference()),
    )

    assert "превышает допустимый размер" in result
    assert _FakeFileClient.instances[-1].download_calls == []


def test_main_pipeline_returns_controlled_messages_without_a_file() -> None:
    pipeline = MainPipeline()

    missing = pipeline.pipe(
        user_message="",
        model_id=MAIN_PIPELINE_ID,
        messages=[],
        body={"mtbank_attachment_error": "missing_attachment"},
    )
    invalid = pipeline.pipe(
        user_message="",
        model_id=MAIN_PIPELINE_ID,
        messages=[],
        body={"mtbank_attachment_error": "invalid_attachment"},
    )
    absent = pipeline.pipe(
        user_message="",
        model_id=MAIN_PIPELINE_ID,
        messages=[],
        body={"model": MAIN_PIPELINE_ID},
    )

    assert missing == invalid == absent
    assert "Вложение недоступно" in missing


def test_rejects_default_pipelines_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_pipeline_environment(monkeypatch)
    monkeypatch.setenv("PIPELINES_API_KEY", "changeme")

    with pytest.raises(ValueError, match="non-default"):
        asyncio.run(MainPipeline().on_startup())


def test_stale_jwt_rejection_does_not_clear_newer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenWebUIFileClient(
        base_url="http://openwebui:8080",
        email="admin@example.test",
        password="not-a-real-password",
        timeout_seconds=1,
    )
    client._token = "expired-token"
    sign_in_calls: list[str] = []

    def sign_in() -> str:
        sign_in_calls.append("fresh-token")
        return "fresh-token"

    monkeypatch.setattr(client, "_sign_in", sign_in)
    refreshed = threading.Event()

    def refresh_old_token() -> None:
        client._clear_token_if_matches("expired-token")
        assert client._get_token() == "fresh-token"
        refreshed.set()

    def reject_stale_token_late() -> None:
        assert refreshed.wait(timeout=1)
        client._clear_token_if_matches("expired-token")

    refresh_thread = threading.Thread(target=refresh_old_token)
    stale_thread = threading.Thread(target=reject_stale_token_late)
    refresh_thread.start()
    stale_thread.start()
    refresh_thread.join(timeout=1)
    stale_thread.join(timeout=1)

    assert not refresh_thread.is_alive()
    assert not stale_thread.is_alive()
    assert client._get_token() == "fresh-token"
    assert sign_in_calls == ["fresh-token"]


def test_openwebui_callback_is_fixed_outside_runtime_valves() -> None:
    pipeline = MainPipeline()

    assert not hasattr(pipeline.valves, "OPENWEBUI_INTERNAL_URL")
    assert _require_exact_openwebui_internal_url(_TRUSTED_OPENWEBUI_INTERNAL_URL) == "http://openwebui:8080"

    for untrusted_url in (
        "https://openwebui:8080",
        "http://openwebui:8080/redirect",
        "http://admin@openwebui:8080",
        "http://openwebui:8080?target=attacker.invalid",
        "http://attacker.invalid:8080",
    ):
        with pytest.raises(ValueError, match="callback"):
            _require_exact_openwebui_internal_url(untrusted_url)
