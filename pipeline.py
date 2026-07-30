"""Закреплённый OpenWebUI Pipeline для безопасной передачи verified audio в Analyze API."""

from __future__ import annotations

import hashlib
import hmac
import html
import json
import os
import re
import threading
from collections.abc import Callable, Generator, Mapping
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from mtbank_ai.workflow.pipeline_adapter import PipelineAnalysisPort

from pydantic import BaseModel, Field, ValidationError

from mtbank_ai.application.ports import FileAnalyzeInput
from mtbank_ai.domain.analysis import AnalyzeResponse
from mtbank_ai.domain.errors import DomainError, ErrorCode
from mtbank_ai.pipeline_bridge import (
    MAIN_PIPELINE_ID,
    AttachmentBridgeError,
    AttachmentMetadataError,
    AttachmentReferenceError,
    VerifiedAttachmentReference,
    create_signed_reference,
    extract_user_id,
    require_signing_key,
    verify_signed_reference,
)
from mtbank_ai.runtime_secrets import SecretConfigurationError, require_runtime_secret
from mtbank_ai.trusted_http import TrustedHttpError, build_trusted_opener, require_exact_base_url

_HARD_MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024
_TRUSTED_OPENWEBUI_INTERNAL_URL = "http://openwebui:8080"
_TRUSTED_ANALYSIS_API_INTERNAL_URL = "http://api:8000"
_JSON_RESPONSE_MAX_BYTES = 1 * 1024 * 1024
_TEXT_ASSISTANT_MAX_HISTORY_MESSAGES = 8
_TEXT_ASSISTANT_MAX_MESSAGE_CHARS = 2_000
_TEXT_ASSISTANT_MAX_STREAM_BYTES = 64 * 1024
_TEXT_ASSISTANT_MAX_STREAM_EVENTS = 1_024
_ATTACHMENT_UNAVAILABLE_MESSAGE = "Вложение недоступно. Загрузите файл заново и повторите запрос."
_TEXT_ASSISTANT_UNAVAILABLE_MESSAGE = "Текстовый помощник временно недоступен. Повторите запрос позже."
_TEXT_ASSISTANT_MARKER = "text_assistant"
_MARKDOWN_FILENAME_TRANSLATION = str.maketrans(
    {
        "\\": "&#92;",
        "`": "&#96;",
        "!": "&#33;",
        "[": "&#91;",
        "]": "&#93;",
        "(": "&#40;",
        ")": "&#41;",
    }
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_UNSAFE_PIPELINES_API_KEYS = frozenset(
    {
        "changeme",
        "default",
        "pipeline-api-key",
        "pipelines-api-key",
        "secret",
        "test",
        "your-api-key",
        "your-pipelines-api-key",
    }
)
_AUDIO_MIME_TYPES = {
    "wav": frozenset({"audio/wav", "audio/wave", "audio/x-wav", "audio/vnd.wave"}),
    "mpeg": frozenset({"audio/mpeg", "audio/mp3"}),
    "ogg": frozenset({"audio/ogg", "audio/opus"}),
    "webm": frozenset({"audio/webm"}),
}
_T = TypeVar("_T")


class FileFetchError(RuntimeError):
    """Внутренний OpenWebUI file contract недоступен или некорректен."""


class FileOwnershipError(FileFetchError):
    """Authoritative владелец файла не совпадает с пользователем чата."""


class FileIntegrityError(FileFetchError):
    """Скачанные bytes не совпадают с authoritative metadata."""


class FileMediaError(FileFetchError):
    """Файл не является поддерживаемым аудио по MIME или magic bytes."""


class FileSizeError(FileFetchError):
    """Authoritative размер файла превышает жёсткий лимит Phase 0."""


class _UnauthorisedFileFetch(FileFetchError):
    """Кешированный OpenWebUI JWT больше не принимается."""


class DownloadedFile:
    """Bytes, полученные от authenticated file-content endpoint."""

    def __init__(self, content: bytes) -> None:
        self.content = content


class VerifiedAudio:
    """Вложение после authoritative ownership, MIME, magic и hash проверок."""

    def __init__(self, *, name: str, content_type: str, content: bytes) -> None:
        self.name = name
        self.content_type = content_type
        self.content = content

    @property
    def size(self) -> int:
        return len(self.content)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()


class AuthoritativeFile:
    """Поля FileModel, обязательные для owner, audio и integrity проверок."""

    def __init__(
        self,
        *,
        file_id: str,
        user_id: str,
        name: str,
        content_type: str,
        size: int,
        file_hash: str,
    ) -> None:
        self.file_id = file_id
        self.user_id = user_id
        self.name = name
        self.content_type = content_type
        self.size = size
        self.file_hash = file_hash


class FileClient(Protocol):
    """Минимальный authenticated OpenWebUI client, нужный Pipeline."""

    def get_file(self, file_id: str) -> AuthoritativeFile:
        """Читает authoritative FileModel до доступа к bytes."""
        ...

    def download(self, file_id: str, *, max_bytes: int) -> DownloadedFile:
        """Скачивает ограниченные bytes только после owner проверки."""
        ...


class OpenWebUIFileClient:
    """Лениво аутентифицируется во внутреннем OpenWebUI и читает файлы."""

    def __init__(
        self,
        *,
        base_url: str,
        email: str,
        password: str,
        timeout_seconds: int,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self._base_url = _require_exact_openwebui_internal_url(base_url)
        self._email = _required_setting(email, "WEBUI_ADMIN_EMAIL")
        self._password = _required_setting(password, "WEBUI_ADMIN_PASSWORD")
        self._timeout_seconds = timeout_seconds
        self._opener = opener or build_trusted_opener(self._base_url)
        self._token: str | None = None
        self._token_lock = threading.Lock()

    def get_file(self, file_id: str) -> AuthoritativeFile:
        """Читает FileModel и один раз обновляет отклонённый JWT."""

        return self._with_fresh_token(lambda token: self._get_file_with_token(file_id, token))

    def download(self, file_id: str, *, max_bytes: int) -> DownloadedFile:
        """Читает content с лимитом bytes и одним обновлением JWT."""

        if max_bytes <= 0 or max_bytes > _HARD_MAX_ATTACHMENT_BYTES:
            raise FileFetchError("лимит bytes вложения некорректен")
        return self._with_fresh_token(lambda token: self._download_with_token(file_id, token, max_bytes=max_bytes))

    def _with_fresh_token(self, operation: Callable[[str], _T]) -> _T:
        rejected_token = self._get_token()
        try:
            return operation(rejected_token)
        except _UnauthorisedFileFetch:
            self._clear_token_if_matches(rejected_token)
            return operation(self._get_token())

    def _get_token(self) -> str:
        with self._token_lock:
            if self._token is None:
                self._token = self._sign_in()
            return self._token

    def _clear_token_if_matches(self, rejected_token: str) -> None:
        """Не удаляет JWT, который уже обновил конкурентный запрос."""

        with self._token_lock:
            if self._token is not None and hmac.compare_digest(self._token, rejected_token):
                self._token = None

    def _sign_in(self) -> str:
        request = Request(
            self._url("/api/v1/auths/signin"),
            data=json.dumps({"email": self._email, "password": self._password}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with self._opener(request, timeout=self._timeout_seconds) as response:
                payload = _read_json_response(response)
        except (HTTPError, URLError, OSError, TrustedHttpError, ValueError, UnicodeDecodeError) as error:
            raise FileFetchError("не удалось аутентифицироваться во внутреннем OpenWebUI") from error

        token = payload.get("token") if isinstance(payload, Mapping) else None
        if not isinstance(token, str) or not token:
            raise FileFetchError("OpenWebUI не вернул JWT")
        return token

    def _get_file_with_token(self, file_id: str, token: str) -> AuthoritativeFile:
        request = Request(
            self._url(f"/api/v1/files/{file_id}"),
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            method="GET",
        )
        try:
            with self._opener(request, timeout=self._timeout_seconds) as response:
                return _parse_authoritative_file(_read_json_response(response), expected_file_id=file_id)
        except HTTPError as error:
            if error.code in (401, 403):
                raise _UnauthorisedFileFetch("OpenWebUI отклонил JWT") from error
            raise FileFetchError("не удалось прочитать metadata вложения из OpenWebUI") from error
        except (URLError, OSError, TrustedHttpError, ValueError, UnicodeDecodeError) as error:
            raise FileFetchError("не удалось прочитать metadata вложения из OpenWebUI") from error

    def _download_with_token(self, file_id: str, token: str, *, max_bytes: int) -> DownloadedFile:
        request = Request(
            self._url(f"/api/v1/files/{file_id}/content"),
            headers={"Authorization": f"Bearer {token}"},
            method="GET",
        )
        try:
            with self._opener(request, timeout=self._timeout_seconds) as response:
                content_length = response.headers.get("Content-Length")
                if content_length is not None:
                    declared_length = int(content_length)
                    if declared_length < 0 or declared_length > max_bytes:
                        raise FileSizeError("вложение превышает жёсткий лимит bytes")
                return DownloadedFile(content=_read_bounded(response, max_bytes))
        except HTTPError as error:
            if error.code in (401, 403):
                raise _UnauthorisedFileFetch("OpenWebUI отклонил JWT") from error
            raise FileFetchError("не удалось скачать вложение из OpenWebUI") from error
        except (URLError, OSError, TrustedHttpError, ValueError) as error:
            raise FileFetchError("не удалось скачать вложение из OpenWebUI") from error

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"


class ApiAnalysisClient:
    """Статический internal HTTP adapter к тому же REST AnalyzeCallUseCase."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: int,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self._base_url = _require_exact_analysis_api_url(base_url)
        try:
            self._api_key = require_runtime_secret("MTBANK_API_KEY", api_key)
        except SecretConfigurationError as error:
            raise ValueError("MTBANK_API_KEY отсутствует или небезопасен") from error
        self._timeout_seconds = timeout_seconds
        self._opener = opener or build_trusted_opener(self._base_url)

    def analyze(self, source: FileAnalyzeInput, *, request_id: UUID) -> AnalyzeResponse:
        body, boundary = _encode_audio_multipart(source)
        request = Request(
            f"{self._base_url}/analyze",
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Length": str(len(body)),
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "X-Request-ID": str(request_id),
            },
            method="POST",
        )
        try:
            with self._opener(request, timeout=self._timeout_seconds) as response:
                return AnalyzeResponse.model_validate(_read_json_response(response))
        except HTTPError as error:
            try:
                raise _map_analysis_api_status(error.code) from None
            finally:
                error.close()
        except (URLError, OSError, TrustedHttpError, UnicodeDecodeError, ValidationError, ValueError):
            raise DomainError(ErrorCode.SERVICE_UNAVAILABLE) from None


class ApiAssistantClient:
    """Pinned internal client for the real text-only LLM assistant."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: int,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self._base_url = _require_exact_analysis_api_url(base_url)
        self._api_key = _require_analysis_api_key(api_key)
        self._timeout_seconds = timeout_seconds
        self._opener = opener or build_trusted_opener(self._base_url)

    def stream(self, message: str, history: list[dict[str, str]]) -> Generator[str, None, None]:
        payload = json.dumps(
            {"message": message[:_TEXT_ASSISTANT_MAX_MESSAGE_CHARS], "history": history},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        request = Request(
            f"{self._base_url}/assistant/stream",
            data=payload,
            headers={
                "Accept": "text/event-stream",
                "Authorization": f"Bearer {self._api_key}",
                "Content-Length": str(len(payload)),
                "Content-Type": "application/json",
            },
            method="POST",
        )
        emitted = False
        try:
            with self._opener(request, timeout=self._timeout_seconds) as response:
                for text in self._parse_stream(response):
                    emitted = True
                    yield text
        except (HTTPError, URLError, OSError, TrustedHttpError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
            if not emitted:
                yield _TEXT_ASSISTANT_UNAVAILABLE_MESSAGE
            else:
                raise

    @staticmethod
    def _parse_stream(response: Any) -> Generator[str, None, None]:
        expected_sequence = 1
        event_name: str | None = None
        data_line: str | None = None
        total_bytes = 0
        event_count = 0
        answer_chars = 0
        saw_start = False
        saw_done = False

        for raw_line in _bounded_response_lines(response, limit=_JSON_RESPONSE_MAX_BYTES):
            if not isinstance(raw_line, bytes):
                raise ValueError("assistant SSE line is invalid")
            total_bytes += len(raw_line)
            if total_bytes > _TEXT_ASSISTANT_MAX_STREAM_BYTES:
                raise ValueError("assistant SSE stream exceeded bound")
            line = raw_line.decode("utf-8").rstrip("\r\n")
            if line.startswith("id: "):
                try:
                    event_id = int(line.removeprefix("id: "))
                except ValueError:
                    raise ValueError("assistant SSE ID is invalid") from None
                if event_id != expected_sequence:
                    raise ValueError("assistant SSE ID is invalid")
            elif line.startswith("event: "):
                event_name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data_line = line.removeprefix("data: ")
            elif not line:
                if event_name is None or data_line is None:
                    raise ValueError("assistant SSE event is incomplete")
                event_count += 1
                if event_count > _TEXT_ASSISTANT_MAX_STREAM_EVENTS:
                    raise ValueError("assistant SSE event count exceeded bound")
                parsed = json.loads(data_line)
                if not isinstance(parsed, Mapping) or parsed.get("v") != 1:
                    raise ValueError("assistant SSE event is invalid")
                sequence = parsed.get("sequence")
                if type(sequence) is not int or sequence != expected_sequence:
                    raise ValueError("assistant SSE sequence is invalid")
                expected_sequence += 1
                if event_name == "start":
                    if saw_start or event_count != 1 or set(parsed) != {"v", "sequence"}:
                        raise ValueError("assistant SSE start is invalid")
                    saw_start = True
                elif event_name == "progress":
                    _validate_assistant_progress(parsed, saw_start=saw_start, saw_done=saw_done)
                elif event_name == "delta":
                    if not saw_start or saw_done or set(parsed) != {"v", "sequence", "text"}:
                        raise ValueError("assistant SSE delta is invalid")
                    text = parsed.get("text")
                    if not isinstance(text, str) or not text or len(text) > 8_000:
                        raise ValueError("assistant SSE delta is invalid")
                    answer_chars += len(text)
                    if answer_chars > 8_000:
                        raise ValueError("assistant SSE answer exceeded bound")
                    yield text
                elif event_name == "done":
                    if not saw_start or saw_done or set(parsed) != {"v", "sequence"}:
                        raise ValueError("assistant SSE done is invalid")
                    saw_done = True
                elif event_name == "error":
                    _validate_assistant_error(parsed, saw_start=saw_start, saw_done=saw_done)
                    raise ValueError("assistant SSE failed")
                else:
                    raise ValueError("assistant SSE event type is invalid")
                event_name = None
                data_line = None
            elif not line.startswith(":"):
                raise ValueError("assistant SSE field is invalid")
        if event_name is not None or data_line is not None or not saw_start or not saw_done or answer_chars <= 0:
            raise ValueError("assistant SSE stream ended before done")

    def answer(self, message: str, history: list[dict[str, str]]) -> str:
        payload = json.dumps(
            {"message": message[:_TEXT_ASSISTANT_MAX_MESSAGE_CHARS], "history": history},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        request = Request(
            f"{self._base_url}/assistant",
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Length": str(len(payload)),
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with self._opener(request, timeout=self._timeout_seconds) as response:
                parsed = _read_json_response(response)
        except (HTTPError, URLError, OSError, TrustedHttpError, UnicodeDecodeError, ValueError):
            return _TEXT_ASSISTANT_UNAVAILABLE_MESSAGE
        answer = parsed.get("answer") if isinstance(parsed, Mapping) else None
        if not isinstance(answer, str) or not answer.strip() or len(answer) > 8_000:
            return _TEXT_ASSISTANT_UNAVAILABLE_MESSAGE
        return answer.strip()


def _bounded_response_lines(response: Any, *, limit: int) -> Generator[bytes, None, None]:
    readline = getattr(response, "readline", None)
    if callable(readline):
        while True:
            line = readline(limit + 1)
            if not isinstance(line, bytes):
                raise ValueError("assistant SSE line is invalid")
            if not line:
                return
            if len(line) > limit:
                raise ValueError("assistant SSE line is invalid")
            yield line
        return
    for line in response:
        if not isinstance(line, bytes) or len(line) > limit:
            raise ValueError("assistant SSE line is invalid")
        yield line


def _validate_assistant_progress(payload: Mapping[str, object], *, saw_start: bool, saw_done: bool) -> None:
    if not saw_start or saw_done:
        raise ValueError("assistant SSE progress is out of order")
    phase = payload.get("phase")
    if phase == "model":
        if set(payload) != {"v", "sequence", "phase"}:
            raise ValueError("assistant SSE model progress is invalid")
        return
    if phase == "tool":
        tool_name = payload.get("tool_name")
        if set(payload) != {"v", "sequence", "phase", "tool_name"} or tool_name not in {
            "demo_capabilities",
            "runtime_metadata",
            "trends_query",
        }:
            raise ValueError("assistant SSE tool progress is invalid")
        return
    raise ValueError("assistant SSE progress phase is invalid")


def _validate_assistant_error(payload: Mapping[str, object], *, saw_start: bool, saw_done: bool) -> None:
    if not saw_start or saw_done or set(payload) != {"v", "sequence", "code", "message", "retryable"}:
        raise ValueError("assistant SSE error is invalid")
    if (
        not isinstance(payload.get("code"), str)
        or not isinstance(payload.get("message"), str)
        or type(payload.get("retryable")) is not bool
    ):
        raise ValueError("assistant SSE error fields are invalid")


class Pipeline:
    """Единый Pipeline для inlet transport, проверки bytes и internal Analyze API."""

    class Valves(BaseModel):
        ATTACHMENT_REF_TTL_SECONDS: int = Field(default=300, ge=1, le=3600)
        MAX_ATTACHMENT_BYTES: int = Field(default=_HARD_MAX_ATTACHMENT_BYTES, ge=1, le=_HARD_MAX_ATTACHMENT_BYTES)
        HTTP_TIMEOUT_SECONDS: int = Field(default=240, ge=1, le=240)
        DISPLAY_NAME: str = Field(default="MTBank Attachment Probe", min_length=1, max_length=80)

    def __init__(
        self,
        client_factory: Callable[..., FileClient] = OpenWebUIFileClient,
        *,
        analysis_adapter: PipelineAnalysisPort | None = None,
        assistant_client: ApiAssistantClient | None = None,
    ) -> None:
        self.id = MAIN_PIPELINE_ID
        self.valves = self.Valves()
        self.name = self.valves.DISPLAY_NAME
        self._client_factory = client_factory
        self._injected_analysis_adapter = analysis_adapter
        self._analysis_adapter = analysis_adapter
        self._assistant_client = assistant_client
        self._file_client: FileClient | None = None
        self._signing_key: bytes | None = None

    async def on_startup(self) -> None:
        self._validate_configuration()
        self._signing_key = require_signing_key(os.getenv("MTBANK_ATTACHMENT_SIGNING_KEY"))
        self._file_client = self._create_file_client()
        self._analysis_adapter = self._create_analysis_adapter()
        if not _pipeline_probe_mode_enabled():
            self._assistant_client = self._assistant_client or self._create_assistant_client()
        self.name = self.valves.DISPLAY_NAME

    async def on_shutdown(self) -> None:
        self._file_client = None
        self._analysis_adapter = self._injected_analysis_adapter
        self._assistant_client = None
        self._signing_key = None

    async def on_valves_updated(self) -> None:
        self._validate_configuration()
        self._signing_key = require_signing_key(os.getenv("MTBANK_ATTACHMENT_SIGNING_KEY"))
        self._file_client = self._create_file_client()
        self._analysis_adapter = self._create_analysis_adapter()
        if not _pipeline_probe_mode_enabled():
            self._assistant_client = self._assistant_client or self._create_assistant_client()
        self.name = self.valves.DISPLAY_NAME

    async def inlet(self, body: dict[str, Any], user: dict[str, Any] | None = None) -> dict[str, Any]:
        """Подписывает единственный UUID до удаления metadata OpenWebUI."""

        clean_body = {key: value for key, value in body.items() if not key.startswith("mtbank_")}
        try:
            user_id = extract_user_id(user)
        except AttachmentBridgeError:
            clean_body["mtbank_attachment_error"] = "invalid_user"
            return clean_body

        metadata = clean_body.get("metadata")
        if not isinstance(metadata, Mapping):
            clean_body["mtbank_text_assistant"] = _TEXT_ASSISTANT_MARKER
            return clean_body
        if _is_auxiliary_task(metadata):
            return clean_body
        if metadata.get("files") is None:
            clean_body["mtbank_text_assistant"] = _TEXT_ASSISTANT_MARKER
            return clean_body

        try:
            reference = create_signed_reference(
                metadata.get("files"),
                subject=user_id,
                audience=MAIN_PIPELINE_ID,
                signing_key=self._signing_key or require_signing_key(os.getenv("MTBANK_ATTACHMENT_SIGNING_KEY")),
                ttl_seconds=self.valves.ATTACHMENT_REF_TTL_SECONDS,
            )
        except AttachmentBridgeError:
            clean_body["mtbank_attachment_error"] = "invalid_attachment"
            return clean_body

        clean_body["mtbank_attachment_user_id"] = user_id
        clean_body["mtbank_attachment_ref"] = reference
        return clean_body

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: list[dict[str, Any]],
        body: dict[str, Any],
    ) -> str | Generator[str, None, None]:
        """Возвращает обычный text/SSE-compatible ответ для проверенного аудио."""

        del model_id
        if body.get("mtbank_text_assistant") == _TEXT_ASSISTANT_MARKER:
            client = self._assistant_client or self._create_assistant_client()
            stream = getattr(client, "stream", None)
            if callable(stream):
                return cast(Generator[str, None, None], stream(user_message, _bounded_assistant_history(messages)))
            return client.answer(user_message, _bounded_assistant_history(messages))
        if body.get("mtbank_attachment_error") or "mtbank_attachment_ref" not in body:
            return _controlled_message(self.name, _ATTACHMENT_UNAVAILABLE_MESSAGE)

        try:
            request_user_id, reference = self._verify_attachment_reference(body)
        except AttachmentBridgeError:
            return _controlled_message(self.name, _ATTACHMENT_UNAVAILABLE_MESSAGE)

        try:
            client = self._file_client or self._create_file_client()
            authoritative_file = self._verify_direct_ownership(client, reference, request_user_id)
            verified_audio = self._load_verified_audio(client, authoritative_file)
        except FileSizeError:
            return _controlled_message(self.name, "Аудиофайл превышает допустимый размер.")
        except FileMediaError:
            return _controlled_message(self.name, "Вложение не является поддерживаемым аудиофайлом.")
        except FileIntegrityError:
            return _controlled_message(
                self.name,
                "Не удалось проверить целостность вложения. Загрузите файл заново и повторите запрос.",
            )
        except FileFetchError:
            return _controlled_message(self.name, _ATTACHMENT_UNAVAILABLE_MESSAGE)

        if _pipeline_probe_mode_enabled():
            return _render_probe(
                [(verified_audio.name, verified_audio.size, verified_audio.sha256)],
                display_name=self.name,
            )
        if self._analysis_adapter is None:
            return _controlled_message(self.name, "Анализ временно недоступен. Повторите запрос позже.")
        try:
            return self._render_analysis(verified_audio)
        except Exception as error:
            return _controlled_analysis_error(self.name, error)

    def _verify_attachment_reference(self, body: dict[str, Any]) -> tuple[str, VerifiedAttachmentReference]:
        try:
            forwarded_user_id = extract_user_id(_as_user_mapping(body.get("user")))
            attachment_user_id = _normalise_user_id(body.get("mtbank_attachment_user_id"))
        except AttachmentBridgeError as error:
            raise AttachmentReferenceError("привязка пользователя вложения некорректна") from error

        if attachment_user_id != forwarded_user_id:
            raise AttachmentReferenceError("пользователь вложения не совпадает с пользователем запроса")

        reference = verify_signed_reference(
            body.get("mtbank_attachment_ref"),
            expected_subject=attachment_user_id,
            expected_audience=MAIN_PIPELINE_ID,
            signing_key=self._signing_key or require_signing_key(os.getenv("MTBANK_ATTACHMENT_SIGNING_KEY")),
            max_ttl_seconds=self.valves.ATTACHMENT_REF_TTL_SECONDS,
        )
        return forwarded_user_id, reference

    def _verify_direct_ownership(
        self,
        client: FileClient,
        reference: VerifiedAttachmentReference,
        request_user_id: str,
    ) -> AuthoritativeFile:
        authoritative_file = client.get_file(reference.file_id)
        if authoritative_file.user_id != reference.subject or authoritative_file.user_id != request_user_id:
            raise FileOwnershipError("authoritative владелец не совпадает с signed пользователем")
        if authoritative_file.size > self.valves.MAX_ATTACHMENT_BYTES:
            raise FileSizeError("authoritative размер превышает жёсткий лимит")
        _audio_format_for_content_type(authoritative_file.content_type)
        return authoritative_file

    def _load_verified_audio(self, client: FileClient, authoritative_file: AuthoritativeFile) -> VerifiedAudio:
        downloaded = client.download(
            authoritative_file.file_id,
            max_bytes=self.valves.MAX_ATTACHMENT_BYTES,
        )
        audio_format = _audio_format_for_content_type(authoritative_file.content_type)
        if not _has_expected_magic(downloaded.content, audio_format):
            raise FileMediaError("magic bytes не соответствуют authoritative MIME")

        actual_size = len(downloaded.content)
        actual_hash = hashlib.sha256(downloaded.content).hexdigest()
        if actual_size != authoritative_file.size or not hmac.compare_digest(actual_hash, authoritative_file.file_hash):
            raise FileIntegrityError("bytes вложения отличаются от authoritative metadata")
        return VerifiedAudio(
            name=authoritative_file.name,
            content_type=authoritative_file.content_type,
            content=downloaded.content,
        )

    def _render_analysis(self, verified_audio: VerifiedAudio) -> str:
        assert self._analysis_adapter is not None
        response = self._analysis_adapter.analyze(
            FileAnalyzeInput(
                filename=verified_audio.name,
                content_type=verified_audio.content_type,
                content=verified_audio.content,
            ),
            request_id=uuid4(),
        )
        return _render_analysis_response(response, display_name=self.name)

    def _create_analysis_adapter(self) -> PipelineAnalysisPort | None:
        if self._injected_analysis_adapter is not None:
            return self._injected_analysis_adapter
        if _pipeline_probe_mode_enabled():
            return None
        return ApiAnalysisClient(
            base_url=_require_exact_analysis_api_url(_TRUSTED_ANALYSIS_API_INTERNAL_URL),
            api_key=_require_analysis_api_key(os.getenv("MTBANK_API_KEY")),
            timeout_seconds=min(self.valves.HTTP_TIMEOUT_SECONDS, 210),
        )

    def _create_assistant_client(self) -> ApiAssistantClient:
        return ApiAssistantClient(
            base_url=_require_exact_analysis_api_url(_TRUSTED_ANALYSIS_API_INTERNAL_URL),
            api_key=_require_analysis_api_key(os.getenv("MTBANK_API_KEY")),
            timeout_seconds=min(self.valves.HTTP_TIMEOUT_SECONDS, 50),
        )

    def _create_file_client(self) -> FileClient:
        return self._client_factory(
            base_url=_require_exact_openwebui_internal_url(_TRUSTED_OPENWEBUI_INTERNAL_URL),
            email=_required_setting(os.getenv("WEBUI_ADMIN_EMAIL"), "WEBUI_ADMIN_EMAIL"),
            password=_required_setting(os.getenv("WEBUI_ADMIN_PASSWORD"), "WEBUI_ADMIN_PASSWORD"),
            timeout_seconds=min(self.valves.HTTP_TIMEOUT_SECONDS, 10),
        )

    def _validate_configuration(self) -> None:
        _require_exact_openwebui_internal_url(_TRUSTED_OPENWEBUI_INTERNAL_URL)
        _require_nondefault_pipelines_api_key(os.getenv("PIPELINES_API_KEY"))
        if not _pipeline_probe_mode_enabled():
            _require_analysis_api_key(os.getenv("MTBANK_API_KEY"))


def _parse_authoritative_file(payload: object, *, expected_file_id: str) -> AuthoritativeFile:
    if not isinstance(payload, Mapping):
        raise FileFetchError("OpenWebUI вернул некорректную metadata файла")

    file_id = _normalise_authoritative_uuid(payload.get("id"), "идентификатор файла")
    if file_id != expected_file_id:
        raise FileFetchError("OpenWebUI вернул другой идентификатор файла")
    user_id = _normalise_authoritative_text(payload.get("user_id"), "владелец файла", max_length=256)
    filename = _normalise_authoritative_text(payload.get("filename"), "имя файла", max_length=512)
    meta = payload.get("meta")
    if not isinstance(meta, Mapping):
        raise FileFetchError("authoritative metadata файла отсутствует")

    meta_name = meta.get("name")
    name = filename
    if meta_name not in (None, ""):
        name = _normalise_authoritative_text(meta_name, "имя файла", max_length=512)
    content_type = _normalise_media_type(meta.get("content_type"))
    size = meta.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise FileFetchError("authoritative размер файла некорректен")
    file_hash = meta.get("file_hash")
    if not isinstance(file_hash, str) or not _SHA256_PATTERN.fullmatch(file_hash):
        raise FileFetchError("authoritative SHA-256 файла некорректен")

    return AuthoritativeFile(
        file_id=file_id,
        user_id=user_id,
        name=name,
        content_type=content_type,
        size=size,
        file_hash=file_hash,
    )


def _is_auxiliary_task(metadata: Mapping[str, Any]) -> bool:
    task = metadata.get("task")
    return isinstance(task, str) and bool(task.strip())


def _as_user_mapping(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    result: dict[str, object] = {}
    for key, item_value in value.items():
        if not isinstance(key, str):
            return None
        result[key] = item_value
    return result


def _audio_format_for_content_type(content_type: str) -> str:
    for audio_format, mime_types in _AUDIO_MIME_TYPES.items():
        if content_type in mime_types:
            return audio_format
    raise FileMediaError("authoritative MIME не поддерживается")


def _has_expected_magic(content: bytes, audio_format: str) -> bool:
    if audio_format == "wav":
        return len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WAVE"
    if audio_format == "mpeg":
        return content.startswith(b"ID3") or (len(content) >= 2 and content[0] == 0xFF and content[1] & 0xE0 == 0xE0)
    if audio_format == "ogg":
        return content.startswith(b"OggS")
    if audio_format == "webm":
        return content.startswith(b"\x1a\x45\xdf\xa3")
    return False


def _require_exact_openwebui_internal_url(value: object) -> str:
    """Допускает только неизменяемый internal authority для admin callback."""

    try:
        return require_exact_base_url(value, expected=_TRUSTED_OPENWEBUI_INTERNAL_URL)
    except TrustedHttpError as error:
        raise ValueError("OpenWebUI callback должен быть http://openwebui:8080") from error


def _require_exact_analysis_api_url(value: object) -> str:
    try:
        return require_exact_base_url(value, expected=_TRUSTED_ANALYSIS_API_INTERNAL_URL)
    except TrustedHttpError as error:
        raise ValueError("analysis API callback должен быть http://api:8000") from error


def _normalise_user_id(value: object) -> str:
    if not isinstance(value, str):
        raise AttachmentMetadataError("идентификатор пользователя вложения должен быть строкой")
    return extract_user_id({"id": value})


def _normalise_authoritative_uuid(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise FileFetchError(f"OpenWebUI {field} некорректен")
    try:
        parsed = UUID(value)
    except (AttributeError, ValueError) as error:
        raise FileFetchError(f"OpenWebUI {field} некорректен") from error
    if str(parsed) != value:
        raise FileFetchError(f"OpenWebUI {field} некорректен")
    return value


def _normalise_authoritative_text(value: object, field: str, *, max_length: int) -> str:
    if not isinstance(value, str):
        raise FileFetchError(f"OpenWebUI {field} некорректно")
    normalised = value.strip()
    if not normalised or len(normalised) > max_length:
        raise FileFetchError(f"OpenWebUI {field} некорректно")
    if any(ord(character) < 32 or ord(character) == 127 for character in normalised):
        raise FileFetchError(f"OpenWebUI {field} некорректно")
    return normalised


def _normalise_media_type(value: object) -> str:
    media_type = _normalise_authoritative_text(value, "MIME", max_length=255).lower().split(";", maxsplit=1)[0].strip()
    if "/" not in media_type:
        raise FileFetchError("OpenWebUI MIME некорректен")
    return media_type


def _require_nondefault_pipelines_api_key(value: str | None) -> str:
    key = _required_setting(value, "PIPELINES_API_KEY")
    if len(key) < 32 or key.casefold() in _UNSAFE_PIPELINES_API_KEYS:
        raise ValueError("PIPELINES_API_KEY должен быть случайным non-default секретом длиной не менее 32 символов")
    return key


def _required_setting(value: str | None, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} обязателен")
    return value.strip()


def _pipeline_probe_mode_enabled() -> bool:
    """Probe разрешён только явным test-only значением true."""

    return os.getenv("MTBANK_PIPELINE_PROBE_MODE") == "true"


def _require_analysis_api_key(value: str | None) -> str:
    try:
        return require_runtime_secret("MTBANK_API_KEY", value)
    except SecretConfigurationError as error:
        raise ValueError("MTBANK_API_KEY отсутствует или небезопасен") from error


def _encode_audio_multipart(source: FileAnalyzeInput) -> tuple[bytes, str]:
    audio_format = _audio_format_for_content_type(source.content_type)
    content_type = {
        "wav": "audio/wav",
        "mpeg": "audio/mpeg",
        "ogg": "audio/ogg",
        "webm": "audio/webm",
    }[audio_format]
    boundary = f"----mtbank-analysis-{uuid4().hex}"
    prefix = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="audio"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("ascii")
    suffix = f"\r\n--{boundary}--\r\n".encode("ascii")
    return prefix + source.content + suffix, boundary


def _map_analysis_api_status(status_code: int) -> DomainError:
    if status_code == 413:
        return DomainError(ErrorCode.PAYLOAD_TOO_LARGE)
    if status_code == 415:
        return DomainError(ErrorCode.UNSUPPORTED_MEDIA)
    if status_code == 422:
        return DomainError(ErrorCode.INVALID_AUDIO)
    if status_code == 429:
        return DomainError(ErrorCode.QUOTA_EXCEEDED)
    if status_code == 502:
        return DomainError(ErrorCode.AGENT_FAILURE)
    if status_code == 504:
        return DomainError(ErrorCode.DEADLINE_EXCEEDED)
    return DomainError(ErrorCode.SERVICE_UNAVAILABLE)


def _read_json_response(response: Any) -> object:
    return json.loads(_read_bounded(response, _JSON_RESPONSE_MAX_BYTES).decode("utf-8"))


def _read_bounded(response: Any, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = response.read(min(64 * 1024, max_bytes - total + 1))
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > max_bytes:
            raise FileSizeError("вложение превышает жёсткий лимит bytes")
        chunks.append(chunk)


def _escape_filename_for_markdown(value: str) -> str:
    return html.escape(value, quote=True).translate(_MARKDOWN_FILENAME_TRANSLATION)


def _render_probe(probes: list[tuple[str, int, str]], *, display_name: str) -> str:
    lines = [f"## {display_name}", "", "OpenWebUI API файла вернул следующие точные bytes:", ""]
    for name, size, content_hash in probes:
        lines.extend(
            [
                f"- **Файл:** <code>{_escape_filename_for_markdown(name)}</code>",
                f"  - Размер: `{size}` bytes",
                f"  - SHA-256: `{content_hash}`",
            ]
        )
    return "\n".join(lines)


def _render_analysis_response(response: object, *, display_name: str) -> str:
    if not isinstance(response, AnalyzeResponse):
        model_dump = getattr(response, "model_dump", None)
        if not callable(model_dump):
            raise ValueError("analysis response не поддерживает canonical JSON serialization")
        rendered = json.dumps(model_dump(mode="json"), ensure_ascii=False, indent=2)
        return f"## {_markdown_text(display_name)}\n\n```json\n{_markdown_text(rendered)}\n```"

    priority_labels = {"low": "низкий", "medium": "средний", "high": "высокий", "critical": "критический"}
    checklist = response.quality_score.checklist
    transcript = "\n".join(
        f"**{_markdown_text(segment.speaker)}** "
        f"`{_format_timestamp(segment.start)}–{_format_timestamp(segment.end)}`  \n"
        f"{_markdown_text(segment.text)}"
        for segment in response.transcript
    )
    actions = "\n".join(f"{index}. {_markdown_text(item)}" for index, item in enumerate(response.action_items, 1))
    if not actions:
        actions = "Дальнейшие действия не определены."
    if response.compliance.issues:
        compliance = "\n".join(
            f"- **{_markdown_text(issue.severity.value)} — {_markdown_text(issue.rule_id)}:** "
            f"{_markdown_text(issue.explanation)}"
            for issue in response.compliance.issues
        )
    else:
        compliance = "Нарушений по проверяемой политике не обнаружено."
    canonical_json = json.dumps(response.model_dump(mode="json"), ensure_ascii=False, indent=2)

    return "\n".join(
        (
            f"## {_markdown_text(display_name)}",
            "",
            "### Итог",
            _markdown_text(response.summary),
            "",
            "### Классификация",
            f"- **Тема:** {_markdown_text(response.classification.topic)}",
            "- **Приоритет:** "
            + _markdown_text(priority_labels.get(response.classification.priority, response.classification.priority)),
            f"- **Уверенность:** {response.classification.confidence:.0%}",
            f"- **Обоснование:** {_markdown_text(response.classification.rationale)}",
            "",
            "### Качество обслуживания",
            f"- **Итоговая оценка:** {response.quality_score.total:.0f}/100",
            f"- Приветствие: {_pass_mark(checklist.greeting)}",
            f"- Выявление потребности: {_pass_mark(checklist.need_detection)}",
            f"- Решение предоставлено: {_pass_mark(checklist.solution_provided)}",
            f"- Завершение разговора: {_pass_mark(checklist.farewell)}",
            "",
            "### Compliance",
            compliance,
            "",
            "### Рекомендуемые действия",
            actions,
            "",
            "### Транскрипт",
            transcript,
            "",
            "<details>",
            "<summary>Технический JSON результата</summary>",
            "",
            "```json",
            canonical_json,
            "```",
            "</details>",
            "",
            f"_Run ID: `{response.meta.run_id}` · Обработка: {response.meta.processing_ms / 1000:.1f} с · "
            f"Needs review: {'да' if response.meta.needs_review else 'нет'}_",
        )
    )


def _format_timestamp(seconds: float) -> str:
    whole = max(0, int(seconds))
    return f"{whole // 60:02d}:{whole % 60:02d}"


def _pass_mark(value: bool) -> str:
    return "✅" if value else "❌"


def _markdown_text(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("<", "&lt;").replace(">", "&gt;")


def _bounded_assistant_history(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    bounded: list[dict[str, str]] = []
    for item in messages[-_TEXT_ASSISTANT_MAX_HISTORY_MESSAGES:]:
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        normalized = content.strip()[:_TEXT_ASSISTANT_MAX_MESSAGE_CHARS]
        if normalized:
            bounded.append({"role": role, "content": normalized})
    return bounded


def _controlled_analysis_error(display_name: str, error: Exception) -> str:
    from mtbank_ai.domain.errors import ERROR_SPECS, DomainError

    if isinstance(error, DomainError):
        return _controlled_message(display_name, ERROR_SPECS[error.code].message)
    return _controlled_message(display_name, "Анализ временно недоступен. Повторите запрос позже.")


def _controlled_message(display_name: str, message: str) -> str:
    return f"## {display_name}\n\n{message}"
