"""Выполняет реальный Phase 0 attachment contract через gateway OpenWebUI."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import secrets
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from mtbank_ai.domain.analysis import AnalyzeResponse

MAIN_PIPELINE_ID = "mtbank-attachment-probe"
BASE_URL = os.getenv("OPENWEBUI_E2E_URL", "http://127.0.0.1:3000").rstrip("/")
_TIMEOUT_SECONDS = 180
_GATEWAY_MAX_REQUEST_BODY_BYTES = 25 * 1024 * 1024 + 64 * 1024
_ROOT = Path(__file__).resolve().parents[1]
_CANONICAL_FIXTURE_ID = "synthetic-card-complaint-telephone"
_PUBLIC_READ_GRANT = {
    "principal_type": "user",
    "principal_id": "*",
    "permission": "read",
}
_DISABLED_ORDINARY_USER_PERMISSIONS = {
    ("chat", "web_upload"): False,
    ("chat", "stt"): False,
    ("chat", "tts"): False,
    ("chat", "call"): False,
    ("features", "web_search"): False,
}
def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} обязателен для host-run OpenWebUI E2E")
    return value


def _request_json(
    path: str,
    *,
    method: str,
    token: str | None = None,
    payload: object | None = None,
    raw_body: bytes | None = None,
    content_type: str = "application/json",
) -> dict[str, Any]:
    if payload is not None and raw_body is not None:
        raise RuntimeError("запрос не может одновременно содержать JSON и raw body")
    data = json.dumps(payload).encode("utf-8") if payload is not None else raw_body
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = content_type
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(f"{BASE_URL}{path}", data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            response_body = response.read()
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"{method} {path} вернул HTTP {error.code}: {detail}") from error

    parsed = json.loads(response_body.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{method} {path} вернул JSON не-объект")
    return parsed


def _assert_json_rejected(path: str, token: str, payload: object, *, detail: str) -> None:
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        urlopen(request, timeout=_TIMEOUT_SECONDS)
    except HTTPError as error:
        response_body = error.read(400).decode("utf-8", errors="replace")
        if error.code == 403 and detail in response_body:
            return
        raise RuntimeError(f"POST {path} вернул HTTP {error.code}: {response_body}") from error
    raise RuntimeError(f"POST {path} не отклонил запрещённый JSON")


def _canonical_speech_fixture() -> tuple[str, bytes, str]:
    manifest = json.loads((_ROOT / "test_data" / "manifest.yaml").read_text(encoding="utf-8"))
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("test_data manifest не содержит entries")
    entry = next((item for item in entries if isinstance(item, dict) and item.get("id") == _CANONICAL_FIXTURE_ID), None)
    if not isinstance(entry, dict):
        raise RuntimeError("canonical speech fixture отсутствует в test_data manifest")
    if entry.get("kind") != "speech_reference" or entry.get("license") != "LicenseRef-MTBank-Synthetic-EdgeTTS-Demo":
        raise RuntimeError("canonical speech fixture не имеет разрешённой synthetic provenance")
    relative_path = entry.get("path")
    expected_hash = entry.get("sha256")
    if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
        raise RuntimeError("canonical speech fixture не имеет path или SHA-256")
    content = (_ROOT / "test_data" / relative_path).read_bytes()
    actual_hash = hashlib.sha256(content).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError("canonical speech fixture не совпадает с manifest SHA-256")
    return Path(relative_path).name, content, actual_hash


def _multipart_file(filename: str, content: bytes, media_type: str = "audio/wav") -> tuple[bytes, str]:
    boundary = f"----mtbank-e2e-{uuid.uuid4().hex}"
    delimiter = boundary.encode("ascii")
    body = b"".join(
        (
            b"--" + delimiter + b"\r\n",
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("ascii"),
            f"Content-Type: {media_type}\r\n\r\n".encode("ascii"),
            content,
            b"\r\n--" + delimiter + b"--\r\n",
        )
    )
    return body, f"multipart/form-data; boundary={boundary}"


def _upload_audio(token: str, *, filename: str, content: bytes, media_type: str = "audio/wav") -> dict[str, Any]:
    body, content_type = _multipart_file(filename, content, media_type)
    return _request_json(
        "/api/v1/files/?process=true",
        method="POST",
        token=token,
        raw_body=body,
        content_type=content_type,
    )


def _browser_file_item(uploaded: dict[str, Any]) -> dict[str, Any]:
    file_id = uploaded.get("id")
    filename = uploaded.get("filename")
    meta = uploaded.get("meta")
    if not isinstance(file_id, str) or not isinstance(filename, str) or not isinstance(meta, dict):
        raise RuntimeError("upload response не содержит browser file descriptor")
    return {
        "type": "file",
        "id": file_id,
        "file": {"id": file_id, "filename": filename, "meta": meta},
    }


def _chat_with_file(token: str, uploaded: dict[str, Any]) -> str:
    response = _request_json(
        "/api/chat/completions",
        method="POST",
        token=token,
        payload={
            "model": MAIN_PIPELINE_ID,
            "messages": [{"role": "user", "content": "Проверь приложенный WAV."}],
            "files": [_browser_file_item(uploaded)],
            "stream": False,
        },
    )
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("chat response не содержит assistant text") from error
    if not isinstance(content, str):
        raise RuntimeError("chat response content не является text")
    return content


def _assert_model_configuration(token: str) -> None:
    response = _request_json("/api/models", method="GET", token=token)
    models = response.get("data")
    if not isinstance(models, list):
        raise RuntimeError("model list не содержит data")
    matching = [model for model in models if isinstance(model, dict) and model.get("id") == MAIN_PIPELINE_ID]
    if len(matching) != 1:
        raise RuntimeError("workspace override Pipeline модели не найден")
    info = matching[0].get("info")
    if not isinstance(info, dict):
        raise RuntimeError("workspace override Pipeline модели не содержит info")
    meta = info.get("meta")
    capabilities = meta.get("capabilities") if isinstance(meta, dict) else None
    if not isinstance(capabilities, dict) or capabilities.get("file_context") is not False:
        raise RuntimeError("Pipeline модель не имеет file_context=false")
    grants = info.get("access_grants")
    if not isinstance(grants, list) or len(grants) != 1:
        raise RuntimeError("Pipeline модель не имеет ровно один public read grant")
    grant = grants[0]
    if not isinstance(grant, dict) or any(grant.get(name) != value for name, value in _PUBLIC_READ_GRANT.items()):
        raise RuntimeError("Pipeline модель не имеет точного public read grant")


def _assert_model_visible_to_user(token: str) -> None:
    response = _request_json("/api/models", method="GET", token=token)
    models = response.get("data")
    if not isinstance(models, list) or not any(
        isinstance(model, dict) and model.get("id") == MAIN_PIPELINE_ID for model in models
    ):
        raise RuntimeError("ordinary user не видит mtbank-attachment-probe")


def _assert_ordinary_user_permissions(token: str) -> None:
    response = _request_json("/api/v1/auths/", method="GET", token=token)
    permissions = response.get("permissions")
    if not isinstance(permissions, dict):
        raise RuntimeError("ordinary user session не содержит permissions")
    for (section, name), expected in _DISABLED_ORDINARY_USER_PERMISSIONS.items():
        values = permissions.get(section)
        if not isinstance(values, dict) or values.get(name) is not expected:
            raise RuntimeError(f"ordinary user permission {section}.{name} не отключён")


def _assert_remote_image_urls_rejected(token: str) -> None:
    remote_image = {
        "type": "image_url",
        "image_url": {"url": "https://example.invalid/unbounded-image"},
    }
    for path in ("/api/chat/completions", "/api/v1/chat/completions"):
        _assert_json_rejected(
            path,
            token,
            {
                "model": MAIN_PIPELINE_ID,
                "messages": [{"role": "user", "content": [remote_image]}],
                "stream": False,
            },
            detail="Remote image URLs disabled for Phase 0",
        )

    _assert_json_rejected(
        "/api/v1/chat/completions",
        token,
        {
            "model": MAIN_PIPELINE_ID,
            "messages": [{"role": "user", "content": "Безопасный client message."}],
            "user_message": {"role": "user", "content": [remote_image]},
            "stream": False,
        },
        detail="Remote image URLs disabled for Phase 0",
    )
    _assert_json_rejected(
        "/api/v1/chats/new",
        token,
        {"chat": {"history": {"messages": {"message-id": {"content": [remote_image]}}}}},
        detail="Remote image URLs disabled for Phase 0",
    )


def _create_user(admin_token: str, label: str) -> str:
    response = _request_json(
        "/api/v1/auths/add",
        method="POST",
        token=admin_token,
        payload={
            "email": f"{label}-{uuid.uuid4().hex}@example.test",
            "password": f"E2e!{secrets.token_urlsafe(24)}",
            "name": f"Пользователь проверки Phase 0 {label}",
            "role": "user",
        },
    )
    token = response.get("token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("admin add-user response не содержит session token")
    return token


def _assert_gateway_rejects_oversized_upload() -> None:
    request = Request(
        f"{BASE_URL}/api/v1/files/?process=false",
        data=b"x" * (_GATEWAY_MAX_REQUEST_BODY_BYTES + 1),
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    try:
        urlopen(request, timeout=_TIMEOUT_SECONDS)
    except HTTPError as error:
        server = error.headers.get("Server", "")
        if error.code == 413 and "nginx" in server.casefold():
            return
        raise RuntimeError(f"gateway вернул HTTP {error.code} вместо nginx 413") from error
    raise RuntimeError("gateway принял request body больше ingress лимита")


def _extract_analysis(content: str) -> AnalyzeResponse:
    match = re.search(r"<pre>(.*?)</pre>", content, flags=re.DOTALL)
    if match is None:
        raise RuntimeError("production Pipeline output не содержит canonical JSON в <pre>")
    try:
        payload = json.loads(html.unescape(match.group(1)))
    except json.JSONDecodeError as error:
        raise RuntimeError("production Pipeline output не содержит валидный escaped JSON") from error
    try:
        response = AnalyzeResponse.model_validate(payload)
    except ValueError as error:
        raise RuntimeError("production Pipeline output не совпадает с AnalyzeResponse") from error
    if (
        not response.transcript
        or not response.classification.topic
        or not response.classification.priority
        or not response.quality_score.policy_version
        or not response.compliance.policy_version
        or not response.summary.strip()
        or not response.meta.versions.code_sha
        or not response.meta.versions.prompt_bundle_hash
    ):
        raise RuntimeError(
            "AnalyzeResponse не содержит populated transcript/classification/quality/compliance/summary/meta"
        )
    return response


def _assert_production_analysis(token: str) -> tuple[int, str, str]:
    filename, content, content_hash = _canonical_speech_fixture()
    uploaded = _upload_audio(token, filename=filename, content=content)
    file_id = uploaded.get("id")
    if not isinstance(file_id, str):
        raise RuntimeError("canonical upload не вернул file id")
    server_file = _request_json(f"/api/v1/files/{file_id}", method="GET", token=token)
    server_meta = server_file.get("meta")
    if not isinstance(server_meta, dict):
        raise RuntimeError("authoritative FileModel не содержит metadata")
    if (
        server_meta.get("size") != len(content)
        or server_meta.get("file_hash") != content_hash
        or server_meta.get("content_type") != "audio/wav"
    ):
        raise RuntimeError("authoritative FileModel не совпадает с canonical fixture")

    output = _chat_with_file(token, uploaded)
    if filename in output:
        raise RuntimeError("production Pipeline output не должен echo authoritative filename")
    analysis = _extract_analysis(output)
    return len(content), content_hash, str(analysis.meta.run_id)


def main() -> None:
    admin_session = _request_json(
        "/api/v1/auths/signin",
        method="POST",
        payload={
            "email": _required_env("WEBUI_ADMIN_EMAIL"),
            "password": _required_env("WEBUI_ADMIN_PASSWORD"),
        },
    )
    admin_token = admin_session.get("token")
    if not isinstance(admin_token, str) or not admin_token:
        raise RuntimeError("admin sign-in response не содержит рабочую сессию")
    _assert_model_configuration(admin_token)

    ordinary_user_token = _create_user(admin_token, "ordinary")
    foreign_user_token = _create_user(admin_token, "idor")
    _assert_model_visible_to_user(ordinary_user_token)
    _assert_ordinary_user_permissions(ordinary_user_token)
    _assert_remote_image_urls_rejected(ordinary_user_token)

    positive_size, positive_hash, run_id = _assert_production_analysis(ordinary_user_token)

    invalid_magic_file = _upload_audio(
        ordinary_user_token,
        filename="attachment-invalid-magic.wav",
        content=b"not-a-wav",
    )
    invalid_magic_content = _chat_with_file(ordinary_user_token, invalid_magic_file)
    if "не является поддерживаемым аудиофайлом" not in invalid_magic_content or "<pre>" in invalid_magic_content:
        raise RuntimeError("Pipeline не отклонил неверные audio magic bytes до production analysis")

    foreign_filename, foreign_bytes, foreign_hash = _canonical_speech_fixture()
    foreign_file = _upload_audio(foreign_user_token, filename=foreign_filename, content=foreign_bytes)
    foreign_content = _chat_with_file(ordinary_user_token, foreign_file)
    if "Вложение недоступно" not in foreign_content or foreign_hash in foreign_content or "<pre>" in foreign_content:
        raise RuntimeError("IDOR запрос не был отклонён до production analysis")

    _assert_gateway_rejects_oversized_upload()

    print(
        json.dumps(
            {
                "model": {"file_context": False, "public_read": True},
                "ordinary_user": {"model_visible": True, "resource_permissions_disabled": True},
                "production_analysis": {
                    "fixture": _CANONICAL_FIXTURE_ID,
                    "bytes": positive_size,
                    "sha256": positive_hash,
                    "run_id": run_id,
                },
                "magic": "неверный WAV magic отклонён до production analysis",
                "idor": "foreign-owner attachment недоступен до production analysis",
                "remote_images": "оба completion aliases, user_message и chat persistence отклонены",
                "ingress": "oversized request отклонён gateway с HTTP 413",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
