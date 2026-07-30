"""Create synthetic/no-PII OpenWebUI demo chats after production readiness is green."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = os.getenv("OPENWEBUI_E2E_URL", "https://mtbank.arbitron.dev").rstrip("/")
MODEL_ID = "mtbank-attachment-probe"
TIMEOUT_SECONDS = 300
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class Scenario:
    slug: str
    title: str
    prompt: str
    relative_file: str | None = None
    media_type: str | None = None


SCENARIOS = (
    Scenario(
        "capabilities",
        "01 — Возможности и ограничения демо",
        "Какие сценарии демонстрирует система и какие ограничения у synthetic/no-PII демо?",
    ),
    Scenario(
        "card-complaint",
        "02 — Жалоба по карте и банкомату",
        "Проверь приложенный аудиозвонок. Покажи роли, тему, качество, compliance, итог и действия.",
        "test_data/synthetic/card-complaint-8k.wav",
        "audio/wav",
    ),
    Scenario(
        "transfer-question",
        "03 — Консультация по переводу",
        "Проверь приложенный аудиозвонок. Выдели роли, тему, результат разговора и дальнейшие действия.",
        "test_data/synthetic/transfer-question-16k.mp3",
        "audio/mpeg",
    ),
    Scenario(
        "mobile-security",
        "04 — Безопасность мобильного приложения",
        "Проверь приложенный аудиозвонок. Особое внимание удели compliance, рискам и рекомендациям.",
        "test_data/synthetic/mobile-app-security-16k.ogg",
        "audio/ogg",
    ),
)


def required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def request_json(
    path: str,
    *,
    method: str,
    token: str | None = None,
    payload: object | None = None,
    body: bytes | None = None,
    content_type: str = "application/json",
) -> dict[str, Any]:
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = content_type
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(BASE_URL + path, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            data = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as error:
        detail = error.read(500).decode(errors="replace")
        raise RuntimeError(f"{method} {path}: HTTP {error.code}: {detail}") from error
    if len(data) > MAX_RESPONSE_BYTES:
        raise RuntimeError(f"{method} {path}: response too large")
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{method} {path}: response is not an object")
    return parsed


def multipart(filename: str, media_type: str, content: bytes) -> tuple[bytes, str]:
    boundary = "----mtbank-demo-" + uuid.uuid4().hex
    marker = boundary.encode()
    body = b"".join(
        (
            b"--" + marker + b"\r\n",
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(),
            f"Content-Type: {media_type}\r\n\r\n".encode(),
            content,
            b"\r\n--" + marker + b"--\r\n",
        )
    )
    return body, f"multipart/form-data; boundary={boundary}"


def validated_fixture(scenario: Scenario) -> tuple[Path, bytes, str]:
    assert scenario.relative_file is not None
    manifest = json.loads((ROOT / "test_data/manifest.yaml").read_text())
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("test manifest has no entries")
    expected_path = scenario.relative_file.removeprefix("test_data/")
    entry = next((item for item in entries if isinstance(item, dict) and item.get("path") == expected_path), None)
    if (
        not isinstance(entry, dict)
        or entry.get("kind") != "speech_reference"
        or entry.get("license") != "LicenseRef-MTBank-Synthetic-EdgeTTS-Demo"
        or "реальных клиентов нет" not in str(entry.get("provenance", ""))
    ):
        raise RuntimeError(f"{scenario.slug}: fixture provenance is not approved")
    path = (ROOT / scenario.relative_file).resolve(strict=True)
    synthetic_root = (ROOT / "test_data/synthetic").resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(synthetic_root):
        raise RuntimeError(f"{scenario.slug}: fixture escapes synthetic corpus")
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != entry.get("sha256"):
        raise RuntimeError(f"{scenario.slug}: fixture hash differs from approved manifest")
    return path, content, digest


def upload(token: str, scenario: Scenario) -> tuple[dict[str, Any], str]:
    assert scenario.media_type is not None
    path, content, digest = validated_fixture(scenario)
    body, content_type = multipart(path.name, scenario.media_type, content)
    uploaded = request_json(
        "/api/v1/files/?process=true",
        method="POST",
        token=token,
        body=body,
        content_type=content_type,
    )
    meta = uploaded.get("meta")
    if not isinstance(meta, dict) or meta.get("file_hash") != digest or meta.get("size") != len(content):
        raise RuntimeError(f"{scenario.slug}: uploaded metadata mismatch")
    return uploaded, digest


def completion(token: str, scenario: Scenario, uploaded: dict[str, Any] | None) -> str:
    payload: dict[str, Any] = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": scenario.prompt}],
        "stream": False,
    }
    if uploaded is not None:
        file_id = uploaded.get("id")
        filename = uploaded.get("filename")
        meta = uploaded.get("meta")
        payload["files"] = [
            {"type": "file", "id": file_id, "file": {"id": file_id, "filename": filename, "meta": meta}}
        ]
    response = request_json("/api/chat/completions", method="POST", token=token, payload=payload)
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"{scenario.slug}: completion has no content") from error
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"{scenario.slug}: empty completion")
    if "недоступ" in content.casefold():
        raise RuntimeError(f"{scenario.slug}: service unavailable response")
    if uploaded is not None and "<pre>" not in content:
        raise RuntimeError(f"{scenario.slug}: analysis JSON is absent")
    return content


def attachment_descriptor(uploaded: dict[str, Any]) -> dict[str, Any]:
    file_id = uploaded.get("id")
    filename = uploaded.get("filename")
    meta = uploaded.get("meta")
    if not isinstance(file_id, str) or not isinstance(filename, str) or not isinstance(meta, dict):
        raise RuntimeError("uploaded file descriptor is invalid")
    return {"type": "file", "id": file_id, "file": {"id": file_id, "filename": filename, "meta": meta}}


def chat_payload(scenario: Scenario, content: str, uploaded: dict[str, Any] | None) -> dict[str, Any]:
    user_id = uuid.uuid4().hex
    assistant_id = uuid.uuid4().hex
    timestamp = int(time.time())
    files = [attachment_descriptor(uploaded)] if uploaded is not None else []
    return {
        "chat": {
            "id": "",
            "title": scenario.title,
            "models": [MODEL_ID],
            "params": {},
            "history": {
                "messages": {
                    user_id: {
                        "id": user_id,
                        "parentId": None,
                        "childrenIds": [assistant_id],
                        "role": "user",
                        "content": scenario.prompt,
                        "timestamp": timestamp,
                        "models": [MODEL_ID],
                        "files": files,
                    },
                    assistant_id: {
                        "id": assistant_id,
                        "parentId": user_id,
                        "childrenIds": [],
                        "role": "assistant",
                        "content": content,
                        "timestamp": timestamp,
                        "model": MODEL_ID,
                        "modelName": "MTBank Attachment Probe",
                        "done": True,
                    },
                },
                "currentId": assistant_id,
            },
            "messages": [
                {
                    "id": user_id,
                    "role": "user",
                    "content": scenario.prompt,
                    "timestamp": timestamp,
                    "files": files,
                },
                {
                    "id": assistant_id,
                    "role": "assistant",
                    "content": content,
                    "timestamp": timestamp,
                    "model": MODEL_ID,
                    "done": True,
                },
            ],
            "files": files,
            "tags": [],
            "timestamp": timestamp * 1_000,
        }
    }


def main() -> None:
    output = Path(os.getenv("MTBANK_DEMO_CHAT_OUTPUT", ROOT / "tmp/e2e-demo-chats/created-chats.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    ready = request_json("/health/ready", method="GET")
    if ready.get("status") != "ready":
        raise RuntimeError("production readiness is not green")
    session = request_json(
        "/api/v1/auths/signin",
        method="POST",
        payload={"email": required("WEBUI_EVALUATOR_EMAIL"), "password": required("WEBUI_EVALUATOR_PASSWORD")},
    )
    token = session.get("token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("sign-in returned no token")

    results: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        uploaded = None
        digest = None
        if scenario.relative_file is not None:
            uploaded, digest = upload(token, scenario)
        content = completion(token, scenario, uploaded)
        created = request_json(
            "/api/v1/chats/new",
            method="POST",
            token=token,
            payload=chat_payload(scenario, content, uploaded),
        )
        chat_id = created.get("id")
        if not isinstance(chat_id, str) or not re.fullmatch(r"[0-9a-f-]{16,64}", chat_id):
            raise RuntimeError(f"{scenario.slug}: invalid chat id")
        saved = request_json(f"/api/v1/chats/{chat_id}", method="GET", token=token)
        saved_chat = saved.get("chat")
        if not isinstance(saved_chat, dict) or saved_chat.get("title") != scenario.title:
            raise RuntimeError(f"{scenario.slug}: saved chat did not round-trip")
        saved_files = saved_chat.get("files", [])
        if uploaded is not None and (
            not isinstance(saved_files, list) or not saved_files or saved_files[0].get("id") != uploaded.get("id")
        ):
            raise RuntimeError(f"{scenario.slug}: saved attachment did not round-trip")
        results.append(
            {
                "slug": scenario.slug,
                "title": scenario.title,
                "chat_id": chat_id,
                "url": f"{BASE_URL}/c/{chat_id}",
                "fixture_sha256": digest,
                "answer_chars": len(content),
            }
        )
    output.write_text(json.dumps({"scope": "synthetic/no-PII", "chats": results}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"created": len(results), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"demo chat creation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None
