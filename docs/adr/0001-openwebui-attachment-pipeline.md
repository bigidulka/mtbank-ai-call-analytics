# ADR 0001: один external Pipeline для безопасных OpenWebUI-вложений

- **Статус:** принято
- **Дата:** 2026-07-15
- **Scope:** только compatibility spike Phase 0; ASR и анализ звонка сюда не входят.

## Контекст

Задание требует настоящий OpenWebUI Pipeline. Контракт закреплён на двух
OCI-артефактах и исходных ревизиях:

- OpenWebUI `ghcr.io/open-webui/open-webui@sha256:9fcea9c6e32ab60b0498f3986c6cdf651ddbe61db48d2213a3d28048ddd673d4`
  (revision `ecd48e2f718220a6400ecf49eafd4867a38feb10`);
- legacy Pipelines `ghcr.io/open-webui/pipelines@sha256:b48e9bc338ce2be0acfbeff01810db72408a12f07739f9e3879c1f2b00952d6e`
  (revision `039f9c54f8e9f9bcbabde02c2c853e80d25c79e4`).

В этой версии OpenWebUI `process_pipeline_inlet_filter` всегда добавляет
выбранный внешний Pipeline в inlet-chain и вызывает
`POST /{model}/filter/inlet` до удаления `body.metadata`. Фронтенд передаёт
`files`, а OpenWebUI формирует `body.metadata.files` для inlet. Затем provider
route удаляет `metadata`, добавляет server-authenticated `body.user` и
пересылает разрешённые custom верхнеуровневые поля в legacy Pipelines.
`OpenAIChatCompletionForm(extra="allow")` вызывает именно синхронный
`Pipeline.pipe(user_message, model_id, messages, body)`. Async skeleton из
README не совместим с этим pinned legacy runtime.

`__event_emitter__`, `__files__` и другие rich injection-параметры принадлежат
internal Functions, а не legacy external Pipelines. Поэтому Phase 0 отдаёт
обычный text/SSE response.

## Решение

`pipeline.py` содержит единственный root `Pipeline` с id
`mtbank-attachment-probe`. Он одновременно реализует:

- `async inlet(body, user)`: удаляет все client-supplied ключи `mtbank_*`,
  берёт только inlet-envelope `user.id`, принимает ровно один direct browser
  item `type=file` с canonical UUID и создаёт minimal HMAC-SHA256 reference;
- синхронный `pipe(...)`: получает эту reference после metadata stripping,
  проверяет HMAC, audience, subject, future skew, expiry и user binding,
  после чего использует только authoritative OpenWebUI `FileModel`.

Подписанный payload строго ограничен полями
`{v, aud, sub, file_id, iat, exp, signature}`. В нём никогда нет
client-provided имени, MIME, размера, hash, path, URL или другого metadata
hint. Phase 0 поддерживает только один файл.

Перед любым content fetch Pipeline выполняет admin-authenticated
`GET /api/v1/files/{id}` и требует точного равенства:

```text
FileModel.user_id == signed sub == mtbank_attachment_user_id == body['user']['id']
```

Поддерживается только direct-owned file. Shared ACL намеренно fail-closed,
даже если API OpenWebUI разрешил бы shared read. Только после проверки owner
Pipeline проверяет authoritative `meta.size` против жёсткого лимита 25 MiB и
`meta.content_type` против allowlist audio MIME, затем вызывает
`GET /api/v1/files/{id}/content`. Он требует matching audio magic bytes,
сверяет фактические byte length и SHA-256 с `meta.size` и `meta.file_hash`, а
в ответе рендерит только authoritative `meta.name` (fallback `filename`),
фактический размер и hash.

Admin JWT кэшируется только в памяти; compare-and-clear при 401/403 не удаляет
более новый JWT конкурентного запроса.

### Доступ модели обычному пользователю

Bootstrap создаёт direct workspace override (без `base_model_id`) с единственным grant:

```json
{
  "principal_type": "user",
  "principal_id": "*",
  "permission": "read"
}
```

Это public read, а не public write: ordinary `role=user` получает модель в
`/api/models`, но не получает прав на изменение её конфигурации. E2E создаёт
ordinary user, проверяет видимость модели и positive WAV-flow от имени этого
пользователя.

### Административный callback

Адрес callback не является Valve и не может быть изменён через runtime UI.
Module config допускает только точный `http://openwebui:8080`: HTTP, host
`openwebui`, port `8080`, без path, userinfo, query и fragment. Поэтому update
Valves не способен перенаправить admin credentials на произвольный URL.

`pipeline.py` и `model_capability_bootstrap.py` используют общий trusted `urllib`
opener: `ProxyHandler({})` игнорирует все upper/lower HTTP(S)/ALL proxy variables,
redirect handler немедленно fail-closed до повторной отправки request, а request и
final response URL проверяются против точной expected scheme/host/port authority.
Compose дополнительно очищает upper/lower proxy variables у OpenWebUI, Pipelines
и bootstrap и устанавливает точный `NO_PROXY`/`no_proxy` только для внутренних
service authorities. Admin password и Bearer JWT не логируются.

### Fail-closed секреты

`.env.example` оставляет пустыми `WEBUI_ADMIN_PASSWORD`, `WEBUI_SECRET_KEY`,
`PIPELINES_API_KEY` и `MTBANK_ATTACHMENT_SIGNING_KEY`; копирование шаблона без
изменений не запускает Compose. Одноразовый `secrets-preflight` на уже
закреплённом OpenWebUI образе выполняется с `network_mode: none`, читает только
эти четыре значения и завершается до старта backend-сервисов.

Validator не логирует значения. Он отклоняет отсутствие, whitespace, короткие
значения, upstream default, example/change/placeholder marker и периодически
повторяющуюся строку. `pipelines` и `openwebui` ждут
`service_completed_successfully`, поэтому небезопасная конфигурация не
деградирует в работающий стек.

## Сетевая граница и ingress

OpenWebUI читает тело upload до собственной проверки размера, поэтому он не
публикуется на host. Единственная host-точка — `gateway`:

- `nginx@sha256:65645c7bb6a0661892a8b03b89d0743208a18dd2f3f17a54ef4b76fb8e2f2a10`;
- upstream tag при проверке: `nginx:1.27.5-alpine`;
- `docker pull`, `docker image inspect` и `nginx -v` 2026-07-15 подтвердили
  digest и `nginx/1.27.5` для linux/amd64;
- только gateway публикует `127.0.0.1:3000` по умолчанию;
- gateway подключён к `frontend` (`internal: true`) для upstream OpenWebUI и к
  отдельной `gateway-ingress` (`internal: false`) для Docker published-port ingress;
- к `gateway-ingress` подключён только gateway; OpenWebUI, Pipelines и
  model-bootstrap не получают non-internal network или host `ports`.

Gateway применяет `client_max_body_size 26279936` bytes: 25 MiB плюс 64 KiB для
multipart boundary и headers. `proxy_request_buffering on` не передаёт тело
upstream до его приёма gateway; oversize получает nginx HTTP 413. Для
SSE/WebSocket gateway использует HTTP/1.1, `Upgrade`/`Connection`, выключенный
response buffering и лимиты connect/send/read timeout.

Docker host всё равно может обратиться к IP container OpenWebUI, поэтому gateway
не является единственной size boundary. `openwebui_wrapper.py` запускает pinned
`open_webui.main:app` через `OpenWebUIPreBodyGuard` до FastAPI multipart/auth.
Wrapper сразу возвращает controlled HTTP 413 при `Content-Length > 26279936`.
При отсутствии, malformed или меньшем/ложном `Content-Length` он читает ASGI
chunks только в bounded `SpooledTemporaryFile` (64 KiB memory threshold, disk
после rollover), прекращает чтение при превышении того же cap и только затем
replay-ит bounded body upstream. WebSocket/lifespan scopes и streaming response
не буферизуются и передаются без изменений.

Phase 0 не использует native web loader, STT, TTS, RAG ingestion или call.
Compose задаёт false для `USER_PERMISSIONS_CHAT_WEB_UPLOAD`,
`USER_PERMISSIONS_CHAT_STT`, `USER_PERMISSIONS_CHAT_TTS`,
`USER_PERMISSIONS_CHAT_CALL`, `USER_PERMISSIONS_FEATURES_WEB_SEARCH` и
`ENABLE_WEB_SEARCH`. До body parsing wrapper возвращает controlled HTTP 403 на
POST `/api/v1/retrieval/process/{file,text,files/batch,web,youtube,web/search}`,
`/api/v1/audio/{transcriptions,speech}`, file data-content update и ordinary
knowledge file add/update/batch routes. Browser upload остаётся разрешённым, но
все `process` query variants канонизируются к единственному `process=false`.

После bounded body parsing wrapper проверяет оба completion aliases и все chat
persistence JSON writes. Remote HTTP(S) `image_url` в любом client-controlled
JSON location получает controlled HTTP 403. Это ingress defense-in-depth, но не
единственная граница: pinned OpenWebUI восстанавливает effective `messages` из
DB и преобразует сохранённые image-file records в structured `image_url` уже
после ingress.

Поэтому `openwebui_wrapper.py` также устанавливает pin-sensitive patch на exact
module global `open_webui.utils.middleware.convert_url_images_to_base64` с
сигнатурой `(form_data, user=None)`. В revision
`ecd48e2f718220a6400ecf49eafd4867a38feb10` этот global вызывается после DB
rehydrate и image-file conversion, но до `validate_url`, DNS, unbounded image
fetch и Pipeline inlet. Guard проверяет только effective `form_data['messages']`
и deterministic exception блокирует remote HTTP(S) structured `image_url.url`
до original converter; data image URLs, ordinary text и top-level audio files
передаются original converter без изменений. Повторная установка idempotent, а
signature drift завершает startup fail-closed. Обе сети OpenWebUI (`frontend` и
`pipeline-internal`) остаются `internal: true` как дополнительная, но не
самостоятельная защита от egress и Docker-internal targets. Non-internal
`gateway-ingress` подключена только к gateway.

OpenWebUI дополнительно имеет `RAG_FILE_MAX_SIZE=25`, `mem_limit: 1g` и
`pids_limit: 256`. У сервисов Pipelines, gateway, bootstrap и preflight также
есть ограниченные memory/pids budgets.

Legacy Pipelines остаётся в `pipeline-internal`, у которой `internal: true`, и
не имеет host `ports`. Но Docker host всё равно может обратиться к container IP
внутренней сети, поэтому firewall не является auth boundary. `auth_wrapper.py`
запускает upstream FastAPI app через ASGI wrapper: для **каждого HTTP route**
требуется единственный точный `Authorization: Bearer PIPELINES_API_KEY`,
проверенный `hmac.compare_digest`. Lifespan и non-HTTP scope передаются
upstream без изменений; healthcheck также аутентифицирован. Wrapper не пишет
key в лог. OpenWebUI и Pipelines запускаются с `GLOBAL_LOG_LEVEL=WARNING`,
Uvicorn warning level и отключённым access log, чтобы INFO payload/auth context
не попадал в runtime logs.

## Защита от task recursion и UI noise

OpenWebUI запускается с `ENABLE_TITLE_GENERATION=False`,
`ENABLE_FOLLOW_UP_GENERATION=False`, `ENABLE_TAGS_GENERATION=False`,
`ENABLE_OLLAMA_API=False`, `BYPASS_EMBEDDING_AND_RETRIEVAL=True`,
`RAG_EMBEDDING_ENGINE=openai` и `ENABLE_PERSISTENT_CONFIG=False`. External
engine предотвращает startup-загрузку локальной default embedding модели, но
из-за bypass не вызывается для attachment flow. Последнее гарантирует
применение Compose env после restart с существующим volume; пустой
`RAG_EMBEDDING_MODEL` не используется.

`Pipeline.inlet` пропускает auxiliary request с `metadata.task` без signed
attachment reference. Отдельный одноразовый `model-bootstrap` после model
discovery через admin API создаёт или обновляет workspace override с
`meta.capabilities.file_context=false` и завершается. Gateway публикуется только
после его `service_completed_successfully`; bootstrap подключён только к
frontend сети и не имеет доступа к Pipelines.

## Проверяемый wire contract

```text
Browser/API upload
  -> gateway: bounded request body, oversized -> nginx 413
  -> OpenWebUI stores FileModel {user_id, filename, meta.{name,content_type,size,file_hash}}
  -> body.metadata.files
  -> selected Pipeline.inlet({user, body})
  -> signed {v,aud,sub,file_id,iat,exp,signature} + attachment user id
  -> OpenWebUI strips metadata and injects authenticated body.user
  -> Bearer-authenticated legacy /v1/chat/completions
  -> Pipeline.pipe
  -> GET /api/v1/files/{id}, exact direct-owner equality
  -> authoritative audio MIME + size precheck
  -> GET /api/v1/files/{id}/content, audio magic + actual size/SHA-256
  -> text/SSE response с HTML-escaped filename без active Markdown/image syntax
```

Missing reference, invalid signature, owner mismatch и metadata/content fetch
failure до успешного probe дают одинаковое controlled сообщение
`Вложение недоступно`, не создавая existence/ownership oracle. Размер, MIME,
magic и integrity failures остаются отдельными безопасными категориями после
получения authoritative metadata.

## Последствия

Bytes остаются в OpenWebUI persistence и не дублируются в host volume. Цена —
жёсткая привязка к pinned legacy contract; digest pins, preflight, gateway,
auth wrapper, bootstrap capability и реальный E2E обязательны для любой
будущей миграции.

Unit tests покрывают canonical reference, UUID/audience/expiry, один inlet,
унифицированный pre-content error, MIME/magic/size/hash failure, безопасный
filename rendering, JWT refresh race, callback authority, secret preflight,
trusted no-proxy/no-redirect credential calls, ASGI auth, оба chat aliases,
remote-image/chat-persistence ingress, fetch-adjacent effective-message sink,
pre-body exact/over/chunked/replay и Compose boundaries.
`scripts/openwebui_image_sink_contract.py` импортирует actual pinned wrapper в
OpenWebUI container и доказывает module-global patch, rejection до original
converter, а также разрешённые data URL/text/top-level audio формы.
`scripts/openwebui_attachment_e2e.py` воспроизводит через
gateway browser-style `process=true` upload → chat proof, disabled permissions,
прямые и indirect resource-route 403, remote-image aliases/user_message/chat
write negatives, inert filename, magic-negative probe, user A → file user B
negative proof и ingress 413. `scripts/openwebui_guard_e2e.py` обращается с
Docker host к прямому OpenWebUI container IP и доказывает immediate 413/403 без
body. `scripts/pipelines_auth_e2e.py` доказывает 401 для missing и wrong Bearer
на `/v1/chat/completions`, `/{id}/filter/inlet` и `/{id}/valves`.
