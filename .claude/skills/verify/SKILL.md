---
name: verify-phase0-attachment
description: Проверяет реальный OpenWebUI Phase 0 attachment contract через gateway.
---

1. Проверить Compose без раскрытия interpolated secret values и пересоздать стек:
   ```bash
   docker compose config --quiet
   docker compose up -d --build --force-recreate --wait --wait-timeout 120
   ```
   `secrets-preflight` и одноразовый `model-bootstrap` должны завершиться с кодом 0,
   а `pipelines`, `openwebui` и `gateway` — стать healthy.
2. Выполнить host-run production flow через `gateway`:
   ```bash
   set -a
   . ./.env
   set +a
   OPENWEBUI_E2E_URL=http://127.0.0.1:3000 uv run --offline --no-sync python scripts/openwebui_attachment_e2e.py
   ```
   Скрипт использует один canonical licensed synthetic speech fixture из `test_data`,
   сверяет authoritative FileModel size/hash/MIME, извлекает escaped JSON из `<pre>`
   и валидирует populated `AnalyzeResponse`. Bad magic и IDOR должны завершиться до
   production analysis; также проверяются ordinary-user permissions, remote-image
   guard и nginx 413 для body больше 25 MiB плюс multipart reserve.
3. Проверить fetch-adjacent patch actual pinned middleware module:
   ```bash
   docker compose exec -T openwebui python - < scripts/openwebui_image_sink_contract.py
   ```
   Скрипт импортирует тот же `openwebui_wrapper`, проверяет exact module global и
   сигнатуру, отклоняет effective persisted-image shape до original converter и
   пропускает data URL, ordinary text и top-level audio file.
4. Проверить host-direct ASGI boundary OpenWebUI и Pipelines:
   ```bash
   OPENWEBUI_ID="$(docker compose ps -q openwebui)"
   OPENWEBUI_IP="$(docker inspect -f '{{with index .NetworkSettings.Networks "mtbank-ai-call-analytics_frontend"}}{{.IPAddress}}{{end}}' "$OPENWEBUI_ID")"
   uv run python scripts/openwebui_guard_e2e.py --base-url "http://$OPENWEBUI_IP:8080"

   PIPELINES_ID="$(docker compose ps -q pipelines)"
   PIPELINES_IP="$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$PIPELINES_ID")"
   uv run python scripts/pipelines_auth_e2e.py --base-url "http://$PIPELINES_IP:9099"
   ```
   Первый script открывает соединение с container IP, передаёт только headers с
   overlimit `Content-Length` и ожидает immediate HTTP 413; также ожидает 403
   на disabled resource routes без передачи body. Второй ожидает HTTP 401 без
   Bearer и с неправильным Bearer для `/v1/chat/completions`, `/{id}/filter/inlet`
   и `/{id}/valves`. Успешный flow шага 2 одновременно доказывает передачу
   корректного key OpenWebUI.
5. Проверить public ingress, единственный host binding и точную сетевую границу
   через `.HostConfig.PortBindings`/`.NetworkSettings.Networks` (не
   `docker compose port`, которое не применимо к `expose`-only services):
   ```bash
   curl --fail --silent --show-error http://127.0.0.1:3000/health
   PREFLIGHT_ID="$(docker compose ps -q --all secrets-preflight)"
   GATEWAY_ID="$(docker compose ps -q gateway)"
   OPENWEBUI_ID="$(docker compose ps -q openwebui)"
   PIPELINES_ID="$(docker compose ps -q pipelines)"
   MODEL_BOOTSTRAP_ID="$(docker compose ps -q --all model-bootstrap)"
   GATEWAY_FULL_ID="$(docker inspect -f '{{.Id}}' "$GATEWAY_ID")"
   FRONTEND_NETWORK=mtbank-ai-call-analytics_frontend
   GATEWAY_INGRESS_NETWORK=mtbank-ai-call-analytics_gateway-ingress
   PIPELINE_NETWORK=mtbank-ai-call-analytics_pipeline-internal
   APPLICATION_NETWORK=mtbank-ai-call-analytics_application-internal
   MONITORING_NETWORK=mtbank-ai-call-analytics_monitoring-internal

   test "$(docker inspect -f '{{len .HostConfig.PortBindings}}' "$GATEWAY_ID")" = 1
   docker inspect -f '{{range $containerPort, $bindings := .HostConfig.PortBindings}}{{range $bindings}}{{printf "%s %s\n" $containerPort .HostIp}}{{end}}{{end}}' "$GATEWAY_ID" | grep -Fx '8080/tcp 127.0.0.1'
   for id in "$PREFLIGHT_ID" "$OPENWEBUI_ID" "$PIPELINES_ID" "$MODEL_BOOTSTRAP_ID"; do
     test "$(docker inspect -f '{{len .HostConfig.PortBindings}}' "$id")" = 0
   done

   test "$(docker network inspect -f '{{.Internal}}' "$FRONTEND_NETWORK")" = true
   test "$(docker network inspect -f '{{.Internal}}' "$GATEWAY_INGRESS_NETWORK")" = false
   test "$(docker network inspect -f '{{.Internal}}' "$PIPELINE_NETWORK")" = true
   test "$(docker network inspect -f '{{.Internal}}' "$APPLICATION_NETWORK")" = true
   test "$(docker network inspect -f '{{.Internal}}' "$MONITORING_NETWORK")" = true

   test "$(docker inspect -f '{{len .NetworkSettings.Networks}}' "$GATEWAY_ID")" = 4
   test "$(docker inspect -f '{{with index .NetworkSettings.Networks "mtbank-ai-call-analytics_frontend"}}present{{end}}' "$GATEWAY_ID")" = present
   test "$(docker inspect -f '{{with index .NetworkSettings.Networks "mtbank-ai-call-analytics_gateway-ingress"}}present{{end}}' "$GATEWAY_ID")" = present
   test "$(docker inspect -f '{{with index .NetworkSettings.Networks "mtbank-ai-call-analytics_application-internal"}}present{{end}}' "$GATEWAY_ID")" = present
   test "$(docker inspect -f '{{with index .NetworkSettings.Networks "mtbank-ai-call-analytics_monitoring-internal"}}present{{end}}' "$GATEWAY_ID")" = present
   test "$(docker inspect -f '{{len .NetworkSettings.Networks}}' "$OPENWEBUI_ID")" = 2
   test "$(docker inspect -f '{{with index .NetworkSettings.Networks "mtbank-ai-call-analytics_frontend"}}present{{end}}' "$OPENWEBUI_ID")" = present
   test "$(docker inspect -f '{{with index .NetworkSettings.Networks "mtbank-ai-call-analytics_pipeline-internal"}}present{{end}}' "$OPENWEBUI_ID")" = present
   test "$(docker inspect -f '{{len .NetworkSettings.Networks}}' "$PIPELINES_ID")" = 2
   test "$(docker inspect -f '{{with index .NetworkSettings.Networks "mtbank-ai-call-analytics_pipeline-internal"}}present{{end}}' "$PIPELINES_ID")" = present
   test "$(docker inspect -f '{{with index .NetworkSettings.Networks "mtbank-ai-call-analytics_application-internal"}}present{{end}}' "$PIPELINES_ID")" = present
   test "$(docker inspect -f '{{len .NetworkSettings.Networks}}' "$MODEL_BOOTSTRAP_ID")" = 1
   test "$(docker inspect -f '{{with index .NetworkSettings.Networks "mtbank-ai-call-analytics_frontend"}}present{{end}}' "$MODEL_BOOTSTRAP_ID")" = present
   test "$(docker network inspect -f '{{len .Containers}}' "$GATEWAY_INGRESS_NETWORK")" = 1
   docker network inspect -f '{{range $id, $_ := .Containers}}{{println $id}}{{end}}' "$GATEWAY_INGRESS_NETWORK" | grep -Fx "$GATEWAY_FULL_ID"
   ```
6. Проверить effective logging boundary и runtime logs без вывода их содержимого:
   ```bash
   GATEWAY_NGINX_CONFIG="$(docker compose exec -T gateway nginx -T 2>&1)"
   test "$(printf '%s\n' "$GATEWAY_NGINX_CONFIG" | grep -Fxc '    access_log off;')" = 1
   test "$(printf '%s\n' "$GATEWAY_NGINX_CONFIG" | grep -Fxc '    error_log /dev/stderr crit;')" = 1
   printf '%s\n' "$GATEWAY_NGINX_CONFIG" | grep -Fq 'location = /analyze {'
   printf '%s\n' "$GATEWAY_NGINX_CONFIG" | grep -Fq 'location = /ws/transcribe {'
   printf '%s\n' "$GATEWAY_NGINX_CONFIG" | grep -Fq 'location ^~ /grafana/ {'
   unset GATEWAY_NGINX_CONFIG
   docker compose logs --no-color --since 10m openwebui pipelines gateway | uv run python scripts/assert_runtime_logs_clean.py
   ```

7. Проверить controlled/native API boundary без передачи API key и без публикации
   нового host port:
   ```bash
   API_ID="$(docker compose ps -q api)"
   API_IP="$(docker inspect -f '{{with index .NetworkSettings.Networks "mtbank-ai-call-analytics_application-internal"}}{{.IPAddress}}{{end}}' "$API_ID")"
   API_BASE_URL="http://$API_IP:8000" uv run python - <<'PY'
   import json
   import os
   from urllib.error import HTTPError
   from urllib.request import Request, urlopen

   def probe(path, method="GET", body=None):
       request = Request(os.environ["API_BASE_URL"] + path, data=body, method=method)
       if body is not None:
           request.add_header("Content-Type", "application/json")
       try:
           response = urlopen(request, timeout=3)
       except HTTPError as error:
           response = error
       return response.status, dict(response.headers), json.loads(response.read())

   status, _, body = probe("/missing")
   assert (status, body) == (404, {"detail": "Not Found"})
   status, headers, body = probe("/analyze")
   assert status == 405 and body == {"detail": "Method Not Allowed"}
   assert headers["Allow"] == "POST"
   status, _, body = probe("/analyze", "POST", b'{"url":"https://example.test/call.wav"}')
   assert status == 401 and set(body) == {"error"}
   PY
   ```
   Controlled application/transport errors имеют `{"error":...}`; native 404/405
   сохраняют Starlette `{"detail":...}` и `Allow`.
8. Для отдельной disposable PostgreSQL базы проверить migration guard и empty
   roundtrip. URL должен уже находиться в `MTBANK_TEST_DATABASE_URL`, иметь dialect
   `postgresql+asyncpg`, test-only path и не иметь query parameters:
   ```bash
   MTBANK_ALLOW_DESTRUCTIVE_DB_TESTS=1 uv run pytest tests/integration/test_postgres_migrations.py
   ```
   Guard сверяет `current_database()` до DDL; populated `analyses` должны fail-closed.

`model-bootstrap` завершает работу после однократного применения capability, а
`gateway` зависит от его `service_completed_successfully`. Не печатайте `.env`,
`MTBANK_TEST_DATABASE_URL`, `docker compose config` без `--quiet`, Bearer key или
любые значения runtime secrets.
