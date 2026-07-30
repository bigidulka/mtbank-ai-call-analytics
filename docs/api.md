# API contract

## Assistant

`POST /assistant` и `POST /assistant/stream` используют тот же bearer token, message
до 2000 символов и максимум восемь history messages. Buffered endpoint возвращает
`{"answer","model_id"}`. Streaming endpoint возвращает SSE `v=1`:

- `start`;
- `progress` только с safe phase и allowlisted tool label;
- `delta` только с visible final-answer text;
- `done`;
- `error` с публичным code/message/retryable после уже отправленных headers.

`id` и `sequence` возрастают локально. Provider/tool IDs, arguments, observations,
prompt, transcript, model metadata и hidden reasoning не входят в public stream.
Disconnect отменяет provider/tool work; partial text не сопровождается `done` после
ошибки. OpenWebUI Pipeline проверяет event order/schema/byte-count и отдаёт deltas как
generator. Ответ целиком валидируется и ограничивается 8000 символами до первого public
delta; tool-protocol markers отклоняются. Trends tool допускает один reviewed-topic query
на assistant request и скрывает малые cohorts/точные counts. Buffered `POST /assistant`
сохранён для compatibility.

## Analyze

`POST /analyze` остаётся internal-only до утверждения внешнего ingress. Требуется
`Authorization: Bearer <MTBANK_API_KEY>` и ровно один из источников:

- `multipart/form-data` с единственным `file` (`audio/wav`, `audio/x-wav`,
  `audio/mpeg`, `audio/ogg`);
- `application/json` ровно вида `{"url":"https://..."}`.

URL и file проходят один shared workflow. Unit/contract tests с injected local fake
проверяют эквивалентность transport semantics, а не реальную ASR, retrieval или cloud
agent execution. URL ingestion выполняет SSRF-safe fetch только в configured workflow.

Успешный response имеет `transcript`, `classification`, `quality_score`,
`compliance`, `summary`, `action_items`, `grounding`, `meta`. Evidence IDs должны
ссылаться на публичные transcript segments; `quality_score.total` и
`compliance.passed` определяются deterministic aggregation, не LLM.

Ошибки приложения используют `{"error":{"code","message","request_id",
"retryable"}}`; нативные 404/405 Starlette сохраняют `detail` и `Allow`. Полный
OpenAPI доступен только из API container network. Реальный release E2E описан в
[operations.md](operations.md) и не заменяется fake transport test.
