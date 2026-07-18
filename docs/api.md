# API contract

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
