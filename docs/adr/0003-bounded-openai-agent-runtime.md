# ADR 0003: bounded OpenAI-compatible agent runtime

- **Статус:** принято для implementation slice; real-cloud release evidence заблокирован отдельным gate.
- **Дата:** 2026-07-16

## Решение

Будущие четыре business agents используют только один configurable HTTPS cloud
OpenAI-compatible gateway через официальный `openai==2.11.0`, `AsyncOpenAI` и
Chat Completions. SDK retries выключены (`max_retries=0`): bounded retries,
backoff, `Retry-After`, semaphore и circuit breaker принадлежат runtime.
Не существует Anthropic adapter, local fallback, Responses API или model fallback.

Каждый `AgentSpec` immutable и содержит exact model/policy/prompt versions,
SHA-256 prompt bundle, typed terminal output, allowlist read-only tools, required
retrieval tools, один terminal submit, максимум три turns и input/output/cost
budgets. Runtime принимает только function calls: text completion, unknown or
duplicate call, невалидные arguments, неразрешённый tool, post-terminal call,
budget/deadline exhaustion и невалидный terminal output завершают run typed failure
без partial success.

Tool registry не исполняет model-provided code, shell, filesystem path, arbitrary
HTTP или MCP. Он содержит заранее внедрённые typed handlers и генерирует strict
OpenAI function schemas. Наблюдения ограничены по размеру, canonical JSON и явно
помечены `untrusted_tool_result` перед следующим model turn.

Lifecycle events и returned trajectory содержат только IDs, hashes, tool names,
statuses, usage и latency. Prompt, transcript, tool arguments, observation body,
raw provider body и API key туда не попадают. Prompt registry отвергает traversal и
symlink escape и хеширует canonical text plus reviewed policy/tool-schema inputs.

## Capability и release gate

`CapabilityProbeRunner` проверяет native tools, strict schema, multi-call order,
tool-result serialization, system role, streaming cancellation/usage и limits.
Offline unit tests передают явный scripted provider из tests; runtime не имеет test
provider или fallback. Live probe требует credentials и fail-closed при любой
неподтверждённой capability.

В этой сессии credentials намеренно отсутствуют. Реальный cloud capability probe и
smoke/E2E с configured gateway обязательны перед release; локальные unit tests не
являются заменой этому доказательству.
