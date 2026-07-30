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
retrieval tools, один terminal submit, configurable максимум шесть turns (hard cap
восемь), total/repeated tool-call guards и input/output/cost budgets. Runtime
принимает только function calls: text completion, unknown or
duplicate call, невалидные arguments, неразрешённый tool, post-terminal call,
budget/deadline exhaustion и невалидный terminal output завершают run typed failure
без partial success.

Tool registry не исполняет model-provided code, shell, filesystem path, arbitrary
HTTP или MCP. Он содержит заранее внедрённые typed handlers и генерирует strict
OpenAI function schemas. Наблюдения ограничены по размеру, canonical JSON и явно
помечены `untrusted_tool_result` перед следующим model turn.

На каждом turn модель видит все server-allowlisted read-only tools и terminal tool,
сама выбирает zero/multiple calls через `tool_choice=auto`. Независимые read-only
calls выполняются параллельно, observations возвращаются в provider order. Required
evidence остаётся terminal invariant: retrieval должен завершиться в предыдущем turn,
поэтому retrieval и terminal submit в одном response не дают обход авторизации.

Provider streaming собирает fragmented tool calls только внутри trusted runtime; наружу
выходят typed text deltas и validated completed response. Stream закрывается при normal
completion, failure и cancellation; semaphore/circuit lifecycle завершается до terminal
event. Lifecycle events и returned trajectory содержат только IDs, hashes, tool names,
statuses, usage и latency. Prompt, transcript, tool arguments, observation body, raw
provider body и API key туда не попадают. Prompt registry отвергает traversal и symlink
escape и хеширует canonical text plus reviewed policy/tool-schema inputs.

## Capability и release gate

`CapabilityProbeRunner` проверяет native tools, strict schema, multi-call order,
tool-result serialization, system role, streaming cancellation/usage и limits.
Offline unit tests передают явный scripted provider из tests; runtime не имеет test
provider или fallback. Live probe требует credentials и fail-closed при любой
неподтверждённой capability.

Text assistant использует отдельный bounded multi-turn loop с static safe tools,
streaming final-answer deltas, authenticated SSE и synchronous OpenWebUI generator.
Промежуточный provider text tool-turns, arguments, observations и chain-of-thought не
публикуются. После tool selection запускается отдельный no-tools final turn; его deltas
идут с bounded safety holdback, а terminal `stop`, exact text и общий лимит проверяются до
`done`. Expensive nested Trends допускается один раз, restricted reviewed taxonomy, small cohorts suppressed,
его usage/cost включаются в parent budget. Buffered `POST /assistant` сохранён как
compatibility adapter поверх stream.

Реальный cloud capability probe и smoke/E2E с configured gateway обязательны после
изменений runtime; локальные unit tests не являются заменой этому доказательству.
