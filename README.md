# MTBank AI Call Analytics

Production-oriented решение тестового задания MTBank: русскоязычная транскрибация звонков, диаризация, четыре независимых LLM-агента, OpenWebUI Pipeline, REST API и бонусные WebSocket/Grafana/trends-компоненты.

Оригинальный текст задания сохранён без изменений в [`docs/assignment.md`](docs/assignment.md).

## Статус

Локально реализованы и покрыты тестами:

- обязательный OpenWebUI Pipeline и безопасная передача загруженного файла;
- общий workflow для Pipeline и `POST /analyze`, включая fail-closed readiness PostgreSQL + speech;
- canonical speech path `Groq whisper-large-v3-turbo → local pinned pyannote Community-1 → deterministic overlap merge → role resolution`;
- четыре независимых LLM-агента, versioned taxonomy, quality rubric и compliance policy;
- PostgreSQL persistence, JSON telemetry, Prometheus/Grafana, bounded streaming и evidence-backed Trends;
- Compose runtime wiring `speech healthy → api healthy → pipelines healthy`, gateway `/analyze`, `/trends`, `/ws/transcribe`, `/grafana/`;
- manifest-pinned Community-1 metadata, controlled canonical-evaluation runner и пять русских synthetic-звонков с reference-разметкой;
- Compose overlays для opt-in GPU и WebSocket runtime, а также reproducible benchmark harnesses;
- offline contract tests, которые не подменяют runtime/release evidence.

Не выдаются за готовое доказательство:

- canonical corpus-wide Groq+pyannote WER/DER/role/speaker-attributed metrics;
- GPU benchmark, GPU WebSocket p95 и five-minute `<60 с` SLA;
- final immutable Git SHA/signed image release bundle и итоговый verified competitor score.

## Архитектура

```text
Browser / OpenWebUI / REST / WebSocket
                │
                ▼
        pinned OpenWebUI Pipeline
                │
                ▼
        AnalyzeCallUseCase
                │
       ┌────────┴────────┐
       │                 │
 secure ingestion   speech service
 file / URL          FFmpeg normalization
 SSRF protection     Groq whisper-large-v3-turbo, один ASR request
                     verbose JSON word+segment timestamps passthrough
                     local pyannote Community-1 diarization
                     deterministic overlap merge and explicit role resolution
       │                 │
       └────────┬────────┘
                ▼
      immutable TranscriptSnapshot
                │
      ┌─────────┼──────────┬───────────┐
      ▼         ▼          ▼           ▼
 classifier  quality   compliance  summarizer
 separate    separate  separate    separate
 prompt      prompt    prompt      prompt
 tools       tools     tools       tools
 model loop  model     model       model
      └─────────┴──────────┴───────────┘
                │
                ▼
 deterministic aggregation
 quality total / compliance passed
                │
                ▼
 PostgreSQL + redacted events + metrics
                │
                ▼
 OpenWebUI Markdown / REST JSON / trends
```

Подробности: [`docs/architecture.md`](docs/architecture.md), ADR [`0002`](docs/adr/0002-canonical-batch-speech.md), [`0003`](docs/adr/0003-bounded-agent-runtime.md) и [`0004`](docs/adr/0004-observability-streaming-and-trends.md).

## Почему собственный Supervisor

Workflow короткий и ограниченный: speech → четыре параллельных агента → deterministic aggregation. Для него не требуются долгоживущие graph checkpoints или human-in-the-loop state, поэтому собственный typed Supervisor проще проверять и тестировать, чем универсальный LangGraph runtime.

Каждый обязательный агент имеет отдельные:

- reviewed `prompt.md`;
- model invocation и sanitized trajectory;
- read-only tool allowlist;
- terminal Pydantic schema;
- deadline, turn, token и cost budget;
- обязательное получение transcript evidence и policy/rubric data.

LLM не задаёт `quality_score.total` и `compliance.passed`: эти значения рассчитывает deterministic aggregator.

## LLM

Runtime использует официальный OpenAI SDK через configurable Chat Completions boundary.
Для development capability probes и local four-agent smoke применялся локальный
OpenAI-compatible CLIProxyAPI на `http://127.0.0.1:8317/v1`: подтверждены exact identity
`gpt-5.6-luna`, `gpt-5.6-terra`, `gpt-5.6-sol`, typed tool calls, usage и required
terminal schemas. Это не final gateway attestation и не заменяет credentials/trace
artifact для release.

Агенты получают transcript и policy только через локальные bounded tools. Remote MCP
Connectors, Gmail, Calendar и Drive не используются: они не входят в задачу и расширили
бы credential boundary.

## Speech pipeline

Canonical batch flow:

> Формальное отклонение от формулировки задания: runtime намеренно использует Groq ASR,
> а не требуемый local `faster-whisper`. `faster-whisper`, `openai-whisper` и ASR fallback
> не реализованы; это не выдаётся за соответствие этому требованию.

```text
FFmpeg decode/normalize
→ Groq OpenAI-compatible `/openai/v1/audio/transcriptions`, `whisper-large-v3-turbo`, один request
→ `language=ru`, `temperature=0`, `response_format=verbose_json`, word+segment timestamps
→ timestamps passthrough without local alignment model
→ local `pyannote/speaker-diarization-community-1` diarization
→ deterministic maximum-overlap word speaker assignment
→ explicit role mapping
→ immutable TranscriptSnapshot
```

Public transcript содержит:

```json
{
  "id": "segment-uuid",
  "original_speaker_id": "SPEAKER_00",
  "speaker": "Оператор",
  "role_confidence": 0.96,
  "start": 0.0,
  "end": 4.2,
  "text": "Добрый день..."
}
```

Первый говорящий не считается оператором автоматически. Если полное role mapping невозможно, completed response не создаётся. Low-confidence полное mapping допускается только с `needs_review=true`.

Поддержаны WAV, MP3 и OGG. Default CPU profile использует `int8`; GPU override — FP16 и NVIDIA runtime.

### Speech provider boundary

Единственный canonical ASR provider — Groq. Speech container получает `GROQ_API_KEY` только из secret environment, отправляет raw normalized audio только на `https://api.groq.com/openai/v1/audio/transcriptions` через `Authorization: Bearer`, не передаёт key в URL/query и не пишет key или body в logs. `httpx` отключает proxy environment и redirects, а response body ограничен. Для этого remote disclosure требуется отдельное privacy/legal approval; Replicate, Deepgram, pyannoteAI и local-ASR fallback не поддерживаются.

Provisioning выполняется отдельно в окружении `services/speech`, никогда не в runtime container: после approved HF credential запустите `PYTHONPATH=src:. services/speech/.venv/bin/python scripts/provision_speech_models.py --artifact-root models/artifacts --output-manifest models/manifest.json --cache-dir tmp/huggingface-cache --component diarization`. Manifest содержит только checked local Community-1 artifact, не перезаписывает artifacts/manifest и не выводит token. `HF_TOKEN` нужен только для controlled provisioning gated Community-1; runtime всегда offline и токен не получает.

## Тестовые аудио

Corpus генерируется по разрешённому заданием варианту Edge TTS:

```bash
uv run --with edge-tts --with pydub python scripts/generate_synthetic_dataset.py
uv run python scripts/validate_test_manifest.py --require-release-corpus
```

Голоса:

- оператор: `ru-RU-SvetlanaNeural`;
- клиент: `ru-RU-DmitryNeural`.

Все сценарии написаны для проекта, не содержат реальных клиентов и имеют exact reference segments с ролями и временными метками.

| ID | Формат | Частота | Длительность | Сценарий |
|---|---:|---:|---:|---|
| `synthetic-credit-consultation` | WAV | 16 kHz | 239.546 с | подробная консультация по кредиту |
| `synthetic-card-complaint-telephone` | WAV | 8 kHz | 119.434 с | спорное списание в банкомате, телефонное качество |
| `synthetic-transfer-question` | MP3 | 16 kHz | 114.850 с | задержка перевода |
| `synthetic-mobile-app-security` | OGG | 16 kHz | 113.194 с | мошеннический звонок и безопасность |
| `synthetic-deposit-consultation` | WAV | 16 kHz | 127.778 с | консультация по вкладу |
| **Итого** | WAV/MP3/OGG | 8/16 kHz | **714.802 с** | пять двухголосых звонков |

Manifest и SHA-256: [`test_data/manifest.yaml`](test_data/manifest.yaml). Reference transcripts: [`test_data/references/`](test_data/references/).

### Метрики

Нет corpus-wide canonical WER/DER/role/speaker-attributed metrics и нет runtime GPU
proof. `scripts/evaluate_canonical_speech.py` воспроизводимо вычисляет metrics только
через configured canonical `/v1/transcribe` и сохраняет privacy-safe hashes, revisions,
latency и results во внешний ignored artifact. Direct `scripts/evaluate_groq_stt.py`
является noncanonical Groq-only ASR sanity evaluator и не заменяет этот запуск.

`scripts/run_websocket_benchmark.py` — последовательный paced diagnostic client; его
CPU output diagnostic-only и не является GPU evidence. GPU evidence создаёт только
`scripts/run_gpu_speech_benchmark.py` после фактической проверки NVIDIA runner и
завершённого controlled WebSocket run. Ни один из harnesses не доказывает five-minute
`<60 с` SLA до реального наблюдаемого запуска.

## OpenWebUI Pipeline

[`pipeline.py`](pipeline.py) — обязательный legacy Pipeline entrypoint.

Browser attachment flow:

1. OpenWebUI сохраняет файл и передаёт metadata descriptor в `inlet()`.
2. Pipeline создаёт короткоживущую HMAC-signed attachment reference.
3. Перед скачиванием проверяются user ownership и signed subject.
4. Content загружается через authoritative OpenWebUI file API.
5. Проверяются actual bytes, SHA-256, MIME, magic и size.
6. Проверенные bytes передаются внутреннему `/analyze`.

Pipeline не получает model-egress и не содержит второй копии business workflow. Только API service вызывает configured LLM gateway.

## REST API

`POST /analyze` защищён Bearer API key и принимает ровно один источник.

Файл:

```bash
curl -X POST http://127.0.0.1:${OPENWEBUI_PORT:-3000}/analyze \
  -H "Authorization: Bearer $MTBANK_API_KEY" \
  -F "file=@test_data/synthetic/credit-consultation-16k.wav"
```

URL:

```bash
curl -X POST http://127.0.0.1:${OPENWEBUI_PORT:-3000}/analyze \
  -H "Authorization: Bearer $MTBANK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.org/call.ogg"}'
```

URL ingestion блокирует loopback, private/link-local/reserved/metadata addresses, повторно проверяет каждый redirect и соединяется только с проверенным DNS IP при сохранении исходных Host/SNI.

Success response содержит обязательные поля задания:

```text
transcript
classification
quality_score.total + checklist
compliance.passed + issues
summary
action_items
meta
```

Controlled application errors имеют стабильную форму:

```json
{"error":{"code":"invalid_input","message":"Некорректный запрос.","request_id":"...","retryable":false}}
```

Native Starlette routing errors намеренно сохраняют `{"detail":"..."}`: `404` и
`405` не маскируются application envelope, а `405` сохраняет заголовок `Allow`.

Контракты и errors: [`docs/api.md`](docs/api.md).

## PostgreSQL migrations

Alembic выполняется только online. `0002_contract_convergence` работает fail-closed,
проверяет live PostgreSQL catalog и не поддерживает `--sql`. Если legacy `analyses`
populated, migration останавливается до DDL/DML: оператор должен отдельно проверить и
удалить legacy payload по утверждённой privacy-процедуре. Автоматический перенос
непроверенных данных намеренно отсутствует.

## Запуск

Требования:

- Docker и Docker Compose;
- NVIDIA Container Toolkit только для GPU profile;
- заполненный `.env`;
- локально provisioned speech artifacts в `models/`.

```bash
cp .env.example .env
# Заполнить secrets и model paths, не добавлять .env в Git.
docker compose up --build --wait
```

OpenWebUI:

```text
http://127.0.0.1:${OPENWEBUI_PORT:-3000}
```

GPU profile:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile gpu up --build --wait
```

В `.env` обязательны как минимум:

```text
POSTGRES_PASSWORD
GROQ_API_KEY
MTBANK_API_KEY
MTBANK_ATTACHMENT_SIGNING_KEY
PIPELINES_API_KEY
WEBUI_ADMIN_EMAIL
WEBUI_ADMIN_PASSWORD
WEBUI_SECRET_KEY
GRAFANA_ADMIN_PASSWORD
MTBANK_AGENT_RUNTIME__GATEWAY__BASE_URL
MTBANK_AGENT_RUNTIME__GATEWAY__API_KEY
MTBANK_AGENT_RUNTIME__GATEWAY__MODELS__DEFAULT_MODEL
MTBANK_WORKFLOW__CODE_SHA
```

Полная operational инструкция: [`docs/operations.md`](docs/operations.md).

## Offline validation

Следующие offline validation commands выполняются без environment sync, package install, Docker build/up и real credentials:

```bash
uv run --offline --no-sync pytest -m "not real_llm and not gpu and not integration"
uv run --offline --no-sync ruff check .
uv run --offline --no-sync pyright
uv lock --check
uv lock --check --directory services/speech
docker compose -f docker-compose.yml config --quiet
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile gpu config --quiet
CLOUDFLARE_TUNNEL_TOKEN_FILE=/tmp/mtbank-cloudflared-token \
  docker compose -f docker-compose.yml -f docker-compose.cloudflare.yml config --quiet
# После безопасной передачи credentials из approved source, без значений в Git/argv:
python scripts/provision_cloudflare_tunnel.py prepare
git diff --check
```

Последняя команда — Cloudflare dry-run: без `--apply` она не создаёт tunnel, token file,
DNS record или Docker service. Controlled `prepare --apply` и `publish --apply` выполнены:
connector healthy, proxied DNS создан, HTTPS hostname отвечает через Cloudflare. Названия
required process-environment inputs и rollback flow приведены в [`docs/operations.md`](docs/operations.md).

Последний полный offline result: **507 passed, 70 deselected**, 1 известное
Starlette deprecation warning; Ruff clean; Pyright 0 errors, 0 warnings. Focused
Cloudflare/Compose/release-static contracts: 32 passed. Локальные Docker images успешно
собраны: `mtbank-ai-speech-cpu:local` (`sha256:d49698efed9c17000c57509e47e24242165b818be4798a1580fa994747c628fd`)
и `mtbank-ai-foundation:local` (`sha256:5deb3e110428641529da869c138f23982e7f2884167fd1d3d2578376bbdde304`).
Build использовал `--network host` из-за default Docker build network. Full Compose затем
поднят с successful migrations и healthy PostgreSQL, speech, API, Pipelines, OpenWebUI,
gateway, Prometheus, Tempo и Grafana.

Для PostgreSQL integration suite используйте только отдельную disposable test database:

```bash
MTBANK_TEST_DATABASE_URL='postgresql+asyncpg://.../mtbank_test_local' \
MTBANK_ALLOW_DESTRUCTIVE_DB_TESTS=1 \
uv run pytest tests/integration/test_postgres_migrations.py
```

## JSON-логи и privacy

Lifecycle записывает sanitized input/output metadata каждого агента:

- agent/model/prompt/policy versions;
- model and local tool correlation IDs;
- usage, latency, status и hashes;
- terminal schema validation status.

Raw audio, transcript, prompts, provider response, Authorization/Cookie headers и API keys не пишутся в default logs/traces. Содержимое transcript передаётся LLM только как untrusted tool observation после redaction.

Подробнее: [`docs/privacy.md`](docs/privacy.md) и [`docs/threat-model.md`](docs/threat-model.md).

## Runtime wiring и preflight

API `/health/ready` fail-closed проверяет PostgreSQL и configured speech `/health/ready`;
в Compose зависимости построены как `speech healthy → api healthy → pipelines healthy`.
Gateway публикует только loopback host port OpenWebUI и exact routes `/analyze`,
`/trends`, `/ws/transcribe`, `/grafana/`; остальные application/monitoring services остаются
internal-only. Grafana configured для `/grafana/`, `serve_from_sub_path`, healthcheck и
separate dashboard JSON mount.

Cloudflare Tunnel подготовлен и опубликован для hostname
`https://cloud.cloud-tunnel-mega-obx1.space`: connector подключён, proxied CNAME создан,
OpenWebUI, authenticated `/analyze`/`/trends` и `/grafana/` доступны по HTTPS. Protected
connector token не хранится в Git. Порядок и rollback описаны в
[`docs/operations.md`](docs/operations.md).

## Бонусы

### WebSocket real-time

`/ws/transcribe` использует internal `/v1/stream` с Groq-only rolling transcription: bounded 16 kHz mono PCM ring временно записывается в WAV, один bounded Groq request даёт provisional partial, а следующий result использует common-prefix stabilization. Rolling provider timeout/429/failure не ретраится и не закрывает session; rolling budgets прекращают только provisional calls. На end обычный canonical Groq+pyannote batch path reconciles полный audio. Нет local faster-whisper, Silero, Replicate, Deepgram или других ASR fallback. Controlled CPU-host run через настоящий authenticated `/ws/transcribe` дал provisional p95 0.518 с по 4 updates из 60 wall-clock frames; evidence — `release/live-websocket-benchmark.json`. Compose defaults остаются disabled, reconciliation в этом latency run не запускалась, а GPU p95 не заявляется.

### Grafana

Provisioned dashboard содержит:

- количество звонков;
- распределение `quality_score`;
- топ тематик;
- latency/errors/agent token metrics.

Конфигурация находится в [`monitoring/`](monitoring/).

### Trends

Protected `/trends` строит evidence-backed patterns минимум по пяти уникальным сохранённым runs. Отдельный bounded Trends agent проходит trajectory `trend_aggregate_query` → `trend_evidence_retrieve` → `submit_trend`; denominator, numerator, rate, filter и supporting run IDs finalizes trusted code. Direct SQL и unsupported free-form claims запрещены. Live E2E по пяти persisted completed calls вернул deterministic `5/5 = 1.0`, пять supporting run IDs и provenance `agent_id=trends`; route доступен через authenticated public HTTPS.

## Живое демо

Живое демо: **https://cloud.cloud-tunnel-mega-obx1.space**. Проверены Cloudflare HTTPS,
OpenWebUI login page, authenticated REST `/analyze`, real OpenWebUI attachment Pipeline,
Grafana health/dashboard, Prometheus scrape и authenticated Trends. Public 30-секундный
canonical analysis завершился с HTTP 200 за 60.287 с. Это не five-minute `<60 с` SLA.

## Известные ограничения и blockers

- Synthetic-only remote-audio authorization зафиксирован в `release/synthetic-remote-audio-approval.json`; реальные клиентские и production данные явно исключены и требуют отдельного organizational approval.
- Нет corpus-wide canonical WER/DER/role/speaker-attributed metrics или five-minute end-to-end SLA claim.
- Controlled Groq rolling streaming E2E дал CPU-host provisional p95 0.518 с по 4 updates, но canonical reconciliation в этом run не запускалась; GPU host/evidence и GPU WebSocket p95 отсутствуют, Compose default остаётся disabled.
- Frozen 44-cohort static benchmark выполнен, но все comparative scores остаются `unknown`; нет immutable final Git SHA/signed image release bundle, текущие image IDs относятся к локальной сборке.
- Deployment использует host-network egress relay из-за неработающего outbound NAT Docker bridge на данном хосте; это operational workaround, а не переносимая Compose default-конфигурация.

## Атрибуция

- Исходное задание: [`ZubikIT/mtbank-ai-hiring`](https://github.com/ZubikIT/mtbank-ai-hiring).
- Synthetic speech: Microsoft Edge TTS voices `ru-RU-SvetlanaNeural` и `ru-RU-DmitryNeural`, использование прямо предложено исходным заданием.
- Speech stack: Groq Whisper API and local pyannote.audio Community-1; only the local diarization artifact is manifest-pinned.
- Архитектурные идеи из других решений не копировались как код без license/provenance review.

Дополнительные документы:

- [`docs/evals.md`](docs/evals.md)
- [`docs/release-checklist.md`](docs/release-checklist.md)
- [`docs/competitive-analysis.md`](docs/competitive-analysis.md)
