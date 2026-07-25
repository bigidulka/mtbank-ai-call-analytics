# MTBank AI Call Analytics

Решение тестового задания MTBank: публичный OpenWebUI Pipeline для транскрибации и аналитики русскоязычных банковских звонков.

- **Демо:** https://mtbank.arbitron.dev
- **Grafana:** https://mtbank.arbitron.dev/grafana/
- **Задание:** [`docs/assignment.md`](docs/assignment.md)
- **Финальные evidence:** [`release-evidence/final-115/`](release-evidence/final-115/)

## Архитектура

```text
OpenWebUI Pipeline / POST /analyze / WSS /ws/transcribe
                         │
                         ▼
RunPod GPU speech: faster-whisper large-v3-turbo
                  + pyannote Community-1
                  + timestamps / speaker roles
                         │
                         ▼
classifier ─ quality ─ compliance ─ summarizer
                         │
                         ▼
PostgreSQL ─ Trends agent ─ Prometheus ─ Grafana
```

App-plane работает на отдельном сервере через Docker Compose. Canonical speech работает на GPU Pod по authenticated HTTPS. Четыре LLM-агента используют OpenAI-compatible серверный gateway. Transcript считается недоверенным input; агенты получают bounded tools и typed output contracts.

## Что реализовано

- обязательный OpenWebUI `Pipeline` с upload WAV/MP3/OGG;
- local `faster-whisper` `large-v3-turbo` в GPU speech container;
- local pyannote Community-1 (`pyannote/speaker-diarization-community-1`) offline;
- timestamps, diarization и роли `Оператор` / `Клиент`;
- четыре независимых агента: classifier, quality, compliance, summarizer;
- deterministic aggregation, evidence IDs и typed schemas;
- `POST /analyze`, `POST /trends`, `WSS /ws/transcribe`;
- PostgreSQL persistence и lifecycle events;
- Prometheus + provisioned Grafana dashboard;
- JSON logging, readiness, bounded retries, circuit breaker, secret validation;
- public HTTPS deployment.

Собственный bounded Supervisor выбран вместо LangGraph: workflow фиксирован как `speech → 4 parallel agents → deterministic aggregation`, поэтому checkpoint-граф и human-in-the-loop state не нужны.

## Результаты ASR и диаризации

Корпус: пять authored synthetic русскоязычных банковских диалогов, 714.802 секунды. Реальные клиенты и production PII не использовались. Reference-разметка и provenance находятся в [`test_data/manifest.yaml`](test_data/manifest.yaml).

| Файл | Формат | Hz | Длительность | WER | DER | Role accuracy | GPU latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| `synthetic-credit-consultation` | WAV | 16000 | 239.546 | 8.89% | 19.11% | 84.20% | 9.797 с |
| `synthetic-card-complaint-telephone` | WAV | 8000 | 119.434 | 5.26% | 19.95% | 84.94% | 4.361 с |
| `synthetic-transfer-question` | MP3 | 16000 | 114.850 | 6.10% | 20.97% | 84.27% | 3.893 с |
| `synthetic-mobile-app-security` | OGG | 16000 | 113.194 | 4.82% | 21.29% | 83.70% | 4.097 с |
| `synthetic-deposit-consultation` | WAV | 16000 | 127.778 | 6.29% | 21.67% | 82.69% | 4.943 с |
| **Micro aggregate** | — | — | **714.802** | **6.76%** | **20.34%** | **83.99%** | **27.091 с** |

Полный машинно-читаемый результат: [`canonical-speech-evaluation.json`](release-evidence/final-115/canonical-speech-evaluation.json). Evaluator повторяет только transient `429/500/503/504`; schema, role-resolution и provider failures остаются fail-closed.

## Live SLA и бонусы

### Публичный файл ровно 5 минут

Публичный `POST /analyze` на финальном hostname обработал workload длительностью **300.0 с** за **35.979 с**, HTTP `200`, включая canonical speech, четыре агента, aggregation и persistence. Требование `<60 с` выполнено.

Evidence: [`public-five-minute-sla.json`](release-evidence/final-115/public-five-minute-sla.json).

### WebSocket real-time: +5

Публичный `wss://mtbank.arbitron.dev/ws/transcribe` отправил первый provisional transcript за **1428.098 мс**; measured p95/max = **1428.098 мс**, затем canonical reconciliation завершился за 18.416 с. Требование `<3 с` выполнено.

Provisional path использует bounded rolling 12-second canonical GPU windows с cadence 1.5 с и timeout 3 с. Финальный результат всегда повторно проходит полный canonical batch + четыре агента.

Evidence: [`websocket-p95.json`](release-evidence/final-115/websocket-p95.json).

### Grafana: +5

Dashboard `MTBank AI observability` содержит:

- Calls;
- Quality total;
- Top topics;
- Stage latency;
- Errors;
- Agent tokens.

Provisioning: [`monitoring/grafana/`](monitoring/grafana/).

### Trends agent: +5

`POST /trends` работает только по sanitized persisted analyses. Финальная проверка: 11 calls, 3 supporting calls, rate 27.27%, confidence 0.9, evidence-backed recommendation.

Evidence: [`trends-response.json`](release-evidence/final-115/trends-response.json).

## Как проверить демо

1. Открыть https://mtbank.arbitron.dev.
2. Войти или создать demo user в OpenWebUI.
3. Выбрать MTBank Pipeline.
4. Загрузить рекомендуемый файл [`test_data/synthetic/mobile-app-security-16k.ogg`](test_data/synthetic/mobile-app-security-16k.ogg).
5. Получить transcript с timestamps/roles, classification, quality checklist, compliance, summary и action items.
6. Для observability открыть `/grafana/` с предоставленными reviewer credentials.

REST API требует `Authorization: Bearer <MTBANK_API_KEY>` и принимает ровно один source: multipart `file` или JSON `{"url":"https://..."}`. Полный contract: [`docs/api.md`](docs/api.md).

## Локальный запуск

```bash
cp .env.example .env
# Заполнить все секреты и gateway/model values.
# MTBANK_WORKFLOW__CODE_SHA=$(git rev-parse HEAD)

HF_TOKEN=... uv run python scripts/provision_speech_models.py \
  --artifact-root models/artifacts \
  --output-manifest models/manifest.json \
  --cache-dir .cache/mtbank-speech-models

docker compose up --build --wait
```

Требования: Docker Compose, FFmpeg, около 16 GB RAM и 12 GB disk. GPU overlay и split RunPod deployment описаны в [`docs/operations.md`](docs/operations.md) и [`deploy/runpod/README.md`](deploy/runpod/README.md).

## Воспроизводимая проверка

```bash
uv run --offline --no-sync ruff check .
uv run --offline --no-sync ruff format --check .
uv run --offline --no-sync pyright
uv run --offline --no-sync pytest -m "not integration and not real_llm and not gpu"
uv lock --check
uv lock --check --directory services/speech
docker compose -f docker-compose.yml config --quiet
```

Canonical evaluator:

```bash
export MTBANK_RUNPOD_SPEECH_BEARER_KEY='...'
uv run --offline --no-sync python scripts/evaluate_canonical_speech.py \
  --base-url 'https://<pod-id>-8010.proxy.runpod.net' \
  --api-key-env MTBANK_RUNPOD_SPEECH_BEARER_KEY \
  --output release-evidence/canonical-speech-evaluation.json
```

## Ограничения

- Public demo предназначен только для synthetic/no-PII audio.
- RunPod GPU — оплачиваемый ephemeral compute; во время reviewer window Pod должен оставаться запущенным.
- DER и role accuracy измерены на authored synthetic corpus и не являются банковским production benchmark.
- Rolling WebSocket partials provisional; authoritative result появляется только после canonical reconciliation.
- Полноценный банковский production требует отдельной модели угроз, DLP/KMS, retention approvals, HA, private networking и formal model governance.
