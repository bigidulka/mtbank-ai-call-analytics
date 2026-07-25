# MTBank AI Call Analytics

AI-система для транскрибации и многоагентного анализа русскоязычных банковских звонков через обязательный **OpenWebUI Pipeline**.

[![CI](https://github.com/bigidulka/mtbank-ai-call-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/bigidulka/mtbank-ai-call-analytics/actions/workflows/ci.yml)

- **OpenWebUI demo:** https://mtbank.arbitron.dev
- **Grafana:** https://mtbank.arbitron.dev/grafana/
- **Исходное ТЗ:** [`docs/assignment.md`](docs/assignment.md)
- **Машинно-читаемые результаты:** [`release-evidence/final-115/`](release-evidence/final-115/)

## Быстрая проверка

1. Открыть [публичный OpenWebUI](https://mtbank.arbitron.dev).
2. Выбрать модель **MTBank Attachment Probe**.
3. Написать текстовый вопрос — отдельный bounded LLM-agent объяснит форматы, API, streaming, Grafana и сценарий проверки с учётом истории диалога.
4. Загрузить [`test_data/synthetic/mobile-app-security-16k.ogg`](test_data/synthetic/mobile-app-security-16k.ogg).
5. Получить единый результат: transcript с timestamps и ролями, classification, quality checklist, compliance, summary, action items.
6. Открыть [Grafana](https://mtbank.arbitron.dev/grafana/) для метрик звонков, качества и тематик.

## Архитектура

```text
OpenWebUI Pipeline ─┬─ POST /analyze
                    ├─ WSS /ws/transcribe
                    └─ upload WAV / MP3 / OGG
                              │
                              ▼
       faster-whisper large-v3-turbo + pyannote Community-1
              timestamps + diarization + role resolution
                              │
                              ▼
       ┌────────────┬───────────┬────────────┬────────────┐
       │ classifier │  quality  │ compliance │ summarizer │
       └────────────┴───────────┴────────────┴────────────┘
                              │
                              ▼
       deterministic aggregation + PostgreSQL persistence
                              │
                              ▼
                 Trends + Prometheus + Grafana
```

Один canonical workflow обслуживает OpenWebUI и REST API. Четыре агента запускаются параллельно через bounded Supervisor, используют retrieval-first Tools/Actions и завершаются typed terminal output. Итоговый score и compliance агрегируются детерминированно, а evidence привязывается к segment IDs.

Собственный Supervisor выбран вместо LangGraph: граф фиксирован как `speech → 4 parallel agents → aggregation`, поэтому отдельный checkpoint/state-machine слой не нужен.

## Покрытие ТЗ

| Критерий задания | Реализация | Проверяемое доказательство |
|---|---|---|
| Pipeline архитектура — 25 | Настоящий OpenWebUI Pipeline, attachment flow, общий workflow для UI и API | [`pipeline.py`](pipeline.py), [`docker-compose.yml`](docker-compose.yml) |
| ASR качество — 20 | local `faster-whisper` `large-v3-turbo`, local pyannote Community-1, timestamps, роли, WAV/MP3/OGG | [`canonical-speech-evaluation.json`](release-evidence/final-115/canonical-speech-evaluation.json) |
| Multi-Agent — 25 | classifier, quality, compliance, summarizer; независимые prompts/tools/contracts | [`src/mtbank_ai/agents/`](src/mtbank_ai/agents/), [`workflow/analysis.py`](src/mtbank_ai/workflow/analysis.py) |
| Код и архитектура — 15 | FastAPI, PostgreSQL, typed schemas, retries, circuit breaker, tests, `.env.example` | [CI](https://github.com/bigidulka/mtbank-ai-call-analytics/actions/workflows/ci.yml), [`tests/`](tests/) |
| Документация — 10 | Схема, demo flow, модели, метрики, запуск, API | этот README, [`docs/`](docs/) |
| Живое демо — 5 | Публичный HTTPS OpenWebUI и измеренный 5-minute end-to-end workload | [`public-five-minute-sla.json`](release-evidence/final-115/public-five-minute-sla.json) |
| Bonus: WebSocket — 5 | Rolling provisional transcription + canonical reconciliation; live diagnostic first partial `<3 с` | [`websocket-p95.json`](release-evidence/final-115/websocket-p95.json) |
| Bonus: Grafana — 5 | Calls, Quality total, Top topics, latency, errors, agent tokens | [`mtbank-overview.json`](monitoring/grafana/dashboards/mtbank-overview.json) |
| Bonus: Trends — 5 | Анализ нескольких persisted calls с evidence-backed recommendation | [`trends-response.json`](release-evidence/final-115/trends-response.json) |

## Метрики ASR и диаризации

Корпус: пять authored synthetic банковских диалогов, **714.802 секунды**, три формата, 8/16 kHz, эталонные transcript/roles и SHA-256 provenance в [`test_data/manifest.yaml`](test_data/manifest.yaml).

| Файл | Формат | Hz | Длительность | WER | DER | Role accuracy | GPU latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| `synthetic-credit-consultation` | WAV | 16000 | 239.546 с | 8.89% | 19.11% | 84.20% | 9.797 с |
| `synthetic-card-complaint-telephone` | WAV | 8000 | 119.434 с | 5.26% | 19.95% | 84.94% | 4.361 с |
| `synthetic-transfer-question` | MP3 | 16000 | 114.850 с | 6.10% | 20.97% | 84.27% | 3.893 с |
| `synthetic-mobile-app-security` | OGG | 16000 | 113.194 с | 4.82% | 21.29% | 83.70% | 4.097 с |
| `synthetic-deposit-consultation` | WAV | 16000 | 127.778 с | 6.29% | 21.67% | 82.69% | 4.943 с |
| **Micro aggregate** | — | — | **714.802 с** | **6.76%** | **20.34%** | **83.99%** | **27.091 с** |

Evaluator сохраняет audio/reference/hypothesis hashes, model revisions и component artifact hashes. Transient retry разрешён только для `429/500/503/504`; schema и role-resolution failures остаются fail-closed.

## Производительность и бонусы

| Проверка | Результат | Evidence |
|---|---:|---|
| Публичный end-to-end анализ файла **300.0 с** | **35.979 с**, HTTP 200 | [`public-five-minute-sla.json`](release-evidence/final-115/public-five-minute-sla.json) |
| Первый WebSocket provisional transcript, live diagnostic | **1428.098 мс** | [`websocket-p95.json`](release-evidence/final-115/websocket-p95.json) |
| Canonical reconciliation после streaming | **18.416 с** | [`websocket-p95.json`](release-evidence/final-115/websocket-p95.json) |
| Trends | 11 calls, 3 supporting calls, confidence 0.9 | [`trends-response.json`](release-evidence/final-115/trends-response.json) |

WebSocket partials создаются bounded rolling GPU windows; финальный ответ всегда повторно проходит canonical full-batch speech и четыре аналитических агента.

## Что отличает решение

- **LLM-помощник внутри demo:** отдельный bounded text-agent использует историю чата, отвечает через тот же OpenAI-compatible gateway и не зависит от speech runtime.
- **Не FastAPI-заглушка:** production flow проходит через обязательный OpenWebUI Pipeline.
- **Один результат для UI и API:** отсутствует расхождение между demo и backend workflow.
- **Не только WER:** отдельно измеряются DER, role accuracy и speaker-attributed WER.
- **Grounded agents:** classification, quality и summary ссылаются на конкретные transcript segment IDs.
- **Fail-closed contracts:** неверная schema, unresolved roles, provider drift и неподдерживаемое audio не превращаются в частичный «успех».
- **Проверка attachment boundary:** owner, signed reference, MIME, magic bytes, size и SHA-256 сверяются до анализа.
- **Наблюдаемость из коробки:** Prometheus + provisioned Grafana + agent token/stage metrics.
- **Версионированный evidence:** финальные JSON-файлы связаны SHA-256 manifest в [`release-evidence/final-115/manifest.json`](release-evidence/final-115/manifest.json).

## REST API

```bash
curl -X POST https://mtbank.arbitron.dev/analyze \
  -H "Authorization: Bearer $MTBANK_API_KEY" \
  -F "file=@test_data/synthetic/mobile-app-security-16k.ogg"
```

Ответ соответствует typed contract:

```json
{
  "transcript": [{"speaker": "Оператор", "start": 0.0, "end": 4.2, "text": "..."}],
  "classification": {"topic": "карты", "priority": "medium"},
  "quality_score": {"total": 100, "checklist": {"greeting": true}},
  "compliance": {"passed": true, "issues": []},
  "summary": "...",
  "action_items": ["..."]
}
```

Также доступны `POST /trends` и `WSS /ws/transcribe`. Полный contract: [`docs/api.md`](docs/api.md).

## Локальный запуск

```bash
cp .env.example .env
# Заполнить secrets и OpenAI-compatible gateway/model values.

HF_TOKEN=... uv run python scripts/provision_speech_models.py \
  --artifact-root models/artifacts \
  --output-manifest models/manifest.json \
  --cache-dir .cache/mtbank-speech-models

docker compose up --build --wait
```

Требования: Docker Compose, FFmpeg, около 16 GB RAM и 12 GB disk. GPU/RunPod deployment: [`docs/operations.md`](docs/operations.md), [`deploy/runpod/README.md`](deploy/runpod/README.md).

## Проверка кода

```bash
uv run --offline --no-sync ruff check .
uv run --offline --no-sync ruff format --check .
uv run --offline --no-sync pyright
uv run --offline --no-sync pytest -m "not integration and not real_llm and not gpu"
uv lock --check
uv lock --check --directory services/speech
```

Evaluation scope: synthetic/no-PII audio. Это делает demo, references и метрики воспроизводимыми без использования клиентских банковских данных.
