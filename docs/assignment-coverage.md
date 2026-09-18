# Assignment coverage

Проект вырос из тестового задания МТБанка на роль AI Engineer (AI-агенты и речевая
аналитика контакт-центра). Исходный текст задания сохранён без изменений в
[assignment.md](assignment.md); контрактный тест
[`tests/contract/test_assignment_copy.py`](../tests/contract/test_assignment_copy.py)
проверяет его побайтовое совпадение.

Ниже — что именно реализовано по каждому критерию задания и где это проверяется.
Таблица описывает соответствие заданию, а не заявление о production-готовности:
production/внешние gates перечислены в [release-checklist.md](release-checklist.md).

| Критерий задания | Реализация | Проверяемое доказательство |
|---|---|---|
| Pipeline архитектура — 25 | Настоящий OpenWebUI Pipeline, attachment flow, общий workflow для UI и API | [`pipeline.py`](../pipeline.py), [`docker-compose.yml`](../docker-compose.yml) |
| ASR качество — 20 | local `faster-whisper` `large-v3-turbo`, local pyannote Community-1, timestamps, роли, WAV/MP3/OGG | [`canonical-speech-evaluation.json`](../release-evidence/final-115/canonical-speech-evaluation.json) |
| Multi-Agent — 25 | classifier, quality, compliance, summarizer; независимые prompts/tools/contracts | [`src/mtbank_ai/agents/`](../src/mtbank_ai/agents/), [`workflow/analysis.py`](../src/mtbank_ai/workflow/analysis.py) |
| Код и архитектура — 15 | FastAPI, PostgreSQL, typed schemas, retries, circuit breaker, tests, `.env.example` | [CI](../.github/workflows/ci.yml), [`tests/`](../tests/) |
| Документация — 10 | Схема, demo flow, модели, метрики, запуск, API | [README](../README.md), [`docs/`](.) |
| Живое демо — 5 | Публичный HTTPS OpenWebUI и измеренный 5-minute end-to-end workload (демо остановлено, evidence сохранён) | [`public-five-minute-sla.json`](../release-evidence/final-115/public-five-minute-sla.json) |
| Bonus: WebSocket — 5 | Rolling provisional transcription + canonical reconciliation; live diagnostic | [`websocket-p95.json`](../release-evidence/final-115/websocket-p95.json) |
| Bonus: Grafana — 5 | Calls, Quality total, Top topics, latency, errors, agent tokens | [`mtbank-overview.json`](../monitoring/grafana/dashboards/mtbank-overview.json) |
| Bonus: Trends — 5 | Анализ нескольких persisted calls с evidence-backed recommendation | [`trends-response.json`](../release-evidence/final-115/trends-response.json) |

## Что deliberately не заявлено

- Внешняя attestation модельных артефактов и registry provenance.
- Согласование обработки реальных клиентских аудио, банковской тайны и PII.
- Repeated load/stability study на шумных production-звонках.
- Production SLO, incident response и долгосрочная ротация секретов.

Эти пункты перечислены как открытые в [release-checklist.md](release-checklist.md) и не
подменяются синтетическим демо.
