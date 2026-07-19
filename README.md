# MTBank AI Call Analytics

Решение тестового задания MTBank для анализа русскоязычных звонков: транскрибация, диаризация, определение ролей и четыре независимых LLM-агента.

**Демо:** https://cloud.cloud-tunnel-mega-obx1.space
**Текст задания:** [`docs/assignment.md`](docs/assignment.md)

## Схема

```text
OpenWebUI Pipeline / REST / WebSocket
                ↓
Groq ASR → local pyannote Community-1 → role resolution
                ↓
classifier · quality · compliance · summarizer
                ↓
PostgreSQL · Trends · Prometheus/Grafana
```

## Возможности

- обязательный OpenWebUI `Pipeline` с проверкой ownership, MIME и audio magic;
- WAV, MP3 и OGG; нормализация через FFmpeg;
- transcript с timestamps, speaker/role, классификация, quality checklist, compliance issues, summary и action items;
- PostgreSQL persistence, `POST /analyze`, `/trends`, opt-in `/ws/transcribe`;
- Grafana, Prometheus и bounded WebSocket streaming;
- versioned policies, typed agent outputs и deterministic aggregation.

## Тестовые звонки

В [`test_data/`](test_data/) находятся пять authored synthetic русскоязычных диалогов (WAV/MP3/OGG, 8/16 kHz, 714.802 с) с reference-разметкой. Реальные клиентские и production-данные не используются.

## Запуск

```bash
cp .env.example .env
# Заполнить обязательные secrets и пути к локальному diarization artifact.
docker compose up --build --wait
```

OpenWebUI: `http://127.0.0.1:${OPENWEBUI_PORT:-3000}`.

GPU и WebSocket включаются отдельными overlays:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile gpu up --build --wait
MTBANK_WEBSOCKET_ALLOWED_ORIGIN=https://example.org \
  docker compose -f docker-compose.yml -f docker-compose.websocket.yml up --build --wait
```

API contract: [`docs/api.md`](docs/api.md). Архитектура и privacy boundary: [`docs/architecture.md`](docs/architecture.md), [`docs/privacy.md`](docs/privacy.md).

## Проверка

```bash
uv run --offline --no-sync pytest -m "not real_llm and not gpu and not integration"
uv run --offline --no-sync ruff check .
uv run --offline --no-sync pyright
uv lock --check && uv lock --check --directory services/speech
docker compose -f docker-compose.yml config --quiet
```

## Бонусы

- WebSocket real-time transcription;
- Grafana dashboard и Prometheus metrics;
- evidence-backed Trends agent.

## Ограничения

Runtime намеренно использует Groq `whisper-large-v3-turbo` и local pyannote Community-1. Это **формально не закрывает** требование задания использовать local `faster-whisper` или `openai-whisper`; fallback ASR не реализован. Canonical corpus-wide WER/DER/role metrics, GPU WebSocket p95 и five-minute `<60 с` SLA не заявляются без реального воспроизводимого запуска.
