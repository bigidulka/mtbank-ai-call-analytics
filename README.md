# MTBank AI Call Analytics

Решение тестового задания MTBank для анализа русскоязычных звонков: транскрибация, диаризация, определение ролей и четыре независимых LLM-агента.

**Демо:** https://cloud.cloud-tunnel-mega-obx1.space
**Текст задания:** [`docs/assignment.md`](docs/assignment.md)

## Схема

```text
OpenWebUI Pipeline / REST / WebSocket
                ↓
local faster-whisper large-v3-turbo → local pyannote Community-1 → role resolution
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

Типизированный собственный Supervisor выбран вместо LangGraph: линейный bounded workflow «speech → 4 агента → aggregation» не требует checkpoint-графа или human-in-the-loop state и остаётся проще для проверки.

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

RunPod GPU Pod запускает только GPU speech container; он не запускает Docker Compose. Split-GPU deployment foundation и remote bearer boundary: [`deploy/runpod/README.md`](deploy/runpod/README.md).

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

Canonical batch runtime использует local `faster-whisper` `large-v3-turbo` (CTranslate2) и local pyannote Community-1; ASR fallback отсутствует. Opt-in WebSocket provisional mode отдельно использует Groq и не влияет на batch ASR requirement. Five-minute CPU benchmark на этом хосте завершился HTTP 200 за **483.178 с** — SLA `<60 с` не выполнен. Canonical corpus-wide WER/DER/role metrics и GPU WebSocket p95 не заявляются: controlled evaluator остановился fail-closed после 3/5 files на HTTP 500.
