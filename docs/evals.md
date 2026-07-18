# Evals и тестовые данные

## Speech corpus

`test_data/manifest.yaml` имеет статус `release_ready` и содержит:

- 5 authored synthetic русских звонков;
- 714.802 секунды аудио;
- два разных голоса и exact роли `Оператор` / `Клиент`;
- WAV, MP3 и OGG;
- 8 kHz telephone-quality fixture;
- reference transcript segments с timestamps;
- SHA-256 аудио и reference artifacts;
- отдельные transport-only silence fixtures, исключённые из scoring.

Сценарии написаны для проекта и сгенерированы Edge TTS голосами
`ru-RU-SvetlanaNeural` и `ru-RU-DmitryNeural`, как прямо разрешено заданием.
Реальных клиентов и персональных данных нет.

Воспроизведение corpus:

```bash
uv run --with edge-tts --with pydub python scripts/generate_synthetic_dataset.py
uv run python scripts/validate_test_manifest.py --require-release-corpus
```

## Canonical speech evaluation

Единственный canonical runtime path — `Groq whisper-large-v3-turbo` для ASR и local
offline `pyannote/speaker-diarization-community-1` для diarization. Это **формальное
отклонение** от формулировки задания про local `faster-whisper`: `faster-whisper`,
`openai-whisper` и ASR fallback не реализованы и не заявляются как runtime capability.

Для controlled environment с уже доступным canonical `/v1/transcribe` используйте
последовательный runner:

```bash
uv run python scripts/evaluate_canonical_speech.py \
  --base-url http://canonical-speech-host:8010 \
  --output release-evidence/canonical-speech-evaluation.json
```

Runner строго валидирует release manifest до первого запроса, передаёт каждый файл
одним multipart `POST /v1/transcribe` с declared MIME (`audio/wav`, `audio/mpeg` или
`audio/ogg`) и не параллелит audio requests. Artifact содержит только fixture/audio/
reference/hypothesis SHA-256, component revisions, latency и WER/DER/role/
speaker-attributed metric results — без audio, transcript, provider body, request ID
или credentials. `502` останавливает evaluation как provider failure; `409` явно
фиксируется как role-resolution failure, а не превращается в metric.

WER сравнивает time-sorted sequence нормализованных слов и не зависит от UUID
canonical segments или иной сегментации. DER, time-weighted role accuracy и
speaker-attributed WER сохраняются отдельными metrics.

На текущем revision canonical corpus-wide metrics **не выполнены и не опубликованы**.
Отсутствие результата не означает pass release gate.

## Direct Groq evaluator

```bash
uv run python scripts/evaluate_groq_stt.py
```

`evaluate_groq_stt.py` обращается к Groq напрямую и является только
**noncanonical external ASR sanity evaluator**. Он не запускает local Community-1,
не вычисляет DER/role/speaker-attributed metrics и не доказывает canonical runtime
quality или GPU SLA. Его output хранит hashes и error counts, но не raw provider
response.

## Artifact policy

`models/manifest.json` и `models/.gitkeep` — reviewable repository metadata.
Provisioned `models/artifacts/`, `tmp/`, `secrets/` и `release-evidence/` не коммитятся.
Privacy-safe artifacts от controlled runs должны
передаваться как external CI/runtime evidence; они не заменяются hand-authored JSON.

## Agent evals

Offline tests проверяют schema, evidence IDs, required retrieval, terminal submit,
prompt injection boundaries, budgets, retry и deterministic aggregation. Любые live
agent claims требуют nonce-bound external attestation; локальные unit tests не
являются такой attestation.
