# Evals и тестовые данные

## Corpus

`test_data/manifest.yaml` содержит пять authored synthetic русскоязычных звонков: 714.802 секунды, WAV/MP3/OGG, 8 kHz telephone fixture, два голоса, reference timestamps и роли `Оператор`/`Клиент`. Реальных клиентов и PII нет.

```bash
uv run python scripts/validate_test_manifest.py --require-release-corpus
```

## Canonical batch evaluation

Canonical path — local `faster-whisper` `dropbox-dash/faster-whisper-large-v3-turbo` (CTranslate2, `int8` CPU / `float16` CUDA) и local offline `pyannote/speaker-diarization-community-1`. Runtime загружает только artifact, проверенный по manifest SHA-256; загрузка модели из сети запрещена.

```bash
uv run python scripts/evaluate_canonical_speech.py \
  --base-url http://canonical-speech-host:8010 \
  --output release-evidence/canonical-speech-evaluation.json
```

Runner последовательно отправляет все пять files в canonical `/v1/transcribe`, затем считает micro WER, DER, time-weighted role accuracy и speaker-attributed WER. Output содержит только hashes, component revisions, latency и metric counts — без audio, transcript, provider body, request ID или credentials. `409` и invalid response останавливают run fail-closed. Ограниченный retry применяется только к transient `429/500/502/503/504`; исчерпание retry останавливает run, а partial result не является corpus-wide metric claim. Актуальный controlled GPU-отчёт: [`../release-evidence/final-115/canonical-speech-evaluation.json`](../release-evidence/final-115/canonical-speech-evaluation.json).

## Five-minute SLA

```bash
uv run python scripts/run_local_speech_sla_benchmark.py \
  --base-url http://canonical-speech-host:8010 \
  --audio test_data/synthetic/credit-consultation-16k.wav \
  --output release-evidence/five-minute-sla.json
```

Benchmark детерминированно создаёт ровно 300-секундный WAV, передаёт его canonical service и сохраняет только hashes, duration, HTTP status и wall latency. SLA `<60 с` объявляется только при `within_sla=true` в actual output. Контролируемый CPU result [`local-faster-whisper-five-minute-cpu.json`](../test_data/evaluations/local-faster-whisper-five-minute-cpu.json): HTTP 200, 300 секунд audio, 483.178 секунд wall latency, `within_sla=false`.

## Streaming

Groq используется только в opt-in WebSocket provisional mode. Он не является batch fallback и не влияет на local-ASR canonical evaluation.

## Artifact policy

`models/manifest.json` и `models/.gitkeep` версионируются; weights в `models/artifacts/` и временные benchmark files игнорируются. Privacy-safe финальный evidence после полного controlled run версионируется в [`../release-evidence/final-115/`](../release-evidence/final-115/) и связан SHA-256 manifest. Audio, transcript, credentials и raw provider bodies туда не входят.
