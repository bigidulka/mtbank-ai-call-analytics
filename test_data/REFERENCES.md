# References and licence status

`test_data/manifest.yaml` содержит пять authored synthetic русскоязычных банковских диалогов: 714.802 секунды, WAV/MP3/OGG, 8/16 kHz, два синтетических голоса, reference transcript/timestamps и роли `Оператор`/`Клиент`.

Корпус создан специально для задания через Edge TTS и маркирован `LicenseRef-MTBank-Synthetic-EdgeTTS-Demo`. Реальных клиентов, банковской тайны и PII нет. Публикуемые WER/DER/role metrics относятся только к этому synthetic/no-PII evaluation scope и не заявляют качество на шумных production-звонках.

Transport-only silence fixtures имеют отдельную лицензию `LicenseRef-MTBank-transport-fixture` и исключены из WER/DER.

Проверка corpus и provenance:

```bash
uv run python scripts/validate_test_manifest.py --require-release-corpus
```

Актуальный canonical GPU-отчёт и SHA-256 manifest: [`../release-evidence/final-115/`](../release-evidence/final-115/).
