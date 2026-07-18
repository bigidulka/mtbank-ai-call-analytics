# References and licence status

В текущем репозитории нет лицензированного корпуса русской речи и поэтому нет
публикуемой WER/DER таблицы. `LicenseRef-MTBank-transport-fixture` относится
только к собственным silence-only transport fixtures и не является лицензией на
речевой dataset.

Release gate: до добавления speech corpus с проверяемыми URL/лицензией,
provenance, SHA-256, reference text и role labels команда
`python scripts/validate_test_manifest.py --require-release-corpus` завершается
с ошибкой. Нельзя заменять этот gate tone, browser WAV или TTS без явной
лицензии и marking `speech_reference`.
