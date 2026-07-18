# Provenance test data

`transport/silence-16k.{wav,mp3,ogg}` создаются
`../scripts/generate_transport_fixtures.py` из цифровой тишины. Это fixtures для
проверки MIME/magic, codec path, FFmpeg и cleanup; в них нет речи, TTS, диалога,
эталонного транскрипта или speaker labels.

Каждый файл и SHA-256 указаны в `manifest.yaml`. Любое использование записи с
`kind: transport_only` для WER, DER, role accuracy или speaker-attributed WER
должно быть отвергнуто validator-ом.
