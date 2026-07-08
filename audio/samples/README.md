# Тестовые аудио-сэмплы

10 записей реальной русской речи из открытых источников для тестирования ASR-пайплайна.

| Файл | Формат | Длит. | Источник |
|---|---|---|---|
| sample_01_ru.mp3 | 16kHz mono | 4.9 с | Tinkoff VoiceKit examples |
| sample_02_ru.mp3 | 16kHz mono | 4.8 с | Tinkoff VoiceKit examples |
| sample_03_ru.wav | 16kHz mono | 7.4 с | Tinkoff VoiceKit examples |
| sample_04_ru.wav | 48kHz mono | 3.5 с | Tinkoff VoiceKit examples |
| sample_05_ru.wav | 48kHz mono | 1.9 с | Tinkoff VoiceKit examples |
| sample_06_ru.wav | 48kHz mono | 9.9 с | Tinkoff VoiceKit examples |
| sample_07_ru.wav | 16kHz mono | 3.2 с | pisets (bond005), Apache 2.0 |
| sample_08_ru.wav | 16kHz mono | 11.3 с | pisets (bond005), Apache 2.0 |
| sample_09_ru.wav | 16kHz **stereo** | 11.3 с | pisets (bond005), Apache 2.0 |
| sample_10_ru.wav | 16kHz mono | 4.0 с | Golos (SberDevices) |

## Зачем разные форматы

Сэмплы намеренно разнородны — ваш pipeline должен обрабатывать всё:
- **Разные sample rate** (16kHz / 48kHz) — проверка ресемплинга
- **MP3 и WAV** — проверка декодирования форматов
- **Стерео** (sample_09) — проверка downmix / канальной обработки
- **Разная длительность** (2–11 сек) — короткие фразы и длинные предложения

## Источники и лицензии

- **Tinkoff VoiceKit examples** — https://github.com/Tinkoff/voicekit-examples (Apache 2.0)
- **pisets** — https://github.com/bond005/pisets (Apache 2.0)
- **Golos** — https://github.com/salute-developers/golos (публичная лицензия SberDevices)

## Как использовать в задании

Прогоните все 10 файлов через ваш ASR-компонент и приложите к README таблицу с транскриптами. Это быстрый способ показать что pipeline работает на разных входных данных.

Для демонстрации Multi-Agent аналитики (диалог оператор/клиент) синтезируйте полный звонок по сценарию из [`../docs/sample-dialog.md`](../../docs/sample-dialog.md) через edge-tts или Silero — команды в том же файле.
