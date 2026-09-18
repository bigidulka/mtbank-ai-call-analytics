# Notice

This repository mixes code written for the project with material that belongs to
third parties. The MIT license in [LICENSE](LICENSE) applies to the implementation code
only.

## Implementation code

`pipeline.py`, `src/`, `services/`, `scripts/`, `tests/`, `monitoring/`, `docker/`,
`deploy/`, `gateway/` and the Compose files are the author's own implementation work.
Copyright (c) 2026 bigidulka, MIT.

## Assignment text

[`docs/assignment.md`](docs/assignment.md) is the original test-assignment text
("ЗАО МТБанк — Тестовое задание AI Engineer"), published by its author in the template
repository `ZubikIT/mtbank-ai-hiring`. It is kept here unchanged as context for what was
asked, is verified byte-for-byte by
[`tests/contract/test_assignment_copy.py`](tests/contract/test_assignment_copy.py), and is
**not** covered by this repository's MIT license.

## Evaluation corpus

`test_data/` contains five authored synthetic Russian bank dialogues generated with Edge
TTS (`ru-RU-SvetlanaNeural`, `ru-RU-DmitryNeural`). They are marked in
[`test_data/manifest.yaml`](test_data/manifest.yaml) as
`LicenseRef-MTBank-Synthetic-EdgeTTS-Demo` (transport-only silence fixtures use
`LicenseRef-MTBank-transport-fixture`). The corpus contains no real customers, no banking
secrecy material and no personal data; it is included so the assignment-scoped WER/DER/role
metrics can be reproduced. It is not re-licensed by this repository.

## Third-party software

OpenWebUI, faster-whisper / CTranslate2, pyannote.audio, PostgreSQL, Prometheus, Grafana,
Tempo, nginx and the Python dependencies in `uv.lock` remain under their own licenses and
are referenced as dependencies, not redistributed here. Model weights are not part of the
repository; `scripts/provision_speech_models.py` verifies them against
`models/manifest.json` after provisioning.
