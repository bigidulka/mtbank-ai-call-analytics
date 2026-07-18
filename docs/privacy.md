# Privacy

Privacy mode — `redacted-cloud`: raw audio retention равен нулю; PostgreSQL хранит
только sanitized analysis record и redacted lifecycle event metadata. В schema нет
колонок raw audio, transcript, prompt или provider response. Canonical batch ASR sends
normalized raw audio to Groq only at `https://api.groq.com/openai/v1/audio/transcriptions`;
this includes bounded temporary-WAV rolling requests for WebSocket provisional updates and
requires approved legal/privacy basis before live operation. Temporary rolling WAV files
are deleted after each provider call. Local pyannote Community-1 runs only from a
provisioned offline artifact.

Для текущего release/demo зафиксировано scope-limited разрешение только на project-generated
synthetic corpus в `release/synthetic-remote-audio-approval.json`. Оно разрешает Groq calls
для validation/demo, но не является legal approval реальных клиентских или production
банковских данных. Любое расширение data classification требует отдельного утверждения.

Release evidence exporter удаляет content-bearing keys (`audio`, `transcript`,
`prompt`, `response`, `secret`, `token`, `key`) рекурсивно. Provider request IDs не
сохраняются в открытом виде: export содержит только SHA-256. Это позволяет подтвердить
наличие четырёх distinct provider requests без раскрытия их значений.

Нельзя передавать в `release-evidence/` аудио, transcript, prompt, HTTP headers,
provider payload, API key, password или screenshots с PII. Browser/Grafana proof
должен быть reviewable и очищен от PII до прикрепления. Retention и legal basis
лицензированного corpus требуют отдельного утверждения владельца данных.
