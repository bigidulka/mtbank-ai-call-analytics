# Privacy

Privacy mode — `redacted-cloud`: PostgreSQL stores only sanitized analysis records and redacted lifecycle-event metadata. The schema has no raw audio, transcript, prompt or provider-response columns.

Canonical batch ASR and diarization run from provisioned local faster-whisper and pyannote artifacts; normalized audio does not leave the canonical speech container. The opt-in WebSocket provisional mode sends bounded temporary WAV input to Groq and deletes it after each provider call; it requires separate approval before live use.

Only project-generated synthetic audio is appropriate for the included validation corpus. Real customer and production banking data require separate organizational legal/privacy approval before remote processing or retention.

Raw audio, transcripts, prompts, HTTP headers, provider payloads, API keys, passwords and PII screenshots must never be committed or attached to public evidence. Generated validation output belongs in ignored local paths.
