# Privacy

Privacy mode — `redacted-cloud`: PostgreSQL stores only sanitized analysis records and redacted lifecycle-event metadata. The schema has no raw audio, transcript, prompt or provider-response columns.

Canonical batch ASR sends normalized raw audio to Groq only at `https://api.groq.com/openai/v1/audio/transcriptions`; bounded temporary WAV inputs used by WebSocket provisional updates are deleted after each provider call. Local pyannote Community-1 runs only from a provisioned offline artifact.

Only project-generated synthetic audio is appropriate for the included validation corpus. Real customer and production banking data require separate organizational legal/privacy approval before remote processing or retention.

Raw audio, transcripts, prompts, HTTP headers, provider payloads, API keys, passwords and PII screenshots must never be committed or attached to public evidence. Generated validation output belongs in ignored local paths.
