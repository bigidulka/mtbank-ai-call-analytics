# External speech benchmark — 2026-08-01

## Scope

Candidate tested on 12 publicly downloadable Russian recordings with different duration and acoustic conditions:

| Source | Files | Audio | Quality | License |
|---|---:|---:|---|---|
| AxonData public samples | 2 | 894.3 s | real call-center, mono 8 kHz MP3 | CC-BY-NC-4.0 |
| Google FLEURS `ru_ru/validation` | 5 | 67.6 s | crowd-recorded read speech, mono 16 kHz WAV | CC-BY-4.0 |
| Golos far-field test | 5 | 13.6 s | real far-field voice commands, mono 16 kHz WAV | public Golos license with attribution/conditions |

Total: 12 files, 975.5 seconds (16.26 minutes). Source URLs, revisions, hashes, raw audio and transcripts remain in ignored local artifacts under `tmp/external-corpus/`. They are intentionally excluded from Git: public download and copyright license do not establish voice/privacy redistribution consent for real calls. Published evidence contains aggregate metrics only in `release-evidence/external-speech-2026-08-01/summary.json`.

No source transcript, timestamp or speaker label was supplied to ChatGPT ASR or Luna inference. References were used only after inference for WER. External samples do not provide trusted `Оператор/Клиент` turn timestamps, so DER and role accuracy cannot be measured honestly.

## Result

| Set | Micro WER | Notes |
|---|---:|---|
| Axon real calls | **25.10%** | Source transcripts themselves contain visible errors/omissions, so this is comparison to supplied references, not verified human-clean truth. |
| FLEURS | **5.56%** | 5 varied-duration crowd-read clips. |
| Golos far-field | **8.33%** | Only 24 reference words; one two-word clip caused both errors. |
| All 12 files | **23.71%** | Dominated by 14.9 minutes of Axon call audio. |

ASR remained fast. Across 975.5 seconds:

- ChatGPT Web ASR: 43.15 s;
- Luna semantic reconstruction: 131.83 s;
- VAD/alignment: 1.09 s;
- sequential wall time: 178.63 s;
- measured real-time factor: 0.183.

Per long call:

| File | Duration | ASR | Luna | VAD/alignment | Sequential total |
|---|---:|---:|---:|---:|---:|
| Finance call | 583.3 s | 18.7 s | 58.2 s | 0.31 s | 77.3 s |
| Pharma call | 311.0 s | 10.9 s | 31.1 s | 0.13 s | 42.1 s |

One observed sequential run exceeded `<60 s` on the 9.7-minute call. This is an observation, not an SLA attestation: provider latency is not reproducible offline. ASR alone completed under 60 s; Luna reconstruction was current bottleneck.

## Critical robustness finding

The synthetic winning profile used fixed edge restoration:

```text
start padding = 0.2 s
end padding   = 0.9 s
```

On both natural call recordings, this exact padding produced overlapping hypothesis intervals and failed closed. A diagnostic retry with reduced `0.1/0.2 s` padding completed alignment. This is not the same preregistered profile and must not inherit the synthetic `1.72% DER` claim.

The two natural calls also contain:

- long within-call hold silence;
- short hesitations and fragments;
- interruptions and rapid turn-taking;
- ASR punctuation/segmentation drift;
- many more semantic turns than clean synthetic calls.

Result confirms previous warning: longest-gap + fixed-padding success depended strongly on authored 800 ms synthetic pauses.

## Verdict

- ChatGPT Web ASR generalizes reasonably to clean/read and far-field samples, but real-call WER against supplied references is much worse than synthetic WER.
- Luna can reconstruct plausible semantic turns, but no trusted external role labels exist here; role accuracy remains unproven.
- Exact fixed-padding VAD alignment fails on both real long calls.
- Candidate is **not ready to replace RunPod**.
- Best next step: obtain or manually create blinded role/timestamp annotations for these calls, replace fixed edge restoration with topology-safe midpoint clipping, and reduce Luna latency through chunked/bounded role reconstruction.

## Evidence

- aggregate summary: `release-evidence/external-speech-2026-08-01/summary.json`;
- evidence manifest: `release-evidence/external-speech-2026-08-01/manifest.json`;
- current collector/evaluator: `scripts/benchmark_external_speech.py`;
- exact evaluator revision used by recorded run: `release-evidence/external-speech-2026-08-01/run-evaluator.py`;
- privacy-safe sanitizer: `scripts/sanitize_external_speech_evidence.py`;
- verifier: `scripts/verify_external_speech_evidence.py`.

Verify aggregate derivations and evidence binding:

```bash
uv run python scripts/verify_external_speech_evidence.py
```

Raw provider output and external media stay ignored locally. Preserved historic evaluator hash binds recorded result to exact executed code; current evaluator contains later hardening and is not presented as identical run code. Published summary states diagnostic adaptive padding was used on both real calls. Fixed-profile failure was reproduced locally from stored Luna assignments: both Axon calls raise `generated hypothesis intervals overlap or reorder` at `0.2/0.9`; all ten short controls pass. This reproduction cannot be rerun from public aggregate evidence because raw voices/content are deliberately excluded.
