# Semantic speaker attribution experiments

Status: research profile only. It does not replace the validated submission speech path.

## Baseline decomposition

Canonical release evidence (`release-evidence/final-115/canonical-speech-evaluation.json`):

| Metric | Result |
|---|---:|
| WER | 7.34% |
| DER | 20.30% |
| DER miss | 14.81 pp |
| DER false alarm | 4.27 pp |
| DER speaker confusion | 1.22 pp |
| Time-weighted role accuracy | 83.97% |
| Speaker-attributed WER | 7.34% |

Speaker-attributed WER equals ordinary WER. Most DER and role error therefore comes from reference-envelope/timestamp miss, not incorrect Operator/Client attribution of recognized words. A replacement must improve attribution quality, not merely expand segment boundaries.

## Completed experiments

| Condition | Role accuracy | Speaker-attributed WER | Interpretation |
|---|---:|---:|---|
| Oracle reference turns, labels withheld, Luna roles | 100.00% | n/a | Upper bound only; true turn boundaries leak structure |
| Saved faster-whisper word stream, one-pass Luna resegmentation | 80.63% | 11.39% | Worse than canonical baseline |
| Pause turns at 1.15 s, then Luna roles | 79.29% | 12.36% | Worse than canonical baseline |
| Pause turns at 1.30 s, then Luna roles | 79.35% | 15.25% | Worse than canonical baseline |
| Pause turns at 1.45 s, then Luna roles | 74.54% | 27.70% | Worse than canonical baseline |

Artifacts are local ignored files under `tmp/`. Evaluators are committed:

- `scripts/evaluate_semantic_role_attribution.py`
- `scripts/evaluate_semantic_resegmentation.py`
- `scripts/evaluate_pause_semantic_roles.py`
- `src/mtbank_ai/agents/semantic_role_attributor/v1.md`
- `src/mtbank_ai/agents/semantic_resegmenter/v1.md`

## Provider hypotheses

### OpenAI direct diarization — highest-priority candidate

`gpt-4o-transcribe-diarize` returns `diarized_json` segments with speaker, start and end. For files longer than 30 seconds it requires `chunking_strategy=auto` or explicit VAD. Optional 2–10 second reference clips can map up to four known speakers. Luna would map anonymous speaker labels to Operator/Client only when known speaker references are unavailable.

This candidate can remove both local Whisper and pyannote. It must beat current WER, speaker confusion, role attribution and end-to-end latency on the same five files. Raw customer audio leaving MTBank requires separate privacy/legal approval.

### Groq Whisper + lightweight boundary detector + Luna — cost/latency candidate

Groq `whisper-large-v3` and `whisper-large-v3-turbo` provide `verbose_json` word and segment timestamps but no documented speaker diarization. Official listed prices are $0.111/hour and $0.04/hour; listed speed factors are 189x and 216x.

Naive Luna-only resegmentation failed. Groq needs an independent boundary signal before Luna:

- stereo/channel metadata when available;
- telephony VAD/pause and interruption features;
- a small acoustic speaker-change detector or CPU diarizer;
- Luna only for uncertain boundaries and business-role mapping.

### OpenAI timestamp path + Luna

`whisper-1` supports word/segment timestamps. `gpt-transcribe` is the recommended high-accuracy transcription model and costs $0.0045/minute, but official docs direct timestamp use to `whisper-1`. Without direct diarization, this path has the same boundary problem as Groq.

## Fair benchmark gate

Candidate replaces canonical path only if all are true on a blinded corpus with raw provider output saved before scoring:

1. WER no worse than 7.34%.
2. Speaker-attributed WER lower than 7.34% or statistically tied with lower cost/latency.
3. Speaker confusion lower than 1.22 pp; report miss and false alarm separately.
4. Time-weighted role accuracy higher than 83.97% without timestamp-envelope inflation.
5. Five-minute end-to-end latency below 27.88 seconds, or a documented cost/availability gain accepted as trade-off.
6. No use of reference speaker labels, true turn boundaries, reference text, filenames containing scenario labels, or evaluation answers in prompts.
7. Fail-closed schema, timeout, size, cancellation, provenance and privacy controls remain intact.

## Recommended architecture

```text
audio
  -> pluggable ASR profile
     - openai_diarized
     - groq_words
     - generic_whisper_words
     - local_gpu_canonical
  -> immutable provider-normalized words/segments
  -> boundary layer
     - direct provider speakers when available
     - channel metadata
     - acoustic change detector / CPU diarizer
     - bounded semantic review only for uncertainty
  -> Luna Operator/Client role mapping
  -> existing typed analytics workflow
```

Run candidates in shadow mode. Never silently fall back between providers. Record provider/model revision, request ID hash, timestamps, latency, cost estimate and exact prompt bundle hash.
