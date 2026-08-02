# Luna vs Terra vs Sol — semantic role comparison

## Design

Paired test used identical frozen inputs:

- same ChatGPT Web ASR text;
- same VAD anchors and durations;
- same prompt/tool schema;
- temperature `0`;
- no reference text, timestamps or roles in inference;
- 12 files × 3 models × 3 repeats = 108 successful runs;
- seeded randomized request order (`20260802`) to reduce provider-time/order confounding;
- 110 actual request attempts: Terra required two retries.

Models:

- `gpt-5.6-luna`;
- `gpt-5.6-terra`;
- `gpt-5.6-sol`.

This isolates model selection from ASR and VAD variation. Raw transcripts, assignments, names and source calls remain ignored locally. Published evidence is aggregate-only.

## Does result depend on model?

**Yes, but not only model.** Across 36 paired file/repeat cases, all three models produced identical role boundaries in 24 cases (66.7%). Mean three-model word-role disagreement was 11.59%. Same-model outputs also changed between repeats, so model selection and provider nondeterminism both matter.

Pairwise mean disagreement:

| Pair | All files | Two real calls |
|---|---:|---:|
| Luna vs Terra | 11.67% | 3.36% |
| Luna vs Sol | 11.73% | 3.73% |
| Terra vs Sol | 11.36% | 1.51% |

Large all-file disagreement is partly caused by single-speaker control clips: prompt forces `Оператор/Клиент` even though those clips have no business roles, so one whole-clip label flip becomes 100% disagreement. Real-call disagreements are smaller but nonzero, and boundaries differ on every real-call repeat.

## Repeat stability

Despite temperature `0`, provider outputs were not fully deterministic:

| Model | Files identical across all 3 repeats |
|---|---:|
| Luna | **10/12 (83.3%)** |
| Terra | 8/12 (66.7%) |
| Sol | 7/12 (58.3%) |

Temperature zero therefore does not guarantee identical output on this gateway/model family.

Mean disagreement between repeats of the **same** model:

| Model | All files | Two real calls |
|---|---:|---:|
| Luna | **0.36%** | 2.16% |
| Terra | 11.40% | 1.70% |
| Sol | 16.80% | **0.77%** |

On real calls, between-model disagreement (1.51–3.73%) is comparable to same-model variation (0.77–2.16%). Therefore real-call output is not controlled by model alone. Sol is least variable on real calls, while Luna is most stable across all files.

## Speed

| Model | Median role-call latency | Mean latency |
|---|---:|---:|
| Terra | **5.19 s** | 13.20 s |
| Sol | 5.56 s | **12.25 s** |
| Luna | 5.70 s | 12.79 s |

Terra had lowest median, Sol lowest mean. Terra required two retries; recorded latency includes retry/backoff time. Differences remain observational provider latency, not stable SLA guarantees.

## Post-run manual reference check

References cryptographically bind frozen comparison output. Author attests they were manually assigned after provider outputs were frozen and hidden from annotator, but chronology/blinding has no independent trusted timestamp. References were not double-annotated and source transcripts contain errors, so metrics are diagnostic only.

| Model | Mean role accuracy | Role DER | SA-WER |
|---|---:|---:|---:|
| Luna | 39.46% | 60.72% | 31.57% |
| Terra | 46.76% | 53.43% | 27.78% |
| Sol | **50.14%** | **50.04%** | **26.86%** |

Sol was best observed model on these two real calls: +3.38 percentage points over Terra and +10.67 over Luna. Sample and reference quality remain too weak to call Sol a production winner.

## Alignment conclusion

All three models failed fixed synthetic edge restoration (`0.2 s` start / `0.9 s` end) on both real calls in every repeat. Model choice does not solve core VAD/topology problem.

Adaptive padding completed all runs, but this is a different profile:

All models completed fixed padding on 30/36 runs: failures were both real calls × three repeats. Adaptive padding completed all 108 successful runs.

## Verdict

1. Semantic role/boundary output is **model-associated**, but provider nondeterminism also matters.
2. Sol scored best on limited post-run real-call references.
3. Luna was most repeat-stable; Terra had lowest median latency; Sol lowest mean latency.
4. Luna showed no quality advantage on this external reference check.
5. Main real-call failure remains alignment design, not model choice.
6. Keep RunPod production path. If continuing no-GPU profile, prefer Sol for next experiment, but first replace fixed padding and add independently annotated natural calls.

Evidence: `release-evidence/semantic-role-model-comparison-2026-08-02/summary.json`.
