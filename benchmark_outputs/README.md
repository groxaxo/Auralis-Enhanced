# WER benchmark artifacts

The JSON files in this directory are historical measurements from one machine. They are evidence, not universal performance claims.

## Review of the committed baseline and optimized runs

The existing `summary_optimized.json` is a regression relative to `summary_baseline.json`:

| Metric | Baseline | Optimized | Relative change |
| --- | ---: | ---: | ---: |
| Mean utterance WER | 0.06234 | 0.08968 | **+43.9% worse** |
| English WER | 0.02356 | 0.03517 | **+49.3% worse** |
| Spanish WER | 0.10112 | 0.14419 | **+42.6% worse** |
| Mean request latency | 7.902 s | 8.175 s | **+3.4% worse** |
| P95 request latency | 14.513 s | 17.511 s | **+20.7% worse** |

Do not promote the measured optimization on the basis of these files. Re-run it with the corrected benchmark and accept it only when the comparison gate passes.

## Important methodology note

The original script removed hyphens instead of replacing them with spaces. For example, `deep-sea` became `deepsea`, while the ASR output commonly became `deep sea`. That inflated WER for hyphenated phrases. It also averaged sentence WERs without reporting corpus WER, included cold-start effects inconsistently, allowed stochastic generation, and could resume into results created with a different configuration.

`wer_benchmark.py` now:

- reports corpus WER and mean/median utterance WER;
- normalizes punctuation without joining hyphenated words;
- uses deterministic generation by default;
- warms both TTS and ASR before timed samples;
- records audio duration and real-time factor;
- fingerprints corpus, endpoints, voices, ASR model, and generation settings;
- writes checkpoints atomically and refuses incompatible resume files;
- returns a non-zero status when a candidate regresses beyond configured gates.

The current historical JSON files use the legacy schema and normalization. Keep them for traceability, but do not compare their absolute WER directly with new schema-v2 runs.
