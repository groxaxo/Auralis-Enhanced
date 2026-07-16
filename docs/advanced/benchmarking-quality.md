# Benchmarking speech quality and latency

Use the repository-level `wer_benchmark.py` for a reproducible bilingual TTS-to-ASR evaluation. The default corpus is the 50-English/50-Spanish text set embedded in the historical baseline result file. NVIDIA Parakeet TDT 0.6B v3 supports both English and Spanish and automatically detects the spoken language.

## Prerequisites

Start Auralis on port `6688` and expose an OpenAI-compatible Parakeet transcription endpoint on port `5092`.

```bash
sudo systemctl restart auralis-enhanced
curl --fail http://127.0.0.1:6688/health
```

## Smoke run

Run two samples per language before spending time on the full corpus:

```bash
python wer_benchmark.py \
  --tag smoke \
  --limit-per-language 2 \
  --no-resume
```

## Full baseline and candidate

```bash
python wer_benchmark.py --tag baseline-v2 --no-resume

# Apply the implementation or runtime change being evaluated, restart the service,
# then run the exact same corpus and generation settings.
python wer_benchmark.py \
  --tag candidate-v2 \
  --no-resume
```

Compare and gate the candidate by running it with `--compare`. The benchmark exits with status `2` when a metric exceeds the allowed regression:

```bash
python wer_benchmark.py \
  --tag candidate-v2 \
  --baseline-tag baseline-v2 \
  --candidate-tag candidate-v2 \
  --compare \
  --max-wer-regression-pct 0 \
  --max-latency-regression-pct 0
```

A small latency tolerance can be intentional for noisy shared hardware, for example `--max-latency-regression-pct 5`. Keep the WER tolerance at zero unless the quality trade-off is explicitly accepted.

## Reproducibility rules

Use the same voices, corpus, endpoints, ASR model, generation options, warm-up count, and GPU allocation for baseline and candidate. The script fingerprints these inputs and refuses to append to an incompatible schema-v2 result file. Use a new tag or `--no-resume` when any benchmark input changes.

Generation is deterministic by default (`do_sample=false`). Use `--stochastic` only for distributional testing, and then run multiple repetitions rather than treating a single sample per sentence as authoritative.

## Outputs

Each run writes:

- `results_<tag>.json`: per-utterance transcript, errors, WER, latency, audio duration, and RTF;
- `summary_<tag>.json`: corpus and utterance WER, language breakdowns, latency percentiles, RTF, and complete run metadata;
- `<tag>_<task-id>.wav`: generated audio, ignored by Git.

Generated result/summary JSON files are ignored by default. Commit only deliberate, reviewed benchmark evidence and include the hardware, software revision, model revisions, and runtime configuration in the surrounding documentation.
