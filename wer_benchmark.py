#!/usr/bin/env python3
"""Reproducible end-to-end quality and latency benchmark for Auralis Enhanced.

The benchmark synthesizes a bilingual corpus through the OpenAI-compatible TTS
endpoint, transcribes the generated WAV files with an OpenAI-compatible ASR
endpoint, and reports both corpus WER and utterance-level WER.

The historical ``benchmark_outputs/results_baseline.json`` file is used as the
default 50-English/50-Spanish corpus source for backward compatibility. Only
its ``text`` and ``lang`` fields are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
import unicodedata
import wave
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import requests

REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "benchmark_outputs"
DEFAULT_CORPUS_PATH = DEFAULT_OUTPUT_DIR / "results_baseline.json"
DEFAULT_AURALIS_URL = "http://127.0.0.1:6688"
DEFAULT_PARAKEET_URL = "http://127.0.0.1:5092"
TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
APOSTROPHES = {"'", "’", "ʼ", "＇"}


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    text: str
    language: str
    voice: str


@dataclass(frozen=True)
class GenerationConfig:
    model: str = "tts-1"
    response_format: str = "wav"
    temperature: float = 0.75
    top_p: float = 0.85
    top_k: int = 50
    repetition_penalty: float = 5.0
    max_ref_length: int = 30
    gpt_cond_len: int = 12
    gpt_cond_chunk_len: int = 6
    do_sample: bool = False
    apply_novasr: bool = False


def normalize_text(text: str) -> str:
    """Normalize text for WER without collapsing hyphenated words together."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    characters: list[str] = []
    for character in normalized:
        if character in APOSTROPHES:
            # Treat contractions and possessives consistently: "earth's" -> "earths".
            continue
        category = unicodedata.category(character)
        if character.isspace() or category[0] in {"P", "S", "C"}:
            characters.append(" ")
        else:
            characters.append(character)
    return " ".join("".join(characters).split())


def _word_error_count(reference: str, hypothesis: str) -> tuple[int, int]:
    """Return (Levenshtein word errors, reference word count)."""

    reference_words = normalize_text(reference).split()
    hypothesis_words = normalize_text(hypothesis).split()
    if not reference_words:
        return (0 if not hypothesis_words else len(hypothesis_words), 0)

    previous = list(range(len(hypothesis_words) + 1))
    for row, reference_word in enumerate(reference_words, start=1):
        current = [row]
        for column, hypothesis_word in enumerate(hypothesis_words, start=1):
            substitution_cost = 0 if reference_word == hypothesis_word else 1
            current.append(
                min(
                    current[column - 1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1], len(reference_words)


def wer(reference: str, hypothesis: str) -> float:
    """Compute utterance Word Error Rate."""

    errors, reference_words = _word_error_count(reference, hypothesis)
    if reference_words == 0:
        return 0.0 if errors == 0 else 1.0
    return errors / reference_words


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _load_json_or_jsonl(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        return rows
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_corpus(
    path: Path,
    *,
    english_voice: str,
    spanish_voice: str,
    limit_per_language: int | None = None,
) -> list[BenchmarkTask]:
    """Load a JSON/JSONL corpus or legacy benchmark result mapping."""

    if not path.is_file():
        raise FileNotFoundError(f"Corpus file does not exist: {path}")
    payload = _load_json_or_jsonl(path)

    rows: Iterable[tuple[str, Mapping[str, Any]]]
    if isinstance(payload, Mapping):
        if "results" in payload and isinstance(payload["results"], Mapping):
            rows = ((str(key), value) for key, value in payload["results"].items())
        else:
            rows = ((str(key), value) for key, value in payload.items())
    elif isinstance(payload, list):
        rows = (
            (str(value.get("id") or value.get("task_id") or f"sample_{index:03d}"), value)
            for index, value in enumerate(payload)
            if isinstance(value, Mapping)
        )
    else:
        raise ValueError("Corpus must be a JSON object, JSON array, or JSONL file")

    tasks: list[BenchmarkTask] = []
    seen_ids: set[str] = set()
    language_counts: dict[str, int] = {}
    for task_id, row in rows:
        text = str(row.get("text", "")).strip()
        language = str(row.get("lang") or row.get("language") or "").strip().lower()
        if not text or not language:
            continue
        if task_id in seen_ids:
            raise ValueError(f"Duplicate corpus task id: {task_id}")
        if limit_per_language is not None and language_counts.get(language, 0) >= limit_per_language:
            continue
        default_voice = spanish_voice if language == "es" else english_voice
        voice = str(row.get("voice") or default_voice).strip()
        if not voice:
            raise ValueError(f"Task {task_id} has an empty voice")
        tasks.append(BenchmarkTask(task_id, text, language, voice))
        seen_ids.add(task_id)
        language_counts[language] = language_counts.get(language, 0) + 1

    if not tasks:
        raise ValueError(f"No valid benchmark tasks found in {path}")
    return tasks


def _benchmark_fingerprint(
    tasks: Sequence[BenchmarkTask],
    generation_config: GenerationConfig,
    *,
    auralis_url: str,
    parakeet_url: str,
    asr_model: str,
) -> str:
    payload = {
        "tasks": [asdict(task) for task in tasks],
        "generation_config": asdict(generation_config),
        "auralis_url": auralis_url.rstrip("/"),
        "parakeet_url": parakeet_url.rstrip("/"),
        "asr_model": asr_model,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _wait_for_auralis(
    session: requests.Session,
    base_url: str,
    *,
    timeout_seconds: float,
    interval_seconds: float = 2.0,
) -> Mapping[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error = "no response"
    health_url = f"{base_url.rstrip('/')}/health"
    while time.monotonic() < deadline:
        try:
            response = session.get(health_url, timeout=min(5.0, timeout_seconds))
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    return data
                last_error = f"unexpected health payload: {data!r}"
            else:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except (requests.RequestException, ValueError) as exc:
            last_error = str(exc)
        time.sleep(interval_seconds)
    raise RuntimeError(f"Auralis health check failed at {health_url}: {last_error}")


def _validate_wav_bytes(content: bytes) -> None:
    if len(content) < 44 or content[:4] != b"RIFF" or content[8:12] != b"WAVE":
        raise ValueError("TTS response was not a valid RIFF/WAVE payload")


def generate_audio(
    session: requests.Session,
    task: BenchmarkTask,
    *,
    auralis_url: str,
    output_path: Path,
    timeout_seconds: float,
    generation_config: GenerationConfig,
) -> tuple[float, float]:
    """Generate one WAV and return (request latency, audio duration)."""

    payload = {
        "model": generation_config.model,
        "input": task.text,
        "voice": task.voice,
        "response_format": generation_config.response_format,
        "language": task.language,
        "temperature": generation_config.temperature,
        "top_p": generation_config.top_p,
        "top_k": generation_config.top_k,
        "repetition_penalty": generation_config.repetition_penalty,
        "max_ref_length": generation_config.max_ref_length,
        "gpt_cond_len": generation_config.gpt_cond_len,
        "gpt_cond_chunk_len": generation_config.gpt_cond_chunk_len,
        "do_sample": generation_config.do_sample,
        "apply_novasr": generation_config.apply_novasr,
    }
    started = time.perf_counter()
    response = session.post(
        f"{auralis_url.rstrip('/')}/v1/audio/speech",
        json=payload,
        timeout=timeout_seconds,
    )
    latency = time.perf_counter() - started
    if response.status_code != 200:
        raise RuntimeError(f"TTS HTTP {response.status_code}: {response.text[:500]}")
    _validate_wav_bytes(response.content)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_bytes(response.content)
    os.replace(temporary_path, output_path)
    with wave.open(str(output_path), "rb") as wav_file:
        frame_rate = wav_file.getframerate()
        duration = wav_file.getnframes() / frame_rate if frame_rate else 0.0
    if duration <= 0:
        raise ValueError("Generated WAV has zero duration")
    return latency, duration


def transcribe_audio(
    session: requests.Session,
    audio_path: Path,
    *,
    parakeet_url: str,
    asr_model: str,
    timeout_seconds: float,
) -> str:
    """Transcribe one WAV through an OpenAI-compatible ASR endpoint."""

    with audio_path.open("rb") as handle:
        response = session.post(
            f"{parakeet_url.rstrip('/')}/v1/audio/transcriptions",
            files={"file": (audio_path.name, handle, "audio/wav")},
            data={"model": asr_model, "response_format": "json"},
            timeout=timeout_seconds,
        )
    if response.status_code != 200:
        raise RuntimeError(f"ASR HTTP {response.status_code}: {response.text[:500]}")
    try:
        transcript = str(response.json().get("text", "")).strip()
    except ValueError as exc:
        raise RuntimeError(f"ASR returned invalid JSON: {response.text[:500]}") from exc
    if not transcript:
        raise RuntimeError("ASR returned an empty transcript")
    return transcript


def _load_checkpoint(path: Path, fingerprint: str, resume: bool) -> dict[str, Any]:
    if not resume or not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 2:
        raise RuntimeError(
            f"{path} is a legacy/incompatible result file. Use --no-resume or a new --tag."
        )
    if payload.get("benchmark_fingerprint") != fingerprint:
        raise RuntimeError(
            f"{path} was produced with a different corpus or configuration. "
            "Use --no-resume or a new --tag instead of mixing runs."
        )
    results = payload.get("results", {})
    if not isinstance(results, dict):
        raise RuntimeError(f"Invalid results mapping in {path}")
    return results


def _summarize_results(
    tag: str,
    tasks: Sequence[BenchmarkTask],
    results: Mapping[str, Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    successful = [result for result in results.values() if result.get("success")]
    latencies = [float(result["latency_s"]) for result in successful]
    durations = [float(result["audio_duration_s"]) for result in successful]
    utterance_wers = [float(result["wer"]) for result in successful]
    total_errors = sum(int(result["word_errors"]) for result in successful)
    total_reference_words = sum(int(result["reference_words"]) for result in successful)

    language_metrics: dict[str, Any] = {}
    for language in sorted({task.language for task in tasks}):
        language_results = [
            result
            for result in successful
            if str(result.get("lang", "")).lower() == language
        ]
        language_errors = sum(int(result["word_errors"]) for result in language_results)
        language_words = sum(int(result["reference_words"]) for result in language_results)
        language_wers = [float(result["wer"]) for result in language_results]
        language_metrics[language] = {
            "n_success": len(language_results),
            "corpus_wer": language_errors / language_words if language_words else None,
            "mean_utterance_wer": statistics.fmean(language_wers) if language_wers else None,
        }

    mean_wer = statistics.fmean(utterance_wers) if utterance_wers else None
    mean_latency = statistics.fmean(latencies) if latencies else None
    total_latency = sum(latencies)
    total_audio = sum(durations)
    summary = {
        "schema_version": 2,
        "tag": tag,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_total": len(tasks),
        "n_success": len(successful),
        "n_failed": len(tasks) - len(successful),
        "corpus_wer": total_errors / total_reference_words if total_reference_words else None,
        "mean_utterance_wer": mean_wer,
        "median_utterance_wer": statistics.median(utterance_wers) if utterance_wers else None,
        "mean_latency_s": mean_latency,
        "p95_latency_s": _percentile(latencies, 95),
        "mean_audio_duration_s": statistics.fmean(durations) if durations else None,
        "overall_rtf": total_latency / total_audio if total_audio else None,
        "mean_rtf": statistics.fmean(
            float(result["rtf"]) for result in successful
        ) if successful else None,
        "languages": language_metrics,
        # Legacy aliases keep old comparison tooling readable. They intentionally
        # preserve the old sentence-mean semantics rather than corpus WER.
        "mean_wer": mean_wer,
        "en_wer": language_metrics.get("en", {}).get("mean_utterance_wer"),
        "es_wer": language_metrics.get("es", {}).get("mean_utterance_wer"),
        "metadata": dict(metadata),
    }
    return summary


def _print_summary(summary: Mapping[str, Any]) -> None:
    def render(name: str, multiplier: float = 1.0, suffix: str = "") -> str:
        value = summary.get(name)
        return "N/A" if value is None else f"{float(value) * multiplier:.3f}{suffix}"

    print("\n" + "=" * 68)
    print(f"SUMMARY [{summary['tag']}]")
    print(f"  Success:              {summary['n_success']}/{summary['n_total']}")
    print(f"  Corpus WER:           {render('corpus_wer')}")
    print(f"  Mean utterance WER:   {render('mean_utterance_wer')}")
    print(f"  Mean latency:         {render('mean_latency_s', suffix='s')}")
    print(f"  P95 latency:          {render('p95_latency_s', suffix='s')}")
    print(f"  Overall speed:        {render('overall_rtf')} RTF")
    for language, metrics in summary.get("languages", {}).items():
        value = metrics.get("corpus_wer")
        rendered = "N/A" if value is None else f"{float(value):.3f}"
        print(f"  {language.upper():<3} corpus WER:       {rendered}")
    print("=" * 68)


def run_benchmark(
    *,
    tag: str,
    tasks: Sequence[BenchmarkTask],
    auralis_url: str,
    parakeet_url: str,
    asr_model: str,
    output_dir: Path,
    generation_config: GenerationConfig,
    request_timeout: float,
    health_timeout: float,
    warmup_samples: int,
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the benchmark and return (summary, checkpoint payload)."""

    if not TAG_PATTERN.fullmatch(tag):
        raise ValueError("--tag may contain only letters, numbers, dot, dash, and underscore")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"results_{tag}.json"
    summary_path = output_dir / f"summary_{tag}.json"
    fingerprint = _benchmark_fingerprint(
        tasks,
        generation_config,
        auralis_url=auralis_url,
        parakeet_url=parakeet_url,
        asr_model=asr_model,
    )
    results = _load_checkpoint(results_path, fingerprint, resume)
    metadata = {
        "benchmark_fingerprint": fingerprint,
        "auralis_url": auralis_url.rstrip("/"),
        "parakeet_url": parakeet_url.rstrip("/"),
        "asr_model": asr_model,
        "generation_config": asdict(generation_config),
    }

    with requests.Session() as session:
        health = _wait_for_auralis(
            session,
            auralis_url,
            timeout_seconds=health_timeout,
        )
        metadata["auralis_health_at_start"] = health

        for warmup_index, task in enumerate(tasks[: max(0, warmup_samples)]):
            warmup_path = output_dir / f".warmup_{warmup_index}.wav"
            print(f"[warmup] {task.task_id}")
            generate_audio(
                session,
                task,
                auralis_url=auralis_url,
                output_path=warmup_path,
                timeout_seconds=request_timeout,
                generation_config=generation_config,
            )
            transcribe_audio(
                session,
                warmup_path,
                parakeet_url=parakeet_url,
                asr_model=asr_model,
                timeout_seconds=request_timeout,
            )
            warmup_path.unlink(missing_ok=True)

        for index, task in enumerate(tasks, start=1):
            existing = results.get(task.task_id)
            if existing and existing.get("success"):
                print(f"[{index:03d}/{len(tasks):03d}] SKIP {task.task_id}")
                continue

            audio_path = output_dir / f"{tag}_{task.task_id}.wav"
            print(f"[{index:03d}/{len(tasks):03d}] {task.task_id} ({task.language}/{task.voice})")
            try:
                latency, duration = generate_audio(
                    session,
                    task,
                    auralis_url=auralis_url,
                    output_path=audio_path,
                    timeout_seconds=request_timeout,
                    generation_config=generation_config,
                )
                transcript = transcribe_audio(
                    session,
                    audio_path,
                    parakeet_url=parakeet_url,
                    asr_model=asr_model,
                    timeout_seconds=request_timeout,
                )
                errors, reference_words = _word_error_count(task.text, transcript)
                utterance_wer = errors / reference_words if reference_words else (0.0 if errors == 0 else 1.0)
                results[task.task_id] = {
                    "text": task.text,
                    "lang": task.language,
                    "voice": task.voice,
                    "latency_s": latency,
                    "audio_duration_s": duration,
                    "rtf": latency / duration,
                    "transcript": transcript,
                    "normalized_reference": normalize_text(task.text),
                    "normalized_hypothesis": normalize_text(transcript),
                    "word_errors": errors,
                    "reference_words": reference_words,
                    "wer": utterance_wer,
                    "success": True,
                }
                print(
                    f"  WER={utterance_wer:.3f} latency={latency:.2f}s "
                    f"audio={duration:.2f}s RTF={latency / duration:.3f}"
                )
            except Exception as exc:
                print(f"  FAILED: {exc}", file=sys.stderr)
                results[task.task_id] = {
                    "text": task.text,
                    "lang": task.language,
                    "voice": task.voice,
                    "success": False,
                    "error": str(exc),
                }

            checkpoint = {
                "schema_version": 2,
                "tag": tag,
                "benchmark_fingerprint": fingerprint,
                "metadata": metadata,
                "results": results,
            }
            _atomic_write_json(results_path, checkpoint)

    checkpoint = {
        "schema_version": 2,
        "tag": tag,
        "benchmark_fingerprint": fingerprint,
        "metadata": metadata,
        "results": results,
    }
    _atomic_write_json(results_path, checkpoint)
    summary = _summarize_results(tag, tasks, results, metadata)
    _atomic_write_json(summary_path, summary)
    _print_summary(summary)
    return summary, checkpoint


def compare_results(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    max_wer_regression_pct: float = 0.0,
    max_latency_regression_pct: float = 0.0,
) -> bool:
    """Print a lower-is-better comparison and return whether gates pass."""

    metrics = (
        ("corpus_wer", max_wer_regression_pct),
        ("mean_wer", max_wer_regression_pct),
        ("en_wer", max_wer_regression_pct),
        ("es_wer", max_wer_regression_pct),
        ("mean_latency_s", max_latency_regression_pct),
        ("p95_latency_s", max_latency_regression_pct),
    )
    print("\n" + "=" * 68)
    print("BASELINE vs CANDIDATE")
    print("=" * 68)
    passed = True
    compared: set[str] = set()
    for metric, allowed_regression in metrics:
        if metric in compared:
            continue
        before = baseline.get(metric)
        after = candidate.get(metric)
        if before is None or after is None:
            continue
        compared.add(metric)
        before_value = float(before)
        after_value = float(after)
        absolute_delta = after_value - before_value
        if before_value == 0:
            relative_delta = 0.0 if after_value == 0 else math.inf
        else:
            relative_delta = absolute_delta / before_value * 100.0
        metric_passed = relative_delta <= allowed_regression
        passed = passed and metric_passed
        relative_label = "∞" if math.isinf(relative_delta) else f"{relative_delta:+.1f}%"
        status = "PASS" if metric_passed else "REGRESSION"
        print(
            f"  {metric:22s} {before_value:.4f} -> {after_value:.4f} "
            f"({relative_label:>8s})  {status}"
        )
    if not compared:
        print("  No comparable metrics were found.")
        passed = False
    print("=" * 68)
    return passed


def _read_summary(output_dir: Path, tag: str) -> Mapping[str, Any]:
    path = output_dir / f"summary_{tag}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing summary file: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid summary file: {path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="candidate")
    parser.add_argument("--auralis-url", default=DEFAULT_AURALIS_URL)
    parser.add_argument("--parakeet-url", default=DEFAULT_PARAKEET_URL)
    parser.add_argument("--asr-model", default="parakeet-tdt-0.6b-v3")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--voice-en", default="nova")
    parser.add_argument("--voice-es", default="alloy")
    parser.add_argument("--limit-per-language", type=int)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--health-timeout", type=float, default=60.0)
    parser.add_argument("--warmup-samples", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Enable sampling; deterministic decoding is the default",
    )
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-p", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=5.0)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare baseline and candidate summaries after the run",
    )
    parser.add_argument("--baseline-tag", default="baseline")
    parser.add_argument("--candidate-tag", default=None, help="Defaults to --tag")
    parser.add_argument("--max-wer-regression-pct", type=float, default=0.0)
    parser.add_argument("--max-latency-regression-pct", type=float, default=0.0)
    parser.add_argument("--allow-regression", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tasks = load_corpus(
        args.corpus,
        english_voice=args.voice_en,
        spanish_voice=args.voice_es,
        limit_per_language=args.limit_per_language,
    )
    generation_config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.stochastic,
    )
    run_benchmark(
        tag=args.tag,
        tasks=tasks,
        auralis_url=args.auralis_url,
        parakeet_url=args.parakeet_url,
        asr_model=args.asr_model,
        output_dir=args.output_dir,
        generation_config=generation_config,
        request_timeout=args.request_timeout,
        health_timeout=args.health_timeout,
        warmup_samples=args.warmup_samples,
        resume=not args.no_resume,
    )

    if args.compare:
        baseline = _read_summary(args.output_dir, args.baseline_tag)
        candidate = _read_summary(args.output_dir, args.candidate_tag or args.tag)
        passed = compare_results(
            baseline,
            candidate,
            max_wer_regression_pct=args.max_wer_regression_pct,
            max_latency_regression_pct=args.max_latency_regression_pct,
        )
        if not passed and not args.allow_regression:
            return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
