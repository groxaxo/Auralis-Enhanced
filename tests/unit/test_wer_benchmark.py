import json
from pathlib import Path

import pytest

import wer_benchmark as benchmark


def test_normalize_text_splits_hyphens_and_removes_apostrophes():
    assert benchmark.normalize_text("Deep-sea Earth's test!") == "deep sea earths test"
    assert benchmark.wer("deep-sea fish", "deep sea fish") == 0.0


def test_word_error_count_supports_corpus_wer():
    errors_a, words_a = benchmark._word_error_count("one two", "one")
    errors_b, words_b = benchmark._word_error_count("three four five six", "three four five")
    assert (errors_a, words_a) == (1, 2)
    assert (errors_b, words_b) == (1, 4)
    assert (errors_a + errors_b) / (words_a + words_b) == pytest.approx(1 / 3)
    mean_utterance_wer = (
        benchmark.wer("one two", "one")
        + benchmark.wer("three four five six", "three four five")
    ) / 2
    assert mean_utterance_wer == pytest.approx(0.375)


def test_load_corpus_reads_legacy_result_mapping(tmp_path: Path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            {
                "en_00": {"text": "Hello world.", "lang": "en", "wer": 0.0},
                "es_00": {"text": "Hola mundo.", "lang": "es", "wer": 0.0},
            }
        ),
        encoding="utf-8",
    )
    tasks = benchmark.load_corpus(
        corpus_path,
        english_voice="nova",
        spanish_voice="alloy",
    )
    assert [(task.task_id, task.voice) for task in tasks] == [
        ("en_00", "nova"),
        ("es_00", "alloy"),
    ]


def test_checkpoint_rejects_legacy_results(tmp_path: Path):
    result_path = tmp_path / "results_candidate.json"
    result_path.write_text(
        json.dumps({"en_00": {"success": True}}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="legacy/incompatible"):
        benchmark._load_checkpoint(result_path, "fingerprint", resume=True)


def test_summary_keeps_zero_wer_and_reports_corpus_metric():
    tasks = [
        benchmark.BenchmarkTask("en_00", "hello world", "en", "nova"),
        benchmark.BenchmarkTask("en_01", "good morning", "en", "nova"),
    ]
    results = {
        "en_00": {
            "success": True,
            "lang": "en",
            "latency_s": 1.0,
            "audio_duration_s": 2.0,
            "rtf": 0.5,
            "word_errors": 0,
            "reference_words": 2,
            "wer": 0.0,
        },
        "en_01": {
            "success": True,
            "lang": "en",
            "latency_s": 2.0,
            "audio_duration_s": 2.0,
            "rtf": 1.0,
            "word_errors": 1,
            "reference_words": 2,
            "wer": 0.5,
        },
    }
    summary = benchmark._summarize_results("candidate", tasks, results, {})
    assert summary["corpus_wer"] == pytest.approx(0.25)
    assert summary["mean_utterance_wer"] == pytest.approx(0.25)
    assert summary["p95_latency_s"] == pytest.approx(1.95)


def test_compare_results_fails_on_regression():
    baseline = {"mean_wer": 0.05, "mean_latency_s": 5.0}
    candidate = {"mean_wer": 0.06, "mean_latency_s": 4.0}
    assert not benchmark.compare_results(baseline, candidate)
    assert benchmark.compare_results(
        baseline,
        candidate,
        max_wer_regression_pct=25.0,
    )
