"""Throughput benchmark for Auralis Enhanced (Blackwell port).

Measures concurrent and serial real-time factor (RTF) on a configurable
number of TTS requests, reports per-batch and per-request timings, and
sanity-checks the output audio.

Usage:
    python scripts/benchmark.py [--conc N] [--n M] [--novasr] [--no-serial]
                                [--ref PATH] [--model NAME] [--gpt-model NAME]
                                [--out PATH]

Examples:
    # Single-instance throughput sweep at concurrency 32
    python scripts/benchmark.py --conc 32 --n 64 --no-serial

    # 48 kHz NovaSR run on a custom reference voice
    python scripts/benchmark.py --conc 16 --n 16 --novasr --ref my_voice.wav

The script prints lines starting with ``RESULT`` for easy grepping in CI:

    RESULT conc=16 n=16 novasr=False
      concurrent: audio=...s wall=...s RTF=...x (...x realtime) ...
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import warnings

import numpy as np

# Silence the noisy default vLLM/transformers warnings; the bench output is
# the signal we want here.
warnings.filterwarnings("ignore")

from auralis import TTS, TTSRequest  # noqa: E402


DEFAULT_REF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "voice_library", "facu", "sample_0.mp3",
)


def build_sentences(n: int) -> list[str]:
    """Reproducible sentence list with enough variation to defeat KV reuse."""
    template = (
        "Sentence number {i}: in the heart of silicon valley a new chapter of "
        "computing unfolds, voice cloning technology has advanced rapidly."
    )
    return [template.format(i=i + 1) for i in range(n)]


def run_serial(tts: TTS, sentences: list[str], ref: str,
               novasr: bool) -> tuple[float, float]:
    audio_total = 0.0
    t0 = time.time()
    for text in sentences:
        out = tts.generate_speech(TTSRequest(
            text=text, speaker_files=[ref], language="en",
            apply_novasr=novasr))
        audio_total += len(out.array) / out.sample_rate
    return audio_total, time.time() - t0


def run_concurrent(tts: TTS, sentences: list[str], ref: str,
                   novasr: bool):
    reqs = [TTSRequest(text=text, speaker_files=[ref], language="en",
                       apply_novasr=novasr) for text in sentences]
    t0 = time.time()
    out = tts.loop.run_until_complete(tts._process_multiple_requests(reqs))
    return out, time.time() - t0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--conc", type=int, default=8,
                   help="vLLM max_num_seqs / scheduler concurrency (default 8)")
    p.add_argument("--n", type=int, default=16,
                   help="number of TTS requests to submit (default 16)")
    p.add_argument("--novasr", action="store_true",
                   help="enable NovaSR 48 kHz super-resolution on every chunk")
    p.add_argument("--no-serial", action="store_true",
                   help="skip the serial baseline pass")
    p.add_argument("--ref", default=DEFAULT_REF,
                   help=f"reference voice audio (default: {DEFAULT_REF})")
    p.add_argument("--model", default="AstraMindAI/xttsv2",
                   help="XTTSv2 HiFi-GAN checkpoint")
    p.add_argument("--gpt-model", default="AstraMindAI/xtts2-gpt",
                   help="XTTSv2 GPT checkpoint")
    p.add_argument("--out", default=None,
                   help="optional WAV path to save the concatenated output")
    args = p.parse_args()

    if not os.path.exists(args.ref):
        print(f"ERROR: reference audio not found: {args.ref}", file=sys.stderr)
        return 2

    print(f"Loading model (conc={args.conc}, novasr={args.novasr})...",
          flush=True)
    t0 = time.time()
    tts = TTS(scheduler_max_concurrency=args.conc).from_pretrained(
        args.model, gpt_model=args.gpt_model, max_concurrency=args.conc)
    load_time = time.time() - t0
    print(f"Model load: {load_time:.2f}s", flush=True)

    # Warmup pass primes caches, downloads NovaSR weights on first use, and
    # flushes the vLLM profile run.
    _ = tts.generate_speech(TTSRequest(
        text="warm up", speaker_files=[args.ref], language="en",
        apply_novasr=args.novasr))
    print("Warmup complete", flush=True)

    sentences = build_sentences(args.n)

    serial_audio = serial_wall = None
    if not args.no_serial:
        print(f"\n=== Serial ({args.n} requests one-by-one) ===", flush=True)
        serial_audio, serial_wall = run_serial(tts, sentences, args.ref,
                                               args.novasr)
        print(f"  serial:     audio={serial_audio:.2f}s wall={serial_wall:.2f}s "
              f"RTF={serial_wall / serial_audio:.3f}x "
              f"({serial_audio / serial_wall:.2f}x realtime)", flush=True)

    print(f"\n=== Concurrent ({args.n} requests, conc={args.conc}) ===",
          flush=True)
    out, conc_wall = run_concurrent(tts, sentences, args.ref, args.novasr)
    conc_audio = len(out.array) / out.sample_rate
    arr = np.asarray(out.array)
    rms = float(np.sqrt((arr ** 2).mean()))
    peak = float(np.abs(arr).max())

    print(f"\nRESULT conc={args.conc} n={args.n} novasr={args.novasr}",
          flush=True)
    if serial_audio is not None:
        print(f"  serial:     audio={serial_audio:.2f}s wall={serial_wall:.2f}s "
              f"RTF={serial_wall / serial_audio:.3f}x "
              f"({serial_audio / serial_wall:.2f}x realtime)", flush=True)
    print(f"  concurrent: audio={conc_audio:.2f}s wall={conc_wall:.2f}s "
          f"RTF={conc_wall / conc_audio:.3f}x "
          f"({conc_audio / conc_wall:.2f}x realtime) "
          f"rms={rms:.3f} peak={peak:.3f}", flush=True)
    if serial_audio is not None and serial_wall > 0:
        speedup = (conc_audio / conc_wall) / (serial_audio / serial_wall)
        print(f"  throughput speedup: {speedup:.2f}x", flush=True)

    if args.out:
        out.save(args.out)
        print(f"  saved concatenated audio to {args.out}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
