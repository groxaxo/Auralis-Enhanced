#!/usr/bin/env python3
"""
Auralis Enhanced — Real-Time Speed Benchmark & Bottleneck Profiler
===================================================================
Measures:
  • Model load time
  • Audio conditioning time (mel + ConditioningEncoder + PerceiverResampler)
  • GPT token generation time  (vLLM first-token + total)
  • HiFi-GAN decode time  (per chunk + total)
  • NovaSR 24 → 48 kHz upsampling time
  • RTF  (real-time factor, lower is faster — <1 = faster-than-realtime)
  • VRAM before/after each stage

Profiling layers:
  1.  Wall-clock timing at every pipeline boundary
  2.  cProfile over the full generate_speech() call (top hotspots by cumtime)
  3.  Per-stage VRAM snapshots via pynvml
"""

import asyncio
import cProfile
import io
import os
import pstats
import sys
import time
from pathlib import Path

# ── GPU selection ────────────────────────────────────────────────────────────
# Use RTX 3060 (index 2) — has ≈6 GB free; model needs ≈4.2 GB
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import torch
import numpy as np

# ── VRAM helper ──────────────────────────────────────────────────────────────
try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # index 0 in CUDA_VISIBLE_DEVICES

    def vram_mib() -> float:
        info = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
        return info.used / 1024 ** 2

except Exception:
    def vram_mib() -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(0) / 1024 ** 2
        return 0.0


# ── Timing context ────────────────────────────────────────────────────────────
class Stage:
    """Context manager that prints stage time + VRAM delta."""

    _indent = 0

    def __init__(self, label: str):
        self.label = label
        self.elapsed = 0.0
        self.vram_before = 0.0
        self.vram_after = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.vram_before = vram_mib()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self._t0
        self.vram_after = vram_mib()
        delta = self.vram_after - self.vram_before
        sign = "+" if delta >= 0 else ""
        print(
            f"  ⏱  {self.label:<45s}  {self.elapsed*1000:8.1f} ms   "
            f"VRAM {self.vram_after:6.0f} MiB ({sign}{delta:.0f})"
        )


# ── Test texts ────────────────────────────────────────────────────────────────
TEXTS = {
    "short (10 words)":
        "Hello, this is a quick test of the TTS engine.",
    "medium (50 words)":
        ("Auralis Enhanced is a production-ready text-to-speech engine. "
         "It uses vLLM for efficient token generation and supports voice cloning. "
         "The NovaSR module upsamples 24 kHz audio to 48 kHz with negligible overhead, "
         "giving broadcast-quality results at well above realtime speed."),
    "long (100 words)":
        ("In the field of artificial intelligence, text-to-speech synthesis has undergone "
         "remarkable transformation over the past decade. Modern neural architectures "
         "combine autoregressive language models with vocoder networks to produce audio "
         "that is virtually indistinguishable from human speech. Auralis Enhanced builds "
         "on the XTTS v2 architecture, using vLLM for efficient multi-request scheduling "
         "and a lightweight NovaSR model to super-resolve 24 kHz mel features into "
         "full-bandwidth 48 kHz waveforms. The result is a system that can process an "
         "entire novel in minutes while running entirely on local hardware with no "
         "external API calls."),
}

SPEAKER_FILE = Path(__file__).parent / "tests/resources/audio_samples/female.wav"
TTS_MODEL    = "AstraMindAI/xttsv2"
GPT_MODEL    = "AstraMindAI/xtts2-gpt"


# ── Patch process_tokens_to_speech to time HiFi-GAN per-chunk ─────────────────
_hifi_times: list[float] = []

def _patch_hifigan(engine):
    """Monkey-patch HifiDecoder.forward to record latency per chunk."""
    original_forward = engine.hifigan_decoder.forward

    def timed_forward(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = original_forward(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _hifi_times.append(time.perf_counter() - t0)
        return result

    engine.hifigan_decoder.forward = timed_forward


# ── Main benchmark ─────────────────────────────────────────────────────────────
def run_benchmark():
    from auralis import TTS, TTSRequest

    print("\n" + "=" * 72)
    print("  🚀  Auralis Enhanced — Speed Benchmark")
    print("=" * 72)
    print(f"  GPU : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  VRAM free at start: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1024**2:.0f} MiB"
          if torch.cuda.is_available() else "  No CUDA")
    print("=" * 72)

    # ── 1. Model load ─────────────────────────────────────────────────────────
    print("\n📦  MODEL LOAD")
    enforce_eager_val = bool(int(os.environ.get("ENFORCE_EAGER", "0")))  # default: CUDA graphs
    num_scheduler_steps_val = int(os.environ.get("NUM_SCHEDULER_STEPS", "1"))
    print(f"  enforce_eager={enforce_eager_val}  num_scheduler_steps={num_scheduler_steps_val}")
    with Stage("TTS.from_pretrained (full stack)") as s_load:
        tts = TTS().from_pretrained(
            TTS_MODEL, gpt_model=GPT_MODEL,
            enforce_eager=enforce_eager_val,
            num_scheduler_steps=num_scheduler_steps_val,
        )
    load_time = s_load.elapsed

    _patch_hifigan(tts.tts_engine)
    print(f"\n  Model loaded in {load_time:.2f}s\n")

    # ── 2. Per-text benchmarks ────────────────────────────────────────────────
    results = {}

    for label, text in TEXTS.items():
        print(f"\n{'─'*72}")
        print(f"  📝  {label}")
        print(f"      \"{text[:70]}{'…' if len(text)>70 else ''}\"")
        print(f"{'─'*72}")

        n_words  = len(text.split())
        # rough estimate: ~0.4 s / word at natural speech pace
        audio_dur_est = n_words * 0.4

        _hifi_times.clear()
        chunk_sizes = []
        ttfc = None  # time-to-first-chunk
        total_audio_samples = 0

        # ── conditioning ──────────────────────────────────────────────────────
        with Stage("Audio conditioning (mel+encoder+perceiver)") as s_cond:
            loop = tts.loop
            gpt_cond_latent, speaker_embeddings = loop.run_until_complete(
                tts.tts_engine.get_audio_conditioning([str(SPEAKER_FILE)])
            )

        # ── streaming generation (measures TTFC + chunks) ─────────────────────
        req = TTSRequest(
            text=text,
            speaker_files=[str(SPEAKER_FILE)],
            stream=True,
        )

        t_gen_start = time.perf_counter()
        chunk_count = 0

        def _run_streaming():
            nonlocal ttfc, total_audio_samples, chunk_count
            for chunk in tts.generate_speech(req):
                if ttfc is None:
                    ttfc = time.perf_counter() - t_gen_start
                chunk_sizes.append(len(chunk.array))
                total_audio_samples += len(chunk.array)
                chunk_count += 1

        with Stage("Full streaming generate (total wall-clock)") as s_total:
            _run_streaming()

        total_time = s_total.elapsed
        audio_dur_actual = total_audio_samples / 24000  # 24 kHz base rate

        # ── per-stage summary ─────────────────────────────────────────────────
        hifi_total = sum(_hifi_times)
        hifi_avg   = (hifi_total / len(_hifi_times) * 1000) if _hifi_times else 0

        print(f"\n  📊  RESULTS:")
        print(f"      TTFC (time-to-first-chunk)  : {ttfc*1000:.0f} ms")
        print(f"      Total generation time       : {total_time:.2f}s")
        print(f"      Audio duration (generated)  : {audio_dur_actual:.2f}s")
        rtf = total_time / audio_dur_actual if audio_dur_actual > 0 else float('nan')
        print(f"      RTF                         : {rtf:.3f}x  ({'faster' if rtf < 1 else 'SLOWER'} than realtime)")
        print(f"      Chunks produced             : {chunk_count}")
        print(f"      Avg chunk size              : {np.mean(chunk_sizes):.0f} samples")
        print(f"      HiFi-GAN decode — total     : {hifi_total*1000:.0f} ms ({hifi_total/total_time*100:.1f}% of total)")
        print(f"      HiFi-GAN decode — avg/chunk : {hifi_avg:.1f} ms")
        print(f"      Conditioning time           : {s_cond.elapsed*1000:.0f} ms ({s_cond.elapsed/total_time*100:.1f}% of total)")

        vllm_est = total_time - hifi_total - s_cond.elapsed
        print(f"      vLLM token-gen (est.)       : {vllm_est*1000:.0f} ms ({vllm_est/total_time*100:.1f}% of total)")

        results[label] = {
            "ttfc_ms": ttfc * 1000,
            "total_s": total_time,
            "audio_s": audio_dur_actual,
            "rtf": rtf,
            "hifi_pct": hifi_total / total_time * 100,
            "cond_pct": s_cond.elapsed / total_time * 100,
            "vllm_pct": vllm_est / total_time * 100,
        }

    # ── 3. cProfile hotspot analysis ─────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  🔬  cPROFILE HOTSPOT ANALYSIS  (medium text, non-streaming)")
    print("="*72)

    req_prof = TTSRequest(
        text=TEXTS["medium (50 words)"],
        speaker_files=[str(SPEAKER_FILE)],
        stream=False,
    )

    profiler = cProfile.Profile()
    profiler.enable()
    _ = tts.generate_speech(req_prof)
    profiler.disable()

    stream = io.StringIO()
    ps = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    ps.print_stats(30)
    profile_output = stream.getvalue()

    # Filter to only interesting lines (skip stdlib boilerplate)
    skip_tokens = ("/runpy.py", "importlib", "codeop", "<frozen", "linecache",
                   "pkgutil", "genericpath", "posixpath", "abc.py")
    lines = profile_output.splitlines()
    print("\n  Top functions by cumulative time:")
    print("  " + "-"*68)
    count = 0
    for line in lines:
        if count < 5 or not any(tok in line for tok in skip_tokens):
            print("  " + line)
            count += 1
            if count > 50:
                break

    # ── 4. Bottleneck summary ─────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  🏁  BOTTLENECK SUMMARY")
    print("="*72)

    for label, r in results.items():
        print(f"\n  [{label}]")
        bottlenecks = sorted(
            [("vLLM token gen",  r["vllm_pct"]),
             ("HiFi-GAN decode", r["hifi_pct"]),
             ("Conditioning",    r["cond_pct"])],
            key=lambda x: x[1], reverse=True
        )
        for name, pct in bottlenecks:
            bar = "█" * int(pct / 2)
            print(f"    {name:<20s}  {pct:5.1f}%  {bar}")
        print(f"    RTF = {r['rtf']:.3f}x   TTFC = {r['ttfc_ms']:.0f} ms")

    print(f"\n  Model load time: {load_time:.2f}s\n")


if __name__ == "__main__":
    run_benchmark()
