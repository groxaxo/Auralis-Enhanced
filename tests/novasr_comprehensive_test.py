#!/usr/bin/env python3
"""
Comprehensive NovaSR Test Suite
================================

Tests all aspects of NovaSR integration without requiring vLLM.
"""

import os
import sys
import time
import json
import traceback
from pathlib import Path

import numpy as np
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Use RTX 3060

print("=" * 70)
print("🧪 NovaSR Comprehensive Test Suite")
print("=" * 70)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # Will be RTX 3060 due to CUDA_VISIBLE_DEVICES
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
print("=" * 70)


class TestResults:
    def __init__(self):
        self.results = {}
        self.passed = 0
        self.failed = 0

    def record(self, name, success, details=""):
        self.results[name] = {"success": success, "details": details}
        if success:
            self.passed += 1
            print(f"   ✅ {name}: PASS")
        else:
            self.failed += 1
            print(f"   ❌ {name}: FAIL - {details}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n   Total: {self.passed}/{total} tests passed")
        return self.failed == 0


results = TestResults()


# ============================================================
# TEST 1: Direct NovaSR Model Test
# ============================================================
print("\n📦 TEST 1: Direct NovaSR Model")
print("-" * 40)

try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "NovaSR"))
    from NovaSR import FastSR

    # Test 1a: Model Loading
    print("   Loading model...")
    start = time.perf_counter()
    upsampler = FastSR(half=True)
    load_time = time.perf_counter() - start
    results.record("1a. Model Loading", True, f"{load_time:.2f}s")

    # Test 1b: Basic Inference
    t = np.linspace(0, 1, 16000)
    audio_16k = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5
    audio_tensor = (
        torch.from_numpy(audio_16k).float().unsqueeze(0).unsqueeze(1).cuda().half()
    )

    start = time.perf_counter()
    with torch.no_grad():
        enhanced = upsampler.infer(audio_tensor)
    process_time = time.perf_counter() - start

    enhanced_np = enhanced.cpu().float().numpy().squeeze()
    expected_len = 48000

    results.record(
        "1b. Basic Inference",
        len(enhanced_np) >= expected_len * 0.99,
        f"Output: {len(enhanced_np)} samples (expected ~{expected_len})",
    )

    # Test 1c: RTF Performance
    rtf = process_time / 1.0
    results.record(
        "1c. RTF < 0.1x", rtf < 0.1, f"RTF: {rtf:.4f}x ({1 / rtf:.0f}x realtime)"
    )

    # Test 1d: Output Range Check
    max_val = np.max(np.abs(enhanced_np))
    results.record(
        "1d. Output in valid range", max_val <= 1.0, f"Max amplitude: {max_val:.4f}"
    )

except Exception as e:
    results.record("1. Direct NovaSR Model", False, str(e))
    traceback.print_exc()


# ============================================================
# TEST 2: NovaSR Processor Module
# ============================================================
print("\n📦 TEST 2: NovaSR Processor Module")
print("-" * 40)

try:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "novasr", "/home/op/Auralis-Enhanced/src/auralis/common/enhancers/novasr.py"
    )
    novasr_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(novasr_module)

    NovaSRProcessor = novasr_module.NovaSRProcessor
    get_novasr_processor = novasr_module.get_novasr_processor

    # Test 2a: Processor Initialization
    processor = NovaSRProcessor(device="cuda")
    results.record("2a. Processor Initialization", True, f"Device: {processor.device}")

    # Test 2b: Fallback Upsampling
    audio_16k = np.random.randn(16000).astype(np.float32) * 0.5
    upsampled = processor._fallback_upsample(audio_16k)
    results.record(
        "2b. Fallback Upsampling",
        len(upsampled) == 48000,
        f"Output: {len(upsampled)} samples",
    )

    # Test 2c: Model Loading via Processor
    start = time.perf_counter()
    processor.load_model()
    load_time = time.perf_counter() - start
    results.record(
        "2c. Model Loading via Processor",
        processor.is_available(),
        f"Load time: {load_time:.2f}s",
    )

    # Test 2d: Process Method
    start = time.perf_counter()
    enhanced, sr = processor.process(audio_16k, sr=16000)
    process_time = time.perf_counter() - start

    results.record(
        "2d. Process Method",
        sr == 48000 and len(enhanced) >= 47900,
        f"SR: {sr}, Samples: {len(enhanced)}",
    )

    # Test 2e: Singleton Pattern
    proc1 = get_novasr_processor()
    proc2 = get_novasr_processor()
    results.record("2e. Singleton Pattern", proc1 is proc2)

    # Test 2f: Invalid Sample Rate
    try:
        processor.process(audio_16k, sr=22050)
        results.record(
            "2f. Invalid SR Rejection", False, "Should have raised ValueError"
        )
    except ValueError as e:
        results.record("2f. Invalid SR Rejection", True, str(e))

except Exception as e:
    results.record("2. NovaSR Processor Module", False, str(e))
    traceback.print_exc()


# ============================================================
# TEST 3: TTSOutput Integration
# ============================================================
print("\n📦 TEST 3: TTSOutput Integration")
print("-" * 40)

try:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "output", "/home/op/Auralis-Enhanced/src/auralis/common/definitions/output.py"
    )
    output_module = importlib.util.module_from_spec(spec)

    # Mock dependencies
    class MockProcessor:
        def process(self, audio, sr):
            return np.zeros(48000, dtype=np.float32), 48000

    sys.modules["auralis.common.enhancers.novasr"] = type(sys)("mock")
    sys.modules["auralis.common.enhancers.novasr"].get_novasr_processor = (
        lambda **kwargs: MockProcessor()
    )

    spec.loader.exec_module(output_module)
    TTSOutput = output_module.TTSOutput

    audio = np.random.randn(24000).astype(np.float32)

    # Test 3a: Has _novasr_applied
    output = TTSOutput(array=audio, sample_rate=24000)
    results.record("3a. Has _novasr_applied", hasattr(output, "_novasr_applied"))

    # Test 3b: Default value is False
    results.record("3b. Default _novasr_applied=False", output._novasr_applied == False)

    # Test 3c: Has apply_super_resolution method
    results.record(
        "3c. Has apply_super_resolution", hasattr(output, "apply_super_resolution")
    )

    # Test 3d: Invalid method raises error
    try:
        output.apply_super_resolution(method="invalid")
        results.record("3d. Invalid method rejected", False)
    except ValueError:
        results.record("3d. Invalid method rejected", True)

except Exception as e:
    results.record("3. TTSOutput Integration", False, str(e))
    traceback.print_exc()


# ============================================================
# TEST 4: TTSRequest Integration
# ============================================================
print("\n📦 TEST 4: TTSRequest Integration")
print("-" * 40)

try:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "requests",
        "/home/op/Auralis-Enhanced/src/auralis/common/definitions/requests.py",
    )
    requests_module = importlib.util.module_from_spec(spec)

    # Mock dependencies
    from dataclasses import dataclass, field

    @dataclass
    class MockAudioConfig:
        sample_rate: int = 22050

    sys.modules["auralis.common.definitions.enhancer"] = type(sys)("mock")
    sys.modules[
        "auralis.common.definitions.enhancer"
    ].AudioPreprocessingConfig = MockAudioConfig
    sys.modules["auralis.common.definitions.enhancer"].EnhancedAudioProcessor = (
        lambda x: None
    )

    spec.loader.exec_module(requests_module)
    TTSRequest = requests_module.TTSRequest

    # Test 4a: Has apply_novasr
    request = TTSRequest(text="Test", speaker_files=["test.wav"])
    results.record("4a. Has apply_novasr", hasattr(request, "apply_novasr"))

    # Test 4b: Default is False
    results.record("4b. Default apply_novasr=False", request.apply_novasr == False)

    # Test 4c: Can be set to True
    request_enabled = TTSRequest(
        text="Test", speaker_files=["test.wav"], apply_novasr=True
    )
    results.record("4c. Can enable apply_novasr", request_enabled.apply_novasr == True)

except Exception as e:
    results.record("4. TTSRequest Integration", False, str(e))
    traceback.print_exc()


# ============================================================
# TEST 5: Audio Quality Tests
# ============================================================
print("\n📦 TEST 5: Audio Quality Tests")
print("-" * 40)

try:
    import torchaudio

    # Test 5a: Generate and save test audio
    output_dir = Path("/home/op/Auralis-Enhanced/test_output")
    output_dir.mkdir(exist_ok=True)

    # Generate sine wave at 440Hz
    t = np.linspace(0, 2, 16000 * 2)  # 2 seconds
    audio_16k = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5

    # Process with NovaSR
    processor = get_novasr_processor()
    enhanced, sr = processor.process(audio_16k, sr=16000)

    # Save
    output_file = output_dir / "novasr_quality_test.wav"
    enhanced_tensor = torch.from_numpy(enhanced).unsqueeze(0).float()
    torchaudio.save(str(output_file), enhanced_tensor, 48000)

    results.record("5a. Save 48kHz audio", output_file.exists(), str(output_file))

    # Test 5b: Verify file size
    file_size = output_file.stat().st_size
    expected_size = 48000 * 2 * 2 + 44  # 2 seconds * 48000 * 2 bytes + header
    results.record(
        "5b. File size reasonable",
        file_size > expected_size * 0.9,
        f"Size: {file_size} bytes",
    )

    # Test 5c: Reload and verify
    reloaded, reload_sr = torchaudio.load(str(output_file))
    results.record("5c. Reload sample rate", reload_sr == 48000, f"SR: {reload_sr}")

    # Test 5d: Check audio isn't silent
    is_not_silent = torch.max(torch.abs(reloaded)) > 0.01
    results.record(
        "5d. Audio not silent",
        is_not_silent,
        f"Max: {torch.max(torch.abs(reloaded)):.4f}",
    )

except Exception as e:
    results.record("5. Audio Quality Tests", False, str(e))
    traceback.print_exc()


# ============================================================
# TEST 6: Whisper Transcription Correlation
# ============================================================
print("\n📦 TEST 6: Whisper Transcription Correlation")
print("-" * 40)

try:
    import whisper

    # Load Whisper
    print("   Loading Whisper model...")
    model = whisper.load_model("base")
    results.record("6a. Whisper loaded", True)

    samples_dir = Path("/home/op/Auralis-Enhanced/samples")
    similarities = []

    for i in range(1, 6):
        base_file = samples_dir / f"sample_en_{i}_base.wav"
        enhanced_file = samples_dir / f"sample_en_{i}_enhanced.wav"

        if base_file.exists() and enhanced_file.exists():
            result_base = model.transcribe(str(base_file))
            result_enhanced = model.transcribe(str(enhanced_file))

            base_words = set(result_base["text"].lower().split())
            enhanced_words = set(result_enhanced["text"].lower().split())
            common = base_words & enhanced_words
            similarity = len(common) / max(len(base_words), len(enhanced_words), 1)
            similarities.append(similarity)

            print(f"   Sample {i}: {similarity * 100:.1f}% similarity")

    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    results.record(
        "6b. Avg similarity > 80%", avg_similarity > 0.8, f"{avg_similarity * 100:.1f}%"
    )
    results.record(
        "6c. Avg similarity > 85%",
        avg_similarity > 0.85,
        f"{avg_similarity * 100:.1f}%",
    )

except ImportError:
    results.record("6. Whisper Tests", False, "Whisper not installed")
except Exception as e:
    results.record("6. Whisper Tests", False, str(e))


# ============================================================
# TEST 7: Performance Benchmarks
# ============================================================
print("\n📦 TEST 7: Performance Benchmarks")
print("-" * 40)

try:
    processor = get_novasr_processor()

    # Test different audio lengths
    durations = [0.5, 1.0, 2.0, 5.0]
    rtfs = []

    for duration in durations:
        samples = int(16000 * duration)
        audio = np.random.randn(samples).astype(np.float32) * 0.5

        start = time.perf_counter()
        enhanced, sr = processor.process(audio, sr=16000)
        process_time = time.perf_counter() - start

        rtf = process_time / duration
        rtfs.append(rtf)
        print(f"   {duration}s audio: RTF={rtf:.4f}x ({1 / rtf:.0f}x realtime)")

    avg_rtf = sum(rtfs) / len(rtfs)
    results.record("7a. Avg RTF < 0.1x", avg_rtf < 0.1, f"{avg_rtf:.4f}x")
    results.record("7b. Max RTF < 0.2x", max(rtfs) < 0.2, f"{max(rtfs):.4f}x")

    # Memory test
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated() / 1024**2

    for _ in range(10):
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        enhanced, sr = processor.process(audio, sr=16000)

    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated() / 1024**2
    mem_increase = mem_after - mem_before

    results.record(
        "7c. Memory stable (<100MB increase)",
        mem_increase < 100,
        f"+{mem_increase:.1f}MB",
    )

except Exception as e:
    results.record("7. Performance Benchmarks", False, str(e))
    traceback.print_exc()


# ============================================================
# TEST 8: Edge Cases
# ============================================================
print("\n📦 TEST 8: Edge Cases")
print("-" * 40)

try:
    processor = get_novasr_processor()

    # Test 8a: Very short audio
    short_audio = np.random.randn(1600).astype(np.float32) * 0.5  # 0.1s
    enhanced, sr = processor.process(short_audio, sr=16000)
    results.record(
        "8a. Very short audio (0.1s)", len(enhanced) > 0, f"{len(enhanced)} samples"
    )

    # Test 8b: Silence
    silence = np.zeros(16000, dtype=np.float32)
    enhanced, sr = processor.process(silence, sr=16000)
    is_still_quiet = np.max(np.abs(enhanced)) < 0.01
    results.record(
        "8b. Silence preserved", is_still_quiet, f"Max: {np.max(np.abs(enhanced)):.6f}"
    )

    # Test 8c: Max amplitude
    max_amp = np.ones(16000, dtype=np.float32) * 0.99
    enhanced, sr = processor.process(max_amp, sr=16000)
    no_clipping = np.max(np.abs(enhanced)) <= 1.0
    results.record(
        "8c. No clipping on max input",
        no_clipping,
        f"Max: {np.max(np.abs(enhanced)):.4f}",
    )

    # Test 8d: DC offset
    dc_offset = np.ones(16000, dtype=np.float32) * 0.3
    enhanced, sr = processor.process(dc_offset, sr=16000)
    results.record("8d. DC offset handled", len(enhanced) > 0)

except Exception as e:
    results.record("8. Edge Cases", False, str(e))
    traceback.print_exc()


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("📋 FINAL SUMMARY")
print("=" * 70)

all_passed = results.summary()

# Save results
output_dir = Path("/home/op/Auralis-Enhanced/test_output")
output_dir.mkdir(exist_ok=True)
results_file = output_dir / "novasr_comprehensive_results.json"

output = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "passed": results.passed,
    "failed": results.failed,
    "results": results.results,
}

with open(results_file, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n📄 Results saved to: {results_file}")
print("=" * 70)

sys.exit(0 if all_passed else 1)
