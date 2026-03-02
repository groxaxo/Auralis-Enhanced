#!/usr/bin/env python3
"""
NovaSR Integration Test Script
==============================

This script tests the NovaSR integration in Auralis Enhanced:
1. Tests NovaSR processor directly
2. Generates TTS audio with and without NovaSR
3. Measures performance metrics
4. Uses Whisper to verify audio quality/transcription

Usage:
    python tests/novasr_integration_test.py
"""

import os
import sys
import time
import asyncio
import json
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

print("=" * 70)
print("🧪 NovaSR Integration Test")
print("=" * 70)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  - {torch.cuda.get_device_name(i)}")
print("=" * 70)


def test_novasr_processor():
    """Test NovaSR processor directly."""
    print("\n📦 Test 1: NovaSR Processor")
    print("-" * 40)

    try:
        from auralis.common.enhancers.novasr import (
            NovaSRProcessor,
            get_novasr_processor,
        )

        # Test singleton
        processor1 = get_novasr_processor()
        processor2 = get_novasr_processor()
        assert processor1 is processor2, "Singleton pattern failed"
        print("✅ Singleton pattern works")

        # Test processor initialization
        processor = NovaSRProcessor(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"✅ Processor initialized on {processor.device}")

        # Test fallback upsampling
        audio_16k = np.random.randn(16000).astype(np.float32)  # 1 second
        upsampled = processor._fallback_upsample(audio_16k)
        assert len(upsampled) == 48000, f"Expected 48000 samples, got {len(upsampled)}"
        print("✅ Fallback upsampling works (16kHz -> 48kHz)")

        return True, processor

    except Exception as e:
        print(f"❌ NovaSR processor test failed: {e}")
        traceback.print_exc()
        return False, None


def test_novasr_model_loading(processor):
    """Test loading the actual NovaSR model."""
    print("\n📦 Test 2: NovaSR Model Loading")
    print("-" * 40)

    try:
        print("Loading NovaSR model from Hugging Face...")
        start = time.perf_counter()
        processor.load_model()
        load_time = time.perf_counter() - start
        print(f"✅ Model loaded in {load_time:.2f}s")

        if processor.is_available():
            print("✅ Model is available and ready")

            # Test actual processing
            audio_16k = np.random.randn(16000).astype(np.float32) * 0.5
            start = time.perf_counter()
            enhanced, sr = processor.process(audio_16k, sr=16000)
            process_time = time.perf_counter() - start

            print(f"✅ Processed 1s audio in {process_time * 1000:.2f}ms")
            print(f"   Input: {len(audio_16k)} samples @ 16kHz")
            print(f"   Output: {len(enhanced)} samples @ {sr}Hz")

            # Calculate RTF
            rtf = process_time / (len(audio_16k) / 16000)
            print(f"   RTF: {rtf:.4f}x ({1 / rtf:.0f}x realtime)")

            return True, enhanced, sr
        else:
            print("⚠️ Model not available (this is expected if NovaSR not installed)")
            return False, None, None

    except Exception as e:
        print(f"⚠️ Model loading test skipped: {e}")
        return False, None, None


def test_tts_output():
    """Test TTSOutput with NovaSR attribute."""
    print("\n📦 Test 3: TTSOutput NovaSR Integration")
    print("-" * 40)

    try:
        from auralis.common.definitions.output import TTSOutput

        audio = np.random.randn(24000).astype(np.float32)
        output = TTSOutput(array=audio, sample_rate=24000)

        assert hasattr(output, "_novasr_applied"), "Missing _novasr_applied attribute"
        assert output._novasr_applied is False, (
            "_novasr_applied should default to False"
        )
        print("✅ TTSOutput has _novasr_applied attribute")

        assert hasattr(output, "apply_super_resolution"), (
            "Missing apply_super_resolution method"
        )
        print("✅ TTSOutput has apply_super_resolution method")

        return True, output

    except Exception as e:
        print(f"❌ TTSOutput test failed: {e}")
        traceback.print_exc()
        return False, None


def test_tts_request():
    """Test TTSRequest with NovaSR flag."""
    print("\n📦 Test 4: TTSRequest NovaSR Flag")
    print("-" * 40)

    try:
        from auralis.common.definitions.requests import TTSRequest

        # Test with NovaSR enabled
        request = TTSRequest(
            text="Test text for NovaSR", speaker_files=["test.wav"], apply_novasr=True
        )
        assert hasattr(request, "apply_novasr"), "Missing apply_novasr attribute"
        assert request.apply_novasr is True, "apply_novasr should be True"
        print("✅ TTSRequest has apply_novasr attribute")

        # Test default value
        request2 = TTSRequest(text="Test text", speaker_files=["test.wav"])
        assert request2.apply_novasr is False, "apply_novasr should default to False"
        print("✅ apply_novasr defaults to False")

        return True

    except Exception as e:
        print(f"❌ TTSRequest test failed: {e}")
        traceback.print_exc()
        return False


async def test_full_tts_pipeline():
    """Test full TTS pipeline with NovaSR."""
    print("\n📦 Test 5: Full TTS Pipeline with NovaSR")
    print("-" * 40)

    speaker_file = (
        Path(__file__).parent.parent
        / "tests"
        / "resources"
        / "audio_samples"
        / "female.wav"
    )
    if not speaker_file.exists():
        print(f"⚠️ Speaker file not found: {speaker_file}")
        return False, None, None

    try:
        from auralis import TTS, TTSRequest

        # Use RTX 3060 (device index 2)
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"

        print("Loading TTS model...")
        tts = TTS().from_pretrained(
            "AstraMindAI/xttsv2", gpt_model="AstraMindAI/xtts2-gpt"
        )
        print("✅ TTS model loaded")

        test_text = "This is a test of the NovaSR audio super-resolution system. It should produce clear, high-quality 48 kilohertz audio output."

        # Test without NovaSR
        print("\nGenerating audio WITHOUT NovaSR...")
        request_base = TTSRequest(
            text=test_text,
            speaker_files=[str(speaker_file)],
            language="en",
            apply_novasr=False,
        )
        start = time.perf_counter()
        output_base = await tts.generate_speech_async(request_base)
        base_time = time.perf_counter() - start

        print(f"✅ Base audio generated in {base_time:.2f}s")
        print(f"   Sample rate: {output_base.sample_rate} Hz")
        print(f"   Duration: {len(output_base.array) / output_base.sample_rate:.2f}s")
        print(f"   _novasr_applied: {output_base._novasr_applied}")

        # Save base audio
        output_dir = Path(__file__).parent.parent / "test_output"
        output_dir.mkdir(exist_ok=True)
        base_file = output_dir / "novasr_test_base.wav"
        output_base.save(str(base_file))
        print(f"   Saved to: {base_file}")

        # Test with NovaSR
        print("\nGenerating audio WITH NovaSR...")
        request_novasr = TTSRequest(
            text=test_text,
            speaker_files=[str(speaker_file)],
            language="en",
            apply_novasr=True,
        )
        start = time.perf_counter()
        output_novasr = await tts.generate_speech_async(request_novasr)
        novasr_time = time.perf_counter() - start

        print(f"✅ NovaSR audio generated in {novasr_time:.2f}s")
        print(f"   Sample rate: {output_novasr.sample_rate} Hz")
        print(
            f"   Duration: {len(output_novasr.array) / output_novasr.sample_rate:.2f}s"
        )
        print(f"   _novasr_applied: {output_novasr._novasr_applied}")

        # Save NovaSR audio
        novasr_file = output_dir / "novasr_test_enhanced.wav"
        output_novasr.save(str(novasr_file))
        print(f"   Saved to: {novasr_file}")

        # Calculate metrics
        base_duration = len(output_base.array) / output_base.sample_rate
        novasr_duration = len(output_novasr.array) / output_novasr.sample_rate

        base_rtf = base_time / base_duration
        novasr_rtf = novasr_time / novasr_duration

        print("\n📊 Performance Metrics:")
        print(
            f"   Base RTF: {base_rtf:.3f}x ({base_time:.2f}s for {base_duration:.2f}s audio)"
        )
        print(
            f"   NovaSR RTF: {novasr_rtf:.3f}x ({novasr_time:.2f}s for {novasr_duration:.2f}s audio)"
        )
        print(f"   NovaSR overhead: {(novasr_rtf - base_rtf) / base_rtf * 100:.1f}%")

        return True, base_file, novasr_file

    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        traceback.print_exc()
        return False, None, None


def test_whisper_transcription(base_file, novasr_file):
    """Use Whisper to transcribe both audio files and compare."""
    print("\n📦 Test 6: Whisper Transcription Verification")
    print("-" * 40)

    try:
        import whisper

        print("Loading Whisper model...")
        model = whisper.load_model("base")
        print("✅ Whisper model loaded")

        results = {}

        for name, file_path in [
            ("Base (24kHz)", base_file),
            ("NovaSR (48kHz)", novasr_file),
        ]:
            if file_path and Path(file_path).exists():
                print(f"\nTranscribing {name}...")
                start = time.perf_counter()
                result = model.transcribe(str(file_path))
                transcribe_time = time.perf_counter() - start

                text = result["text"].strip()
                print(f"✅ Transcribed in {transcribe_time:.2f}s")
                print(f"   Text: {text[:100]}...")

                results[name] = {
                    "text": text,
                    "time": transcribe_time,
                    "language": result.get("language", "unknown"),
                }
            else:
                print(f"⚠️ File not found: {file_path}")

        # Compare transcriptions
        if len(results) == 2:
            base_text = results["Base (24kHz)"]["text"]
            novasr_text = results["NovaSR (48kHz)"]["text"]

            # Simple similarity check
            base_words = set(base_text.lower().split())
            novasr_words = set(novasr_text.lower().split())
            common_words = base_words & novasr_words
            similarity = len(common_words) / max(len(base_words), len(novasr_words), 1)

            print(f"\n📊 Transcription Comparison:")
            print(f"   Word similarity: {similarity * 100:.1f}%")
            print(f"   Base transcription: {base_text}")
            print(f"   NovaSR transcription: {novasr_text}")

        return True, results

    except ImportError:
        print("⚠️ Whisper not installed. Skipping transcription test.")
        print("   Install with: pip install openai-whisper")
        return False, None
    except Exception as e:
        print(f"⚠️ Whisper test failed: {e}")
        traceback.print_exc()
        return False, None


def main():
    """Run all tests."""
    results = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "tests": {}}

    # Test 1: NovaSR Processor
    success, processor = test_novasr_processor()
    results["tests"]["novasr_processor"] = success

    # Test 2: Model Loading
    if processor:
        success, enhanced_audio, sr = test_novasr_model_loading(processor)
        results["tests"]["novasr_model_loading"] = success
    else:
        results["tests"]["novasr_model_loading"] = False

    # Test 3: TTSOutput
    success, output = test_tts_output()
    results["tests"]["tts_output"] = success

    # Test 4: TTSRequest
    success = test_tts_request()
    results["tests"]["tts_request"] = success

    # Test 5: Full Pipeline
    success, base_file, novasr_file = asyncio.run(test_full_tts_pipeline())
    results["tests"]["full_pipeline"] = success

    # Test 6: Whisper Verification
    if base_file and novasr_file:
        success, whisper_results = test_whisper_transcription(base_file, novasr_file)
        results["tests"]["whisper_transcription"] = success
        if whisper_results:
            results["whisper_results"] = whisper_results

    # Summary
    print("\n" + "=" * 70)
    print("📋 Test Summary")
    print("=" * 70)

    passed = sum(1 for v in results["tests"].values() if v)
    total = len(results["tests"])

    for test, success in results["tests"].items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test}: {status}")

    print(f"\n   Total: {passed}/{total} tests passed")
    print("=" * 70)

    # Save results
    output_dir = Path(__file__).parent.parent / "test_output"
    output_dir.mkdir(exist_ok=True)
    results_file = output_dir / "novasr_test_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 Results saved to: {results_file}")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
