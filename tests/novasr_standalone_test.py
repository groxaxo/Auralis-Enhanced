#!/usr/bin/env python3
"""
Standalone NovaSR Test Script
==============================

Tests NovaSR integration without requiring the full Auralis import.
"""

import os
import sys
import time
import json
from pathlib import Path

import numpy as np
import torch

print("=" * 70)
print("🧪 NovaSR Standalone Test")
print("=" * 70)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  - {torch.cuda.get_device_name(i)}")
print("=" * 70)


def test_novasr_direct():
    """Test NovaSR model directly without auralis import."""
    print("\n📦 Test 1: Direct NovaSR Model Test")
    print("-" * 40)

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Use RTX 3060

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "NovaSR"))
        from NovaSR import FastSR

        print("Loading NovaSR model...")
        start = time.perf_counter()
        upsampler = FastSR(half=True)
        load_time = time.perf_counter() - start
        print(f"✅ Model loaded in {load_time:.2f}s")

        # Generate test audio (1 second of 16kHz sine wave)
        t = np.linspace(0, 1, 16000)
        audio_16k = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_16k).float().unsqueeze(0).unsqueeze(1)
        if torch.cuda.is_available():
            audio_tensor = audio_tensor.cuda().half()

        # Process
        print("Processing 1 second of audio...")
        start = time.perf_counter()
        with torch.no_grad():
            enhanced = upsampler.infer(audio_tensor)
        process_time = time.perf_counter() - start

        enhanced_np = enhanced.cpu().float().numpy().squeeze()

        print(f"✅ Processed in {process_time * 1000:.2f}ms")
        print(f"   Input: {len(audio_16k)} samples @ 16kHz")
        print(f"   Output: {len(enhanced_np)} samples @ 48kHz")

        # Calculate RTF
        rtf = process_time / 1.0  # 1 second of audio
        print(f"   RTF: {rtf:.4f}x ({1 / rtf:.0f}x realtime)")

        # Save audio
        import torchaudio

        output_dir = Path(__file__).parent.parent / "test_output"
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / "novasr_direct_test.wav"
        enhanced_tensor = torch.from_numpy(enhanced_np).unsqueeze(0)
        torchaudio.save(str(output_file), enhanced_tensor, 48000)
        print(f"   Saved to: {output_file}")

        return True, output_file

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False, None


def test_novasr_processor_module():
    """Test the novasr.py module directly."""
    print("\n📦 Test 2: NovaSR Processor Module")
    print("-" * 40)

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    try:
        # Import directly without triggering auralis.__init__
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "novasr", "/home/op/Auralis-Enhanced/src/auralis/common/enhancers/novasr.py"
        )
        novasr_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(novasr_module)

        NovaSRProcessor = novasr_module.NovaSRProcessor
        get_novasr_processor = novasr_module.get_novasr_processor

        # Test processor
        processor = NovaSRProcessor(device="cuda")
        print(f"✅ Processor initialized on {processor.device}")

        # Test fallback
        audio_16k = np.random.randn(16000).astype(np.float32) * 0.5
        upsampled = processor._fallback_upsample(audio_16k)
        assert len(upsampled) == 48000
        print("✅ Fallback upsampling works")

        # Test model loading
        print("Loading NovaSR model...")
        start = time.perf_counter()
        processor.load_model()
        load_time = time.perf_counter() - start
        print(f"✅ Model loaded in {load_time:.2f}s")

        if processor.is_available():
            # Test processing
            start = time.perf_counter()
            enhanced, sr = processor.process(audio_16k, sr=16000)
            process_time = time.perf_counter() - start

            print(f"✅ Processed in {process_time * 1000:.2f}ms")
            print(f"   Output: {len(enhanced)} samples @ {sr}Hz")

            rtf = process_time / 1.0
            print(f"   RTF: {rtf:.4f}x ({1 / rtf:.0f}x realtime)")

            return True, enhanced, sr
        else:
            print("⚠️ Model not available")
            return False, None, None

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False, None, None


def test_tts_output_attributes():
    """Test TTSOutput has NovaSR attributes."""
    print("\n📦 Test 3: TTSOutput NovaSR Attributes")
    print("-" * 40)

    try:
        import importlib.util

        # Load output module directly
        spec = importlib.util.spec_from_file_location(
            "output",
            "/home/op/Auralis-Enhanced/src/auralis/common/definitions/output.py",
        )
        output_module = importlib.util.module_from_spec(spec)

        # Mock dependencies
        sys.modules["auralis.common.enhancers.novasr"] = type(sys)("mock")
        sys.modules["auralis.common.enhancers.novasr"].get_novasr_processor = (
            lambda **kwargs: None
        )

        spec.loader.exec_module(output_module)
        TTSOutput = output_module.TTSOutput

        audio = np.random.randn(24000).astype(np.float32)
        output = TTSOutput(array=audio, sample_rate=24000)

        assert hasattr(output, "_novasr_applied"), "Missing _novasr_applied"
        assert output._novasr_applied is False, "Should default to False"
        print("✅ TTSOutput has _novasr_applied attribute (default: False)")

        assert hasattr(output, "apply_super_resolution"), (
            "Missing apply_super_resolution"
        )
        print("✅ TTSOutput has apply_super_resolution method")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_tts_request_attributes():
    """Test TTSRequest has NovaSR attributes."""
    print("\n📦 Test 4: TTSRequest NovaSR Attributes")
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
        from typing import Optional, List

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

        # Test with apply_novasr
        request = TTSRequest(text="Test", speaker_files=["test.wav"], apply_novasr=True)
        assert hasattr(request, "apply_novasr"), "Missing apply_novasr"
        assert request.apply_novasr is True
        print("✅ TTSRequest has apply_novasr attribute")

        # Test default
        request2 = TTSRequest(text="Test", speaker_files=["test.wav"])
        assert request2.apply_novasr is False
        print("✅ apply_novasr defaults to False")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_whisper_on_existing_samples():
    """Use Whisper to transcribe existing sample files."""
    print("\n📦 Test 5: Whisper Transcription on Existing Samples")
    print("-" * 40)

    try:
        import whisper

        print("Loading Whisper model...")
        model = whisper.load_model("base")
        print("✅ Whisper model loaded")

        samples_dir = Path("/home/op/Auralis-Enhanced/samples")

        # Find sample pairs
        results = {}
        for i in range(1, 6):
            base_file = samples_dir / f"sample_en_{i}_base.wav"
            enhanced_file = samples_dir / f"sample_en_{i}_enhanced.wav"

            if base_file.exists() and enhanced_file.exists():
                print(f"\nTranscribing sample pair {i}...")

                # Base
                start = time.perf_counter()
                result_base = model.transcribe(str(base_file))
                base_time = time.perf_counter() - start
                base_text = result_base["text"].strip()

                # Enhanced
                start = time.perf_counter()
                result_enhanced = model.transcribe(str(enhanced_file))
                enhanced_time = time.perf_counter() - start
                enhanced_text = result_enhanced["text"].strip()

                # Calculate similarity
                base_words = set(base_text.lower().split())
                enhanced_words = set(enhanced_text.lower().split())
                common = base_words & enhanced_words
                similarity = len(common) / max(len(base_words), len(enhanced_words), 1)

                print(f"   Base (24kHz): {base_text[:60]}...")
                print(f"   Enhanced (48kHz): {enhanced_text[:60]}...")
                print(f"   Similarity: {similarity * 100:.1f}%")

                results[f"sample_{i}"] = {
                    "base_text": base_text,
                    "enhanced_text": enhanced_text,
                    "similarity": similarity,
                    "base_time": base_time,
                    "enhanced_time": enhanced_time,
                }

        # Summary
        if results:
            avg_similarity = sum(r["similarity"] for r in results.values()) / len(
                results
            )
            print(f"\n📊 Average transcription similarity: {avg_similarity * 100:.1f}%")

            if avg_similarity > 0.8:
                print("✅ High transcription correlation - audio quality is preserved")
            else:
                print(
                    "⚠️ Lower transcription correlation - may indicate quality differences"
                )

        return True, results

    except ImportError:
        print("⚠️ Whisper not installed")
        return False, None
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False, None


def main():
    results = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "tests": {}}

    # Test 1: Direct NovaSR
    success, output_file = test_novasr_direct()
    results["tests"]["novasr_direct"] = success

    # Test 2: NovaSR Processor Module
    success, _, _ = test_novasr_processor_module()
    results["tests"]["novasr_processor_module"] = success

    # Test 3: TTSOutput Attributes
    success = test_tts_output_attributes()
    results["tests"]["tts_output_attributes"] = success

    # Test 4: TTSRequest Attributes
    success = test_tts_request_attributes()
    results["tests"]["tts_request_attributes"] = success

    # Test 5: Whisper on samples
    success, whisper_results = test_whisper_on_existing_samples()
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
    output_dir = Path("/home/op/Auralis-Enhanced/test_output")
    output_dir.mkdir(exist_ok=True)
    results_file = output_dir / "novasr_standalone_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📄 Results saved to: {results_file}")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
