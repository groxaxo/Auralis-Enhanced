import os
import asyncio
import numpy as np
import pytest
from auralis import TTS, TTSRequest
from auralis.common.definitions.output import peak_attenuate_only, to_float32_audio

# --- Unit Tests ---


def test_to_float32_audio():
    """Test valid conversion of various dtypes to float32 in [-1, 1]."""
    # int16
    x_int16 = np.array([32767, -32768, 0], dtype=np.int16)
    x_float = to_float32_audio(x_int16)
    assert x_float.dtype == np.float32
    assert np.isclose(x_float[0], 32767 / 32768.0)
    assert np.isclose(x_float[1], -1.0)

    # int32
    x_int32 = np.array([2147483647, -2147483648], dtype=np.int32)
    x_float_32 = to_float32_audio(x_int32)
    assert np.isclose(x_float_32[0], 1.0, atol=1e-5)

    # float32 passthrough
    x_f32 = np.array([0.5, -0.5], dtype=np.float32)
    x_out = to_float32_audio(x_f32)
    assert np.all(x_f32 == x_out)


def test_peak_attenuate_only():
    """Test that signal is attenuated only if peak exceeds target."""
    target = 0.8

    # Case 1: Hot signal (needs attenuation)
    x_hot = np.ones(100, dtype=np.float32) * 1.2
    x_out, peak, gain = peak_attenuate_only(x_hot, target_peak=target)

    assert np.isclose(peak, 1.2, atol=1e-5)
    assert gain < 1.0
    assert np.allclose(gain, target / 1.2, atol=1e-5)
    assert np.max(np.abs(x_out)) <= target + 1e-6

    # Case 2: Cold signal (no attenuation)
    x_cold = np.ones(100, dtype=np.float32) * 0.5
    x_out, peak, gain = peak_attenuate_only(x_cold, target_peak=target)

    assert peak == 0.5
    assert gain == 1.0
    assert np.allclose(x_out, x_cold)


# --- Integration Test ---


async def run_integration_test(tts):
    """Verify that end-to-end FlashSR generation respects the 0.95 cap."""

    text = "Testing robust saturation guards. This audio should be loud but completely free of digital clipping."

    # Find a speaker file
    speaker_file = "samples/benchmark_en.wav"
    if not os.path.exists(speaker_file):
        for f in os.listdir("samples"):
            if f.endswith(".wav"):
                speaker_file = os.path.join("samples", f)
                break

    request = TTSRequest(
        text=text,
        speaker_files=[speaker_file],
        language="en",
        apply_flashsr=True,  # Triggers the guards
    )

    print("Generating...")
    output = await tts.generate_speech_async(request)

    # Verify
    max_amp = np.max(np.abs(output.array))
    print(f"Integration Check - Max Amplitude: {max_amp}")

    assert max_amp <= 0.9501, f"Output exceeded 0.95 cap! Got {max_amp}"
    print("✅ Integration Test Passed")


if __name__ == "__main__":
    # Run Unit Tests
    print("Running Unit Tests...")
    test_to_float32_audio()
    test_peak_attenuate_only()
    print("✅ Unit Tests Passed")

    # Run Integration Test
    print("Running Integration Test...")
    print("Initializing TTS for integration test...")
    # Initialize TTS (ensure we simply load it here)
    tts = TTS().from_pretrained("AstraMindAI/xttsv2", gpt_model="AstraMindAI/xtts2-gpt")

    asyncio.run(run_integration_test(tts))
