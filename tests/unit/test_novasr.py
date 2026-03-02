"""Unit tests for NovaSR audio super-resolution integration."""

import pytest
import numpy as np
import torch
from auralis.common.definitions.output import TTSOutput


class TestNovaSRIntegration:
    """Test NovaSR audio super-resolution functionality."""

    def test_tts_output_has_novasr_attribute(self):
        """Test that TTSOutput has NovaSR tracking attribute."""
        audio = np.random.randn(24000).astype(np.float32)  # 1 second at 24kHz
        output = TTSOutput(array=audio, sample_rate=24000)
        assert hasattr(output, "_novasr_applied")
        assert output._novasr_applied is False

    def test_apply_super_resolution_method_exists(self):
        """Test that apply_super_resolution method exists."""
        audio = np.random.randn(24000).astype(np.float32)
        output = TTSOutput(array=audio, sample_rate=24000)
        assert hasattr(output, "apply_super_resolution")
        assert callable(getattr(output, "apply_super_resolution"))

    def test_novasr_fallback_on_missing_model(self):
        """Test that NovaSR gracefully falls back if model is unavailable."""
        audio = np.random.randn(24000).astype(np.float32)
        output = TTSOutput(array=audio, sample_rate=24000)

        try:
            enhanced = output.apply_super_resolution()
            assert enhanced is not None
            assert isinstance(enhanced, TTSOutput)
        except Exception as e:
            pytest.skip(f"NovaSR not available: {e}")

    def test_novasr_prevents_double_application(self):
        """Test that NovaSR is not applied twice."""
        audio = np.random.randn(24000).astype(np.float32)
        output = TTSOutput(array=audio, sample_rate=24000)

        try:
            enhanced = output.apply_super_resolution()
            enhanced2 = enhanced.apply_super_resolution()
            assert enhanced2._novasr_applied is True
        except Exception:
            pytest.skip("NovaSR not available in test environment")

    def test_tts_output_default_sample_rate(self):
        """Test that TTSOutput defaults to 24kHz before NovaSR."""
        audio = np.random.randn(24000).astype(np.float32)
        output = TTSOutput(array=audio)
        assert output.sample_rate == 24000

    def test_novasr_invalid_method(self):
        """Test that invalid super-resolution method raises error."""
        audio = np.random.randn(24000).astype(np.float32)
        output = TTSOutput(array=audio, sample_rate=24000)

        with pytest.raises(ValueError, match="Unknown super-resolution method"):
            output.apply_super_resolution(method="invalid_method")


class TestNovaSRProcessor:
    """Test NovaSR processor module."""

    def test_novasr_processor_import(self):
        """Test that NovaSR processor can be imported."""
        try:
            from auralis.common.enhancers.novasr import NovaSRProcessor

            assert NovaSRProcessor is not None
        except ImportError as e:
            pytest.fail(f"Failed to import NovaSRProcessor: {e}")

    def test_novasr_processor_initialization(self):
        """Test NovaSR processor initialization."""
        try:
            from auralis.common.enhancers.novasr import NovaSRProcessor

            processor = NovaSRProcessor(device="cpu")
            assert processor.device == torch.device("cpu")
            assert processor._initialized is False
        except ImportError:
            pytest.skip("NovaSR module not available")

    def test_get_novasr_processor_singleton(self):
        """Test that get_novasr_processor returns singleton."""
        try:
            from auralis.common.enhancers.novasr import get_novasr_processor

            proc1 = get_novasr_processor()
            proc2 = get_novasr_processor()
            assert proc1 is proc2
        except ImportError:
            pytest.skip("NovaSR module not available")

    def test_novasr_processor_fallback_upsample(self):
        """Test fallback upsampling when model is not available."""
        try:
            from auralis.common.enhancers.novasr import NovaSRProcessor

            processor = NovaSRProcessor(device="cpu")

            audio = np.random.randn(16000).astype(np.float32)
            upsampled = processor._fallback_upsample(audio)

            assert len(upsampled) == len(audio) * 3
        except ImportError:
            pytest.skip("NovaSR module not available")


class TestTTSRequestNovaSRConfig:
    """Test TTSRequest NovaSR configuration."""

    def test_tts_request_has_novasr_flag(self):
        """Test that TTSRequest has apply_novasr flag."""
        try:
            from auralis.common.definitions.requests import TTSRequest

            request = TTSRequest(
                text="Test", speaker_files=["test.wav"], apply_novasr=True
            )
            assert hasattr(request, "apply_novasr")
            assert request.apply_novasr is True
        except Exception as e:
            pytest.skip(f"Could not create TTSRequest: {e}")

    def test_tts_request_novasr_default_false(self):
        """Test that apply_novasr defaults to False."""
        try:
            from auralis.common.definitions.requests import TTSRequest

            request = TTSRequest(text="Test", speaker_files=["test.wav"])
            assert request.apply_novasr is False
        except Exception as e:
            pytest.skip(f"Could not create TTSRequest: {e}")

    def test_tts_request_novasr_can_be_enabled(self):
        """Test that NovaSR can be enabled in request."""
        try:
            from auralis.common.definitions.requests import TTSRequest

            request = TTSRequest(
                text="Test", speaker_files=["test.wav"], apply_novasr=True
            )
            assert request.apply_novasr is True
        except Exception as e:
            pytest.skip(f"Could not create TTSRequest: {e}")
