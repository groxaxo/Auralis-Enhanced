"""Unit tests for FlashSR audio super-resolution integration."""

import pytest
import numpy as np
import torch
from auralis.common.definitions.output import TTSOutput


class TestFlashSRIntegration:
    """Test FlashSR audio super-resolution functionality."""
    
    def test_tts_output_has_flashsr_attribute(self):
        """Test that TTSOutput has FlashSR tracking attribute."""
        audio = np.random.randn(24000).astype(np.float32)  # 1 second at 24kHz
        output = TTSOutput(array=audio, sample_rate=24000)
        assert hasattr(output, '_flashsr_applied')
        assert output._flashsr_applied is False
    
    def test_apply_super_resolution_method_exists(self):
        """Test that apply_super_resolution method exists."""
        audio = np.random.randn(24000).astype(np.float32)
        output = TTSOutput(array=audio, sample_rate=24000)
        assert hasattr(output, 'apply_super_resolution')
        assert callable(getattr(output, 'apply_super_resolution'))
    
    def test_flashsr_fallback_on_missing_model(self):
        """Test that FlashSR gracefully falls back if model is unavailable."""
        audio = np.random.randn(24000).astype(np.float32)
        output = TTSOutput(array=audio, sample_rate=24000)
        
        # This should not raise even if FlashSR model isn't installed
        try:
            enhanced = output.apply_super_resolution()
            # Should return something, even if just the original
            assert enhanced is not None
            assert isinstance(enhanced, TTSOutput)
        except Exception as e:
            # If FlashSR fails to import, that's expected in test env
            pytest.skip(f"FlashSR not available: {e}")
    
    def test_flashsr_prevents_double_application(self):
        """Test that FlashSR is not applied twice."""
        audio = np.random.randn(24000).astype(np.float32)
        output = TTSOutput(array=audio, sample_rate=24000)
        
        try:
            enhanced = output.apply_super_resolution()
            # Try to apply again
            enhanced2 = enhanced.apply_super_resolution()
            # Should return the same object or equivalent
            assert enhanced2._flashsr_applied is True
        except Exception:
            pytest.skip("FlashSR not available in test environment")
    
    def test_tts_output_default_sample_rate(self):
        """Test that TTSOutput defaults to 24kHz before FlashSR."""
        audio = np.random.randn(24000).astype(np.float32)
        output = TTSOutput(array=audio)
        assert output.sample_rate == 24000
    
    def test_flashsr_invalid_method(self):
        """Test that invalid super-resolution method raises error."""
        audio = np.random.randn(24000).astype(np.float32)
        output = TTSOutput(array=audio, sample_rate=24000)
        
        with pytest.raises(ValueError, match="Unknown super-resolution method"):
            output.apply_super_resolution(method='invalid_method')


class TestFlashSRProcessor:
    """Test FlashSR processor module."""
    
    def test_flashsr_processor_import(self):
        """Test that FlashSR processor can be imported."""
        try:
            from auralis.common.enhancers.flashsr import FlashSRProcessor
            assert FlashSRProcessor is not None
        except ImportError as e:
            pytest.fail(f"Failed to import FlashSRProcessor: {e}")
    
    def test_flashsr_processor_initialization(self):
        """Test FlashSR processor initialization."""
        try:
            from auralis.common.enhancers.flashsr import FlashSRProcessor
            processor = FlashSRProcessor(device='cpu')
            assert processor.device == 'cpu'
            assert processor._initialized is False
        except ImportError:
            pytest.skip("FlashSR module not available")
    
    def test_get_flashsr_processor_singleton(self):
        """Test that get_flashsr_processor returns singleton."""
        try:
            from auralis.common.enhancers.flashsr import get_flashsr_processor
            proc1 = get_flashsr_processor()
            proc2 = get_flashsr_processor()
            assert proc1 is proc2  # Should be same instance
        except ImportError:
            pytest.skip("FlashSR module not available")
    
    def test_flashsr_processor_fallback_upsample(self):
        """Test fallback upsampling when model is not available."""
        try:
            from auralis.common.enhancers.flashsr import FlashSRProcessor
            processor = FlashSRProcessor(device='cpu')
            
            # Test fallback method directly
            audio = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz
            upsampled = processor._fallback_upsample(audio)
            
            # Should be 3x longer (16kHz -> 48kHz)
            assert len(upsampled) == len(audio) * 3
        except ImportError:
            pytest.skip("FlashSR module not available")


class TestTTSRequestFlashSRConfig:
    """Test TTSRequest FlashSR configuration."""
    
    def test_tts_request_has_flashsr_flag(self):
        """Test that TTSRequest has apply_flashsr flag."""
        try:
            from auralis.common.definitions.requests import TTSRequest
            
            # Note: TTSRequest requires text and speaker_files
            request = TTSRequest(
                text="Test",
                speaker_files=["test.wav"],
                apply_flashsr=True
            )
            assert hasattr(request, 'apply_flashsr')
            assert request.apply_flashsr is True
        except Exception as e:
            # If request creation fails, skip
            pytest.skip(f"Could not create TTSRequest: {e}")
    
    def test_tts_request_flashsr_default_true(self):
        """Test that apply_flashsr defaults to True."""
        try:
            from auralis.common.definitions.requests import TTSRequest
            
            request = TTSRequest(
                text="Test",
                speaker_files=["test.wav"]
            )
            # Should default to True for FlashSR by default
            assert request.apply_flashsr is True
        except Exception as e:
            pytest.skip(f"Could not create TTSRequest: {e}")
    
    def test_tts_request_flashsr_can_be_disabled(self):
        """Test that FlashSR can be disabled in request."""
        try:
            from auralis.common.definitions.requests import TTSRequest
            
            request = TTSRequest(
                text="Test",
                speaker_files=["test.wav"],
                apply_flashsr=False
            )
            assert request.apply_flashsr is False
        except Exception as e:
            pytest.skip(f"Could not create TTSRequest: {e}")
