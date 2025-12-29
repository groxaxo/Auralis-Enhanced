"""FlashSR Audio Super-Resolution Integration for Auralis Enhanced.

This module provides audio super-resolution capabilities using the FlashSR model,
which upscales audio from 16kHz to 48kHz with exceptional speed (200-400x real-time).

FlashSR is based on HierSpeech++ upsampler architecture and provides high-quality
audio enhancement with minimal computational overhead (~2MB model).

License: Apache-2.0 / CC-BY-4.0 (compatible with Auralis Enhanced)
"""

from typing import Optional
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


class FlashSRProcessor:
    """Audio super-resolution processor using FlashSR model.
    
    This processor upsamples audio from 16kHz to 48kHz using the FlashSR model,
    providing significant quality improvements for professional audio applications.
    
    Attributes:
        device (str): Processing device ('cuda' or 'cpu')
        model: Lazy-loaded FlashSR model instance
    """
    
    def __init__(self, device: Optional[str] = None):
        """Initialize FlashSR processor.
        
        Args:
            device (str, optional): Processing device. Defaults to 'cuda' if available.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model = None
        self._initialized = False
        
    def load_model(self):
        """Lazy load FlashSR model from Hugging Face Hub.
        
        Downloads and initializes the model on first use to avoid
        unnecessary loading overhead.
        """
        if self._initialized:
            return
            
        try:
            from FastAudioSR import FASR
            from huggingface_hub import hf_hub_download
            
            logger.info("Loading FlashSR model from Hugging Face Hub...")
            
            # Download model weights
            file_path = hf_hub_download(
                repo_id="YatharthS/FlashSR",
                filename="upsampler.pth",
                local_dir="."
            )
            
            # Initialize the upsampler
            self.model = FASR(file_path)
            
            # Move to appropriate device
            if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)
            
            self._initialized = True
            logger.info(f"FlashSR model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load FlashSR model: {e}")
            logger.warning("FlashSR super-resolution will be disabled. Falling back to standard resampling.")
            self._initialized = False
            raise
    
    def process(self, audio: np.ndarray, sr: int = 16000) -> tuple[np.ndarray, int]:
        """Upsample audio from 16kHz to 48kHz using FlashSR.
        
        Args:
            audio (np.ndarray): Input audio at 16kHz (1D array)
            sr (int): Sample rate (must be 16000)
            
        Returns:
            tuple[np.ndarray, int]: Upsampled audio at 48kHz and sample rate
            
        Raises:
            ValueError: If sample rate is not 16kHz
        """
        if sr != 16000:
            raise ValueError(
                f"FlashSR requires 16kHz input, got {sr}Hz. "
                "Please resample to 16kHz before processing."
            )
        
        # Ensure model is loaded
        self.load_model()
        
        if not self._initialized or self.model is None:
            # Fallback to simple interpolation if model failed to load
            logger.warning("FlashSR model not available, using simple interpolation")
            return self._fallback_upsample(audio), 48000
        
        try:
            # Convert to PyTorch tensor with batch dimension
            audio_tensor = torch.from_numpy(audio).float()
            
            # Add batch dimension if not present
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Move to device
            audio_tensor = audio_tensor.to(self.device)
            
            # Run FlashSR upsampling
            with torch.no_grad():
                enhanced = self.model.run(audio_tensor)
            
            # Convert back to numpy
            enhanced_np = enhanced.cpu().numpy().squeeze()
            
            logger.debug(f"FlashSR upsampled audio from {len(audio)} to {len(enhanced_np)} samples")
            
            return enhanced_np, 48000
            
        except Exception as e:
            logger.error(f"FlashSR processing failed: {e}")
            logger.warning("Falling back to simple interpolation")
            return self._fallback_upsample(audio), 48000
    
    def _fallback_upsample(self, audio: np.ndarray) -> np.ndarray:
        """Fallback upsampling using simple interpolation.
        
        Used when FlashSR model is not available or fails.
        
        Args:
            audio (np.ndarray): Input audio at 16kHz
            
        Returns:
            np.ndarray: Upsampled audio at 48kHz (simple interpolation)
        """
        # Simple linear interpolation for 3x upsampling (16kHz -> 48kHz)
        target_length = len(audio) * 3
        upsampled = np.interp(
            np.linspace(0, len(audio) - 1, target_length),
            np.arange(len(audio)),
            audio
        )
        return upsampled
    
    def is_available(self) -> bool:
        """Check if FlashSR model is available and initialized.
        
        Returns:
            bool: True if model is loaded and ready to use
        """
        return self._initialized and self.model is not None


# Global instance for reuse across calls
_global_processor: Optional[FlashSRProcessor] = None


def get_flashsr_processor(device: Optional[str] = None) -> FlashSRProcessor:
    """Get or create global FlashSR processor instance.
    
    This function maintains a singleton pattern to avoid loading the model
    multiple times.
    
    Args:
        device (str, optional): Processing device
        
    Returns:
        FlashSRProcessor: Global processor instance
    """
    global _global_processor
    
    if _global_processor is None:
        _global_processor = FlashSRProcessor(device=device)
    
    return _global_processor
