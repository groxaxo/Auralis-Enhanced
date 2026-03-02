"""NovaSR Audio Super-Resolution Integration for Auralis Enhanced.

This module provides audio super-resolution capabilities using the NovaSR model,
which upscales audio from 16kHz to 48kHz with exceptional speed (3600x real-time).

NovaSR is a tiny 52KB model based on BigVGAN-style snake activations, providing
high-quality audio enhancement with minimal computational overhead.

License: Apache-2.0 / CC-BY-4.0 (compatible with Auralis Enhanced)
"""

from typing import Optional, Tuple
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


class NovaSRProcessor:
    """Audio super-resolution processor using NovaSR model.

    This processor upsamples audio from 16kHz to 48kHz using the NovaSR model,
    providing significant quality improvements for professional audio applications.

    Attributes:
        device (str): Processing device ('cuda' or 'cpu')
        model: Lazy-loaded NovaSR model instance
    """

    def __init__(self, device: Optional[str] = None):
        """Initialize NovaSR processor.

        Args:
            device (str, optional): Processing device. Defaults to 'cuda' if available.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = None
        self._initialized = False
        self._half = True

    def load_model(self):
        """Lazy load NovaSR model from Hugging Face Hub.

        Downloads and initializes the model on first use to avoid
        unnecessary loading overhead.
        """
        if self._initialized:
            return

        try:
            from NovaSR import FastSR

            logger.info("Loading NovaSR model from Hugging Face Hub...")

            self.model = FastSR(half=self._half)

            self._initialized = True
            logger.info(f"NovaSR model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load NovaSR model: {e}")
            logger.warning(
                "NovaSR super-resolution will be disabled. Falling back to standard resampling."
            )
            self._initialized = False
            raise

    def process(self, audio: np.ndarray, sr: int = 16000) -> Tuple[np.ndarray, int]:
        """Upsample audio from 16kHz to 48kHz using NovaSR.

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
                f"NovaSR requires 16kHz input, got {sr}Hz. "
                "Please resample to 16kHz before processing."
            )

        self.load_model()

        if not self._initialized or self.model is None:
            logger.warning("NovaSR model not available, using simple interpolation")
            return self._fallback_upsample(audio), 48000

        try:
            audio_tensor = torch.from_numpy(audio).float()

            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            audio_tensor = audio_tensor.to(self.device)

            if self._half and self.device.type == "cuda":
                audio_tensor = audio_tensor.half()

            audio_tensor = audio_tensor.unsqueeze(1)

            with torch.no_grad():
                enhanced = self.model.infer(audio_tensor)

            enhanced_np = enhanced.cpu().float().numpy().squeeze()

            if enhanced_np.ndim > 1:
                enhanced_np = enhanced_np.flatten()

            logger.debug(
                f"NovaSR upsampled audio from {len(audio)} to {len(enhanced_np)} samples"
            )

            return enhanced_np, 48000

        except Exception as e:
            logger.error(f"NovaSR processing failed: {e}")
            logger.warning("Falling back to simple interpolation")
            return self._fallback_upsample(audio), 48000

    def _fallback_upsample(self, audio: np.ndarray) -> np.ndarray:
        """Fallback upsampling using simple interpolation.

        Used when NovaSR model is not available or fails.

        Args:
            audio (np.ndarray): Input audio at 16kHz

        Returns:
            np.ndarray: Upsampled audio at 48kHz (simple interpolation)
        """
        target_length = len(audio) * 3
        upsampled = np.interp(
            np.linspace(0, len(audio) - 1, target_length), np.arange(len(audio)), audio
        )
        return upsampled

    def is_available(self) -> bool:
        """Check if NovaSR model is available and initialized.

        Returns:
            bool: True if model is loaded and ready to use
        """
        return self._initialized and self.model is not None


_global_processor: Optional[NovaSRProcessor] = None


def get_novasr_processor(device: Optional[str] = None) -> NovaSRProcessor:
    """Get or create global NovaSR processor instance.

    This function maintains a singleton pattern to avoid loading the model
    multiple times.

    Args:
        device (str, optional): Processing device

    Returns:
        NovaSRProcessor: Global processor instance
    """
    global _global_processor

    if _global_processor is None:
        _global_processor = NovaSRProcessor(device=device)

    return _global_processor
