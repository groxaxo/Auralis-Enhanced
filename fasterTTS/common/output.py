import io
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, Tuple, List
import sounddevice as sd

from IPython.display import Audio, display


import numpy as np
import torch
import torchaudio


@dataclass
class TTSOutput:
    """Container for XTTS inference output with integrated audio utilities"""
    wav: np.ndarray
    sample_rate: int = 24000

    @staticmethod
    def combilne_outputs(outputs: List['TTSOutput']) -> 'TTSOutput':
        """Combine multiple TTSOutput instances into a single instance.

        Args:
            outputs: List of TTSOutput instances

        Returns:
            New TTSOutput instance with concatenated audio
        """
        # Concatenate audio
        combined_audio = np.concatenate([out.wav for out in outputs])

        # Use sample rate of first output
        return TTSOutput(
            wav=combined_audio,
            sample_rate=outputs[0].sample_rate
        )

    def to_tensor(self) -> Union[torch.Tensor, np.ndarray]:
        """Convert numpy array to torch tensor"""
        if isinstance(self.wav, np.ndarray):
            return torch.from_numpy(self.wav)
        return self.wav

    def to_bytes(self, format: str = 'wav', sample_width: int = 2) -> bytes:
        """Convert audio to bytes format.

        Args:
            format: Output format ('wav' or 'raw')
            sample_width: Bit depth (1, 2, or 4 bytes per sample)

        Returns:
            Audio data as bytes
        """
        # Convert to tensor if needed
        wav_tensor = self.to_tensor()

        # Ensure correct shape (1, N) for torchaudio
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)

        # Normalize to [-1, 1]
        wav_tensor = torch.clamp(wav_tensor, -1.0, 1.0)

        if format == 'wav':
            buffer = io.BytesIO()
            torchaudio.save(
                buffer,
                wav_tensor,
                self.sample_rate,
                format="wav",
                encoding="PCM_S" if sample_width == 2 else "PCM_F",
                bits_per_sample=sample_width * 8
            )
            return buffer.getvalue()

        elif format == 'raw':
            # Scale to appropriate range based on sample width
            if sample_width == 2:  # 16-bit
                wav_tensor = (wav_tensor * 32767).to(torch.int16)
            elif sample_width == 4:  # 32-bit
                wav_tensor = (wav_tensor * 2147483647).to(torch.int32)
            else:  # 8-bit
                wav_tensor = (wav_tensor * 127).to(torch.int8)
            return wav_tensor.cpu().numpy().tobytes()

        else:
            raise ValueError(f"Unsupported format: {format}")

    def save(self,
             filename: Union[str, Path],
             sample_rate: Optional[int] = None,
             format: Optional[str] = None) -> None:
        """Save audio to file.

        Args:
            filename: Output filename
            sample_rate: Optional new sample rate for resampling
            format: Optional format override (default: inferred from extension)
        """
        wav_tensor = self.to_tensor()
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)

        # Resample if needed
        if sample_rate and sample_rate != self.sample_rate:
            wav_tensor = torchaudio.functional.resample(
                wav_tensor,
                orig_freq=self.sample_rate,
                new_freq=sample_rate
            )
        else:
            sample_rate = self.sample_rate

        torchaudio.save(
            filename,
            wav_tensor,
            sample_rate,
            format=format
        )

    def resample(self, new_sample_rate: int) -> 'TTSOutput':
        """Create new TTSOutput with resampled audio.

        Args:
            new_sample_rate: Target sample rate

        Returns:
            New TTSOutput instance with resampled audio
        """
        wav_tensor = self.to_tensor()
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)

        resampled = torchaudio.functional.resample(
            wav_tensor,
            orig_freq=self.sample_rate,
            new_freq=new_sample_rate
        )

        return TTSOutput(
            wav=resampled.squeeze().numpy(),
            sample_rate=new_sample_rate
        )

    def get_info(self) -> Tuple[int, int, float]:
        """Get audio information.

        Returns:
            Tuple of (number of samples, sample rate, duration in seconds)
        """
        n_samples = len(self.wav)
        duration = n_samples / self.sample_rate
        return n_samples, self.sample_rate, duration

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, sample_rate: int = 24000) -> 'TTSOutput':
        """Create TTSOutput from torch tensor.

        Args:
            tensor: Audio tensor
            sample_rate: Sample rate of the audio

        Returns:
            New TTSOutput instance
        """
        return cls(
            wav=tensor.squeeze().cpu().numpy(),
            sample_rate=sample_rate
        )

    @classmethod
    def from_file(cls, filename: Union[str, Path]) -> 'TTSOutput':
        """Create TTSOutput from audio file.

        Args:
            filename: Path to audio file

        Returns:
            New TTSOutput instance
        """
        wav_tensor, sample_rate = torchaudio.load(filename)
        return cls.from_tensor(wav_tensor, sample_rate)

    def play(self) -> None:
        """Play the audio through the default sound device.
        For use in regular Python scripts/applications."""
        # Ensure the audio is in the correct format
        if isinstance(self.wav, torch.Tensor):
            audio_data = self.wav.cpu().numpy()
        else:
            audio_data = self.wav

        # Ensure float32 and normalize
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        audio_data = np.clip(audio_data, -1.0, 1.0)

        # Play the audio
        sd.play(audio_data, self.sample_rate)
        sd.wait()  # Wait until the audio is finished playing

    def display(self) -> Optional[Audio]:
        """Display audio player in Jupyter notebook.
        Returns Audio widget if in notebook, None otherwise."""
        try:
            # Convert to bytes
            audio_bytes = self.to_bytes(format='wav')

            # Create and display audio widget
            audio_widget = Audio(audio_bytes, rate=self.sample_rate, autoplay=False)
            display(audio_widget)
            return audio_widget
        except Exception as e:
            print(f"Could not display audio widget: {str(e)}")
            print("Try using .play() method instead")
            return None

    def preview(self) -> None:
        """Smart play method that chooses appropriate playback method."""
        try:
            # Try notebook display first
            if self.display() is None:
                # Fall back to sounddevice if not in notebook
                self.play()
        except Exception as e:
            print(f"Error playing audio: {str(e)}")