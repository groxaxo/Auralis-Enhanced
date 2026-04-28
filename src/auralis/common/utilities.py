from typing import Union, Callable, Dict, Any, Tuple

import fsspec
import functools
import torch
import torchaudio
import io


# ---------------------------------------------------------------------------
# Module-level LRU cache so we never rebuild the same MelSpectrogram transform.
# The cache key is all transform hyper-parameters plus the target device string.
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=16)
def _cached_mel_transform(
    n_fft: int,
    hop_length: int,
    win_length: int,
    power: int,
    normalized: bool,
    sample_rate: int,
    f_min: int,
    f_max: int,
    n_mels: int,
    device_str: str,
) -> torchaudio.transforms.MelSpectrogram:
    """Build and return a cached MelSpectrogram transform for the given parameters."""
    return torchaudio.transforms.MelSpectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=power,
        normalized=normalized,
        sample_rate=sample_rate,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        norm="slaney",
    ).to(torch.device(device_str))


def wav_to_mel_cloning(
        wav,
        mel_norms_file="../experiments/clips_mel_norms.pth",
        mel_norms=None,
        device=torch.device("cpu"),
        n_fft=4096,
        hop_length=1024,
        win_length=4096,
        power=2,
        normalized=False,
        sample_rate=22050,
        f_min=0,
        f_max=8000,
        n_mels=80,
):
    """Convert waveform to normalized mel-spectrogram for voice cloning.

    This function converts a raw audio waveform to a mel-spectrogram using the
    specified parameters, then normalizes it using pre-computed mel norms for
    consistent voice cloning results.

    Args:
        wav (torch.Tensor): Input waveform tensor.
        mel_norms_file (str, optional): Path to mel norms file. Defaults to
            "../experiments/clips_mel_norms.pth".
        mel_norms (torch.Tensor, optional): Pre-loaded mel norms. Defaults to None.
        device (torch.device, optional): Device to perform computation on.
            Defaults to CPU.
        n_fft (int, optional): FFT size. Defaults to 4096.
        hop_length (int, optional): Number of samples between STFT columns.
            Defaults to 1024.
        win_length (int, optional): Window size. Defaults to 4096.
        power (int, optional): Exponent for the magnitude spectrogram.
            Defaults to 2.
        normalized (bool, optional): Whether to normalize by magnitude after STFT.
            Defaults to False.
        sample_rate (int, optional): Audio sample rate. Defaults to 22050.
        f_min (int, optional): Minimum frequency. Defaults to 0.
        f_max (int, optional): Maximum frequency. Defaults to 8000.
        n_mels (int, optional): Number of mel filterbanks. Defaults to 80.

    Returns:
        torch.Tensor: Normalized mel-spectrogram.
    """
    mel_stft = _cached_mel_transform(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=power,
        normalized=normalized,
        sample_rate=sample_rate,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        device_str=str(device),
    )
    wav = wav.to(device)
    mel = mel_stft(wav)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    if mel_norms is None:
        mel_norms = torch.load(mel_norms_file, map_location=device)
    mel = mel / mel_norms.unsqueeze(0).unsqueeze(-1)
    return mel


def load_audio(audiopath, sampling_rate):
    """Load and preprocess audio file.

    This function loads an audio file, converts it to mono if needed,
    resamples to the target sampling rate, and ensures valid amplitude range.

    Args:
        audiopath (Union[str, Path]): Path to audio file.
        sampling_rate (int): Target sampling rate.

    Returns:
        torch.Tensor: Preprocessed audio tensor of shape [1, samples].
    """
    audio, lsr = torchaudio.load(audiopath)

    # Stereo to mono if needed
    if audio.size(0) != 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)

    # Clip audio invalid values
    audio.clip_(-1, 1)
    return audio

def load_fsspec(
    path: str,
    map_location: Union[str, Callable, torch.device, Dict[Union[str, torch.device], Union[str, torch.device]]] = None,
    **kwargs,
) -> Any:
    """Load PyTorch checkpoint from any fsspec-supported location.

    This function extends torch.load to support loading from various file systems
    and cloud storage providers (e.g., S3, GCS) using fsspec.

    Args:
        path (str): Any path or URL supported by fsspec (e.g., 's3://', 'gs://').
        map_location (Union[str, Callable, torch.device, Dict], optional): Device
            mapping specification for torch.load. Defaults to None.
        **kwargs: Additional arguments passed to torch.load.

    Returns:
        Any: Object stored in the checkpoint.

    Example:
        >>> state_dict = load_fsspec('s3://my-bucket/model.pth', map_location='cuda')
    """
    with fsspec.open(path, "rb") as f:
            return torch.load(f, map_location=map_location, **kwargs)
