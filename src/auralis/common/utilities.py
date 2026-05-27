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


# ---------------------------------------------------------------------------
# Auto-concurrency for the vLLM-backed XTTSv2 engine
# ---------------------------------------------------------------------------

# Default ``max_concurrency`` ceiling. The empirical throughput curve
# on the XTTSv2 GPT plateaus past 32 in-flight sequences -- on a 16 GiB
# RTX 5080 Laptop we measure ~23x realtime at conc=16 vs ~27x at
# conc=32 vs ~29x at conc=64, so each doubling past 32 buys <10% of
# throughput at 2x the activation + KV pressure. Operators with very
# large GPUs and specific throughput targets can override by passing
# ``max_concurrency`` explicitly.
DEFAULT_MAX_CONCURRENCY_CAP = 32

# Per-concurrent-slot memory budget inside the vLLM allocation:
#   activations: ~30 MiB/slot (measured: 0.49 GiB @ conc=16 ->
#                              0.97 GiB @ conc=32 -> 1.93 GiB @ conc=64)
#   paged KV at the XTTS GPT config (max_model_len=1047, 30 layers,
#                              h=1024, bf16): ~125 MiB peak/slot
#   --------------------------------------------------------------
#   total per slot:                          ~0.155 GiB
PER_SLOT_GB = 0.155

# Inside the vLLM budget, fixed costs not scaling with concurrency:
#   GPT weights (bf16):           ~0.75 GiB
#   vLLM scheduler bookkeeping:   ~0.10 GiB
FIXED_INSIDE_VLLM_GB = 0.85

# What HiFiGAN + conditioning encoder + per-step decoder tensors hold
# OUTSIDE vLLM's accounted budget. Conservative to keep room for the
# super-resolution head (NovaSR) and for transient peak allocations
# during conditioning.
OUTSIDE_VLLM_OVERHEAD_GB = 3.0


def suggest_max_concurrency(
    gpu_memory_utilization: float = 0.5,
    *,
    cap: int = DEFAULT_MAX_CONCURRENCY_CAP,
    device: "torch.device | int | None" = None,
) -> int:
    """Pick a sensible ``max_concurrency`` default from currently-free
    VRAM and the configured ``gpu_memory_utilization``.

    Used by both :class:`auralis.core.tts.TTS` (to size the two-phase
    scheduler) and :class:`auralis.models.xttsv2.XTTSv2Engine` (to
    forward to vLLM's ``max_num_seqs``) so the two stay in lockstep
    without the caller having to set them manually.

    The vLLM budget is the smaller of (a) the engine's self-cap
    ``gpu_memory_utilization * total_gpu_memory``, and (b) what is
    currently free on the device minus an out-of-vLLM allowance for
    the HiFiGAN decoder + conditioning encoder. From that budget we
    subtract a fixed term (weights + scheduler) and divide the
    remainder by the per-slot cost (activations + paged KV).

    The result is clamped to ``[1, cap]``. ``cap`` defaults to
    :data:`DEFAULT_MAX_CONCURRENCY_CAP` (32) -- see that constant's
    docstring for the throughput curve that motivates it. Pass
    ``cap=64`` (or higher) on big-memory GPUs where you want the
    extra 5-10% throughput at the cost of significantly more VRAM.

    Returns 1 on CPU-only systems and on GPUs too cramped to fit even
    a single sequence.
    """
    if not torch.cuda.is_available():
        return 1
    try:
        dev = (torch.cuda.current_device() if device is None
               else torch.device(device).index if isinstance(device, torch.device)
               else int(device))
        free_bytes, total_bytes = torch.cuda.mem_get_info(dev)
    except Exception:
        # mem_get_info can fail on some driver / CUDA combinations;
        # return a middle-ground default rather than guessing further.
        return min(cap, 8)
    free_gb = free_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)

    vllm_budget_gb = min(
        total_gb * float(gpu_memory_utilization),
        free_gb - OUTSIDE_VLLM_OVERHEAD_GB,
    )
    if vllm_budget_gb <= FIXED_INSIDE_VLLM_GB:
        return 1
    slots = int((vllm_budget_gb - FIXED_INSIDE_VLLM_GB) / PER_SLOT_GB)
    return max(1, min(cap, slots))
