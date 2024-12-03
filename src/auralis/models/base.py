from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncGenerator, List, Union, Tuple, Optional

import torch
import torchaudio
from dataclasses import dataclass

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetCount
from vllm import RequestOutput

from auralis.common.definitions.output import TTSOutput
from auralis.common.definitions.requests import TTSRequest

Token = Union[int, List[int]]

AudioTokenGenerator = AsyncGenerator[RequestOutput, None]
AudioOutputGenerator = AsyncGenerator[TTSOutput, None]

SpeakerEmbeddings = torch.Tensor
GPTLikeDecoderConditioning = torch.Tensor
RequestsIds = List

TokenGeneratorsAndPossiblyConditioning = Union[
    Tuple[
        List[AudioTokenGenerator],
        RequestsIds,
        SpeakerEmbeddings,
        Union[List[GPTLikeDecoderConditioning], GPTLikeDecoderConditioning]
    ],
    Tuple[
        List[AudioTokenGenerator],
        RequestsIds,
        SpeakerEmbeddings
    ],
    Tuple[
        List[AudioTokenGenerator],
        RequestsIds,
        GPTLikeDecoderConditioning
    ],
    List[AudioTokenGenerator],
    RequestsIds
    ]

@dataclass
class ConditioningConfig:
    """Conditioning configuration for the model."""
    speaker_embeddings: bool = False
    gpt_like_decoder_conditioning: bool = False


class BaseAsyncTTSEngine(ABC, torch.nn.Module):
    """
    Base interface for TTS engines.
    It assumes a two-phase generation process:
    1. Token generation
    2. Audio generation
    """


    @abstractmethod
    async def get_generation_context(
            self,
            request: TTSRequest,
    ) -> TokenGeneratorsAndPossiblyConditioning:
        """
        Get token generator for audio generation.

        Args:
            request: TTS request object.

        Returns:
            A list of async generators of RequestOutput objects.
        """
        raise NotImplementedError

    @abstractmethod
    async def process_tokens_to_speech(
            self,
            generator: AudioTokenGenerator,
            speaker_embeddings: SpeakerEmbeddings,
            multimodal_data: GPTLikeDecoderConditioning = None,
            request: TTSRequest = None,
    ) -> AudioOutputGenerator:
        """
        Generate speech from token generators.

        Args:
            generator: A token generator (for now just vllm generators).
            speaker_embeddings: Speaker embeddings for voice cloning.
            multimodal_data: Multimodal data (used for vllm conditional generation).

        Returns:
            An async generator of TTSOutput objects.
        """
        raise NotImplementedError

    @property
    def conditioning_config(self) -> ConditioningConfig:
        """Get the conditioning configuration of the model."""
        raise NotImplementedError

    '''@abstractmethod
    async def get_speaker_embeddings(self, request: TTSRequest) -> torch.Tensor:
        """Get speaker embeddings from audio file"""
        pass

    @abstractmethod
    async def get_multimodal_conditioning(self, request: TTSRequest) -> torch.Tensor:
        """Get GPT-like conditioning from audio file"""
        pass'''

    @property
    def device(self):
        """Get the current device of the model."""
        return next(self.parameters()).device

    @property
    def dtype(self):
        """Get the current dtype of the model."""
        return next(self.parameters()).dtype

    @abstractmethod
    def get_memory_usage_curve(self):
        """Get memory usage curve by manually testing for vllm memory usage at different concurrency."""
        raise NotImplementedError

    @staticmethod
    def get_memory_percentage(memory: int) -> Optional[float]:
        """Get memory percentage."""

        for i in range(torch.cuda.device_count()):
            free_memory, total_memory = torch.cuda.mem_get_info(i)
            used_memory = total_memory - free_memory
            estimated_mem_occupation = (memory + used_memory) / total_memory
            if estimated_mem_occupation > 0 and estimated_mem_occupation < 1:
                return estimated_mem_occupation
        return None

    @classmethod
    def from_pretrained(
            cls,
            *args,
            **kwargs
    )-> 'BaseAsyncTTSEngine':
        """Load a pretrained model."""
        raise NotImplementedError

    @staticmethod
    def load_audio(audio_path: Union[str, Path], sampling_rate: int = 22050) -> torch.Tensor:
        audio, lsr = torchaudio.load(audio_path)

        # Stereo to mono if needed
        if audio.size(0) != 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        if lsr != sampling_rate:
            audio = torchaudio.functional.resample(audio, lsr, sampling_rate)

        # Clip audio invalid values
        audio.clip_(-1, 1)
        return audio