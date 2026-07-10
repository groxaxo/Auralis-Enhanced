from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, List, Optional, Tuple, Union

import torch
import torchaudio

try:
    from vllm import RequestOutput
except ImportError:  # vLLM is optional when using the MLX backend.
    RequestOutput = Any  # type: ignore[misc, assignment]

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
        Union[List[GPTLikeDecoderConditioning], GPTLikeDecoderConditioning],
    ],
    Tuple[List[AudioTokenGenerator], RequestsIds, SpeakerEmbeddings],
    Tuple[List[AudioTokenGenerator], RequestsIds, GPTLikeDecoderConditioning],
    Tuple[List[AudioTokenGenerator], RequestsIds],
]


@dataclass
class ConditioningConfig:
    """Conditioning capabilities exposed by a TTS engine."""

    speaker_embeddings: bool = False
    gpt_like_decoder_conditioning: bool = False


class BaseAsyncTTSEngine(ABC, torch.nn.Module):
    """Base interface for asynchronous, two-phase PyTorch TTS engines."""

    @abstractmethod
    async def get_generation_context(
        self, request: TTSRequest
    ) -> TokenGeneratorsAndPossiblyConditioning:
        raise NotImplementedError

    @abstractmethod
    async def process_tokens_to_speech(
        self,
        generator: AudioTokenGenerator,
        speaker_embeddings: SpeakerEmbeddings,
        multimodal_data: GPTLikeDecoderConditioning = None,
        request: TTSRequest = None,
    ) -> AudioOutputGenerator:
        raise NotImplementedError

    @property
    @abstractmethod
    def conditioning_config(self) -> ConditioningConfig:
        raise NotImplementedError

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @abstractmethod
    def get_memory_usage_curve(self):
        raise NotImplementedError

    @staticmethod
    def get_memory_percentage(memory: int) -> Optional[float]:
        for index in range(torch.cuda.device_count()):
            free_memory, total_memory = torch.cuda.mem_get_info(index)
            used_memory = total_memory - free_memory
            estimated_mem_occupation = (memory + used_memory) / total_memory
            if 0 < estimated_mem_occupation < 1:
                return estimated_mem_occupation
        return None

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "BaseAsyncTTSEngine":
        raise NotImplementedError

    @staticmethod
    def load_audio(
        audio_path: Union[str, Path], sampling_rate: int = 22050
    ) -> torch.Tensor:
        audio, loaded_sample_rate = torchaudio.load(audio_path)
        if audio.size(0) != 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if loaded_sample_rate != sampling_rate:
            audio = torchaudio.functional.resample(
                audio, loaded_sample_rate, sampling_rate
            )
        audio.clip_(-1, 1)
        return audio
