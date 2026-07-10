"""MLX-backed TTS adapter for Apple Silicon.

The implementation intentionally imports MLX and mlx-audio only when this
backend is selected. CUDA/vLLM users therefore do not inherit Apple-specific
runtime dependencies, and Mac users do not need vLLM installed.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator, Generator, Iterator, Optional

import numpy as np

from auralis.backends.selection import is_apple_silicon
from auralis.common.definitions.output import TTSOutput
from auralis.common.definitions.requests import TTSRequest

logger = logging.getLogger(__name__)

_VLLM_ONLY_ARGUMENTS = {
    "gpt_model",
    "device_map",
    "max_concurrency",
    "gpu_memory_utilization",
    "cpu_offload_gb",
    "swap_space",
    "tensor_parallel_size",
}

_LANGUAGE_ALIASES = {
    "zh-cn": "zh",
}


class MLXBackendUnavailableError(RuntimeError):
    """Raised when the optional MLX backend cannot be loaded."""


def _next_or_sentinel(iterator: Iterator[TTSOutput], sentinel: object) -> object:
    try:
        return next(iterator)
    except StopIteration:
        return sentinel


class MLXTTSEngine:
    """Adapter exposing mlx-audio models through Auralis' public API."""

    backend_name = "mlx"

    def __init__(
        self,
        model: Any,
        model_name_or_path: str,
        *,
        voice: Optional[str] = None,
        ref_text: Optional[str | list[str]] = None,
        instruct: Optional[str] = None,
        max_tokens: Optional[int] = 1200,
        streaming_interval: float = 2.0,
        generation_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.model_name_or_path = model_name_or_path
        self.default_voice = voice
        self.default_ref_text = ref_text
        self.default_instruct = instruct
        self.default_max_tokens = max_tokens
        self.default_streaming_interval = streaming_interval
        self.generation_kwargs = dict(generation_kwargs or {})
        self.sample_rate = int(getattr(model, "sample_rate", 24000))

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> "MLXTTSEngine":
        """Load an MLX-native TTS model from disk or Hugging Face Hub."""

        allow_unsupported = os.getenv("AURALIS_ALLOW_MLX_NON_APPLE") == "1"
        if not is_apple_silicon() and not allow_unsupported:
            raise MLXBackendUnavailableError(
                "The MLX backend requires macOS on Apple Silicon. "
                "Use backend='vllm' on NVIDIA/Linux systems."
            )

        try:
            from mlx_audio.tts.utils import load_model
        except ImportError as exc:
            raise MLXBackendUnavailableError(
                "MLX support is not installed. From the repository root run "
                "`pip install -e \".[mlx]\"`."
            ) from exc

        ignored = sorted(key for key in kwargs if key in _VLLM_ONLY_ARGUMENTS)
        for key in ignored:
            kwargs.pop(key, None)
        if ignored:
            logger.debug("Ignoring vLLM-only MLX arguments: %s", ", ".join(ignored))

        voice = kwargs.pop("voice", None)
        ref_text = kwargs.pop("ref_text", None)
        instruct = kwargs.pop("instruct", None)
        max_tokens = kwargs.pop("max_tokens", 1200)
        streaming_interval = kwargs.pop("streaming_interval", 2.0)
        generation_kwargs = kwargs.pop("generation_kwargs", None)
        lazy = bool(kwargs.pop("lazy", False))
        strict = bool(kwargs.pop("strict", True))

        model = load_model(
            model_name_or_path,
            lazy=lazy,
            strict=strict,
            **kwargs,
        )
        return cls(
            model=model,
            model_name_or_path=model_name_or_path,
            voice=voice,
            ref_text=ref_text,
            instruct=instruct,
            max_tokens=max_tokens,
            streaming_interval=streaming_interval,
            generation_kwargs=generation_kwargs,
        )

    @property
    def conditioning_config(self) -> Any:
        """Compatibility shim for callers that inspect the loaded engine."""

        return type(
            "MLXConditioningConfig",
            (),
            {
                "speaker_embeddings": False,
                "gpt_like_decoder_conditioning": False,
            },
        )()

    def _temporary_reference_files(
        self, speaker_files: Any
    ) -> tuple[list[Any], list[str]]:
        if speaker_files is None:
            return [], []
        values = speaker_files if isinstance(speaker_files, list) else [speaker_files]
        resolved: list[Any] = []
        temporary_paths: list[str] = []

        for value in values:
            if isinstance(value, (bytes, bytearray, memoryview)):
                handle = tempfile.NamedTemporaryFile(
                    prefix="auralis-mlx-ref-", suffix=".wav", delete=False
                )
                with handle:
                    handle.write(bytes(value))
                temporary_paths.append(handle.name)
                resolved.append(handle.name)
            elif isinstance(value, Path):
                resolved.append(str(value))
            else:
                resolved.append(value)

        return resolved, temporary_paths

    @staticmethod
    def _collapse(values: list[Any]) -> Any:
        if not values:
            return None
        return values[0] if len(values) == 1 else values

    def _filter_generation_kwargs(self, values: dict[str, Any]) -> dict[str, Any]:
        """Pass only arguments supported by the selected mlx-audio model."""

        try:
            signature = inspect.signature(self.model.generate)
        except (TypeError, ValueError):
            return values

        parameters = signature.parameters
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts_kwargs:
            return values
        return {key: value for key, value in values.items() if key in parameters}

    def _generation_arguments(
        self, request: TTSRequest, text: str, references: list[Any]
    ) -> dict[str, Any]:
        language = _LANGUAGE_ALIASES.get(request.language, request.language)
        # XTTS historically defaults to 5.0, while MLX autoregressive TTS
        # models generally use a much gentler repetition penalty. Preserve an
        # explicitly changed value, but translate the legacy default.
        repetition_penalty = (
            1.05 if request.repetition_penalty == 5.0 else request.repetition_penalty
        )
        values: dict[str, Any] = {
            "text": text,
            "voice": request.voice or self.default_voice,
            "speed": request.speed,
            "lang_code": language,
            "ref_audio": self._collapse(references),
            "ref_text": request.ref_text or self.default_ref_text,
            "instruct": request.instruct or self.default_instruct,
            "temperature": request.temperature,
            "top_k": request.top_k,
            "top_p": request.top_p,
            "repetition_penalty": repetition_penalty,
            "max_tokens": (
                request.max_tokens
                if request.max_tokens is not None
                else self.default_max_tokens
            ),
            "stream": request.stream,
            "streaming_interval": request.streaming_interval
            or self.default_streaming_interval,
            "verbose": False,
        }
        values.update(self.generation_kwargs)
        values.update(request.backend_kwargs)
        return self._filter_generation_kwargs(
            {key: value for key, value in values.items() if value is not None}
        )

    @staticmethod
    def _to_output(result: Any) -> TTSOutput:
        audio = np.asarray(result.audio, dtype=np.float32).squeeze()
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        return TTSOutput(
            array=audio,
            sample_rate=int(getattr(result, "sample_rate", 24000)),
            token_length=getattr(result, "token_count", None),
        )

    @staticmethod
    def _text_segments(text: Any) -> list[str]:
        if isinstance(text, str):
            return [text]
        if isinstance(text, list) and all(isinstance(item, str) for item in text):
            return text
        raise TypeError(
            "The MLX backend currently accepts request.text as a string or list of strings."
        )

    def _iter_outputs(self, request: TTSRequest) -> Iterator[TTSOutput]:
        references, temporary_paths = self._temporary_reference_files(
            request.speaker_files
        )
        try:
            for text in self._text_segments(request.text):
                arguments = self._generation_arguments(request, text, references)
                results = self.model.generate(**arguments)
                for result in results:
                    output = self._to_output(result)
                    if request.apply_novasr:
                        output = output.apply_super_resolution(device="mps")
                    yield output
        finally:
            for path in temporary_paths:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass

    def generate_speech(
        self, request: TTSRequest
    ) -> TTSOutput | Generator[TTSOutput, None, None]:
        if request.stream:
            return self._iter_outputs(request)

        outputs = list(self._iter_outputs(request))
        if not outputs:
            raise RuntimeError("The MLX model returned no audio.")
        return TTSOutput.combine_outputs(outputs)

    async def generate_speech_async(
        self, request: TTSRequest
    ) -> TTSOutput | AsyncGenerator[TTSOutput, None]:
        if not request.stream:
            result = await asyncio.to_thread(self.generate_speech, request)
            assert isinstance(result, TTSOutput)
            return result

        iterator = iter(self._iter_outputs(request))
        sentinel = object()

        async def stream() -> AsyncGenerator[TTSOutput, None]:
            while True:
                item = await asyncio.to_thread(_next_or_sentinel, iterator, sentinel)
                if item is sentinel:
                    break
                assert isinstance(item, TTSOutput)
                yield item

        return stream()

    async def shutdown(self) -> None:
        self.model = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except ImportError:
            pass
