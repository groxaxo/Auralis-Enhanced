import asyncio
import json
import inspect
import logging
import os
import time
import uuid
from functools import partial
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Union

from huggingface_hub import hf_hub_download

from auralis.backends import resolve_backend
from auralis.common.definitions.output import TTSOutput
from auralis.common.definitions.requests import TTSRequest
from auralis.common.logging.logger import setup_logger, set_vllm_logging_level
from auralis.common.metrics.performance import track_generation


class TTS:
    """High-performance text-to-speech facade with pluggable inference backends.

    The existing vLLM/XTTS path remains the default outside Apple Silicon. The
    optional MLX path is selected automatically on Apple Silicon or explicitly
    with ``backend="mlx"``.
    """

    def __init__(
        self,
        scheduler_max_concurrency: int = 1,
        vllm_logging_level: int = logging.DEBUG,
        backend: str = "auto",
    ):
        self.backend_name = resolve_backend(backend)
        self.scheduler = None
        self.tts_engine: Optional[Any] = None
        self.concurrency = scheduler_max_concurrency
        self.max_vllm_memory: Optional[int] = None
        self.logger = setup_logger(__file__)
        self.loop = None

        if self.backend_name == "vllm":
            set_vllm_logging_level(vllm_logging_level)
            from auralis.common.scheduling.two_phase_scheduler import TwoPhaseScheduler

            self.scheduler = TwoPhaseScheduler(scheduler_max_concurrency)

    @property
    def backend(self) -> str:
        """Name of the resolved inference backend."""

        return self.backend_name

    def _ensure_event_loop(self):
        if not self.loop:
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)

    def from_pretrained(self, model_name_or_path: str, **kwargs):
        """Load a pretrained model for the selected backend."""

        if self.backend_name == "mlx":
            from auralis.backends.mlx import MLXTTSEngine

            self.tts_engine = MLXTTSEngine.from_pretrained(
                model_name_or_path, **kwargs
            )
            return self

        from auralis.models import load_builtin_models
        from auralis.models.registry import MODEL_REGISTRY

        load_builtin_models()
        self._ensure_event_loop()

        try:
            with open(os.path.join(model_name_or_path, "config.json"), "r") as file:
                config = json.load(file)
        except FileNotFoundError:
            try:
                config_path = hf_hub_download(
                    repo_id=model_name_or_path, filename="config.json"
                )
                with open(config_path, "r") as file:
                    config = json.load(file)
            except Exception as exc:
                raise ValueError(
                    f"Could not load model from {model_name_or_path} locally or online: {exc}"
                ) from exc

        async def _load_model():
            model_type = config.get("model_type") or config.get("model", "xtts")
            try:
                model_class = MODEL_REGISTRY[model_type]
            except KeyError as exc:
                available = ", ".join(sorted(MODEL_REGISTRY)) or "none"
                raise ValueError(
                    f"Unsupported Auralis model type {model_type!r}. "
                    f"Registered model types: {available}."
                ) from exc
            return model_class.from_pretrained(model_name_or_path, **kwargs)

        self.tts_engine = self.loop.run_until_complete(_load_model())
        return self

    def _require_vllm_engine(self) -> None:
        if self.backend_name != "vllm":
            raise RuntimeError(
                "This operation belongs to the vLLM two-phase engine and is not "
                f"available for backend={self.backend_name!r}."
            )
        if self.tts_engine is None or self.scheduler is None:
            raise RuntimeError("Load a model with from_pretrained() before generation.")

    async def prepare_for_streaming_generation(self, request: TTSRequest):
        self._require_vllm_engine()
        conditioning_config = self.tts_engine.conditioning_config
        if (
            conditioning_config.speaker_embeddings
            or conditioning_config.gpt_like_decoder_conditioning
        ):
            gpt_cond_latent, speaker_embeddings = (
                await self.tts_engine.get_audio_conditioning(request.speaker_files)
            )
            return partial(
                self.tts_engine.get_generation_context,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embeddings=speaker_embeddings,
            )
        return None

    async def _prepare_generation_context(self, input_request: TTSRequest):
        self._require_vllm_engine()
        conditioning_config = self.tts_engine.conditioning_config
        input_request.start_time = time.time()
        if input_request.context_partial_function:
            (
                audio_token_generators,
                requests_ids,
                speaker_embeddings,
                gpt_like_decoder_conditioning,
            ) = await input_request.context_partial_function(input_request)
        else:
            audio_token_generators = None
            speaker_embeddings = None
            gpt_like_decoder_conditioning = None

            if (
                conditioning_config.speaker_embeddings
                and conditioning_config.gpt_like_decoder_conditioning
            ):
                (
                    audio_token_generators,
                    requests_ids,
                    speaker_embeddings,
                    gpt_like_decoder_conditioning,
                ) = await self.tts_engine.get_generation_context(input_request)
            elif conditioning_config.speaker_embeddings:
                (
                    audio_token_generators,
                    requests_ids,
                    speaker_embeddings,
                ) = await self.tts_engine.get_generation_context(input_request)
            elif conditioning_config.gpt_like_decoder_conditioning:
                (
                    audio_token_generators,
                    requests_ids,
                    gpt_like_decoder_conditioning,
                ) = await self.tts_engine.get_generation_context(input_request)
            else:
                audio_token_generators, requests_ids = (
                    await self.tts_engine.get_generation_context(input_request)
                )

        parallel_inputs = [
            {
                "generator": generator,
                "speaker_embedding": (
                    speaker_embeddings[index]
                    if speaker_embeddings is not None
                    and isinstance(speaker_embeddings, list)
                    else speaker_embeddings
                    if speaker_embeddings is not None
                    else None
                ),
                "multimodal_data": (
                    gpt_like_decoder_conditioning[index]
                    if gpt_like_decoder_conditioning is not None
                    and isinstance(gpt_like_decoder_conditioning, list)
                    else gpt_like_decoder_conditioning
                    if gpt_like_decoder_conditioning is not None
                    else None
                ),
                "request": input_request,
            }
            for index, generator in enumerate(audio_token_generators)
        ]

        return {"parallel_inputs": parallel_inputs, "request": input_request}

    async def _process_single_generator(self, gen_input: Dict) -> AsyncGenerator:
        try:
            async for chunk in self.tts_engine.process_tokens_to_speech(
                generator=gen_input["generator"],
                speaker_embeddings=gen_input["speaker_embedding"],
                multimodal_data=gen_input["multimodal_data"],
                request=gen_input["request"],
            ):
                yield chunk
        except Exception:
            raise

    @track_generation
    async def _second_phase_fn(self, gen_input: Dict) -> AsyncGenerator:
        async for chunk in self._process_single_generator(gen_input):
            yield chunk

    async def generate_speech_async(
        self, request: TTSRequest
    ) -> Union[AsyncGenerator[TTSOutput, None], TTSOutput]:
        if self.tts_engine is None:
            raise RuntimeError("Load a model with from_pretrained() before generation.")

        if self.backend_name == "mlx":
            return await self.tts_engine.generate_speech_async(request)

        self._require_vllm_engine()
        self._ensure_event_loop()

        async def process_chunks():
            chunks = []
            try:
                async for chunk in self.scheduler.run(
                    inputs=request,
                    request_id=request.request_id,
                    first_phase_fn=self._prepare_generation_context,
                    second_phase_fn=self._second_phase_fn,
                ):
                    if request.stream:
                        yield chunk
                    else:
                        chunks.append(chunk)
            except Exception as exc:
                self.logger.error(f"Error during speech generation: {exc}")
                raise

            if not request.stream:
                yield TTSOutput.combine_outputs(chunks)

        if request.stream:
            return process_chunks()

        async for result in process_chunks():
            return result
        raise RuntimeError("The TTS engine returned no audio.")

    @staticmethod
    def split_requests(request: TTSRequest, max_length: int = 100000) -> List[TTSRequest]:
        if not isinstance(request.text, str) or len(request.text) <= max_length:
            return [request]

        text_chunks = [
            request.text[index : index + max_length]
            for index in range(0, len(request.text), max_length)
        ]

        requests: list[TTSRequest] = []
        for chunk in text_chunks:
            copied = request.copy()
            copied.text = chunk
            copied.request_id = uuid.uuid4().hex
            requests.append(copied)
        return requests

    async def _process_multiple_requests(
        self, requests: List[TTSRequest], results: Optional[List] = None
    ) -> Optional[TTSOutput]:
        self._require_vllm_engine()
        output_queues = [asyncio.Queue() for _ in requests] if results is not None else None

        async def process_subrequest(
            index: int, sub_request: TTSRequest, queue: Optional[asyncio.Queue] = None
        ):
            chunks = []
            async for chunk in self.scheduler.run(
                inputs=sub_request,
                request_id=sub_request.request_id,
                first_phase_fn=self._prepare_generation_context,
                second_phase_fn=self._second_phase_fn,
            ):
                chunks.append(chunk)
                if queue is not None:
                    await queue.put(chunk)

            if queue is not None:
                await queue.put(None)
            return chunks

        tasks = [
            asyncio.create_task(
                process_subrequest(
                    index,
                    sub_request,
                    output_queues[index] if output_queues else None,
                )
            )
            for index, sub_request in enumerate(requests)
        ]

        if results is not None:
            for index, queue in enumerate(output_queues):
                while True:
                    chunk = await queue.get()
                    if chunk is None:
                        break
                    results[index].append(chunk)
            await asyncio.gather(*tasks)
            return None

        all_chunks = await asyncio.gather(*tasks)
        complete_audio = [chunk for chunks in all_chunks for chunk in chunks]
        return TTSOutput.combine_outputs(complete_audio)

    def generate_speech(
        self, request: TTSRequest
    ) -> Union[Generator[TTSOutput, None, None], TTSOutput]:
        if self.tts_engine is None:
            raise RuntimeError("Load a model with from_pretrained() before generation.")

        if self.backend_name == "mlx":
            return self.tts_engine.generate_speech(request)

        self._require_vllm_engine()
        self._ensure_event_loop()
        requests = self.split_requests(request)

        if request.stream:

            def streaming_wrapper():
                for sub_request in requests:

                    async def process_stream():
                        try:
                            async for chunk in self.scheduler.run(
                                inputs=sub_request,
                                request_id=sub_request.request_id,
                                first_phase_fn=self._prepare_generation_context,
                                second_phase_fn=self._second_phase_fn,
                            ):
                                yield chunk
                        except Exception as exc:
                            self.logger.error(f"Error during streaming: {exc}")
                            raise

                    generator = process_stream()
                    try:
                        while True:
                            chunk = self.loop.run_until_complete(anext(generator))
                            yield chunk
                    except StopAsyncIteration:
                        pass

            return streaming_wrapper()

        return self.loop.run_until_complete(self._process_multiple_requests(requests))

    async def shutdown(self):
        if self.scheduler:
            await self.scheduler.shutdown()
        if self.tts_engine and hasattr(self.tts_engine, "shutdown"):
            result = self.tts_engine.shutdown()
            if inspect.isawaitable(result):
                await result
