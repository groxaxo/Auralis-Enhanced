import asyncio
import json
import queue
import threading
from typing import AsyncGenerator, Optional, Dict, Union, Generator, List
from huggingface_hub import hf_hub_download

from fasterTTS.common.definitions.output import TTSOutput
from fasterTTS.common.definitions.requests import TTSRequest
from fasterTTS.common.scheduling.two_phase_scheduler import TwoPhaseScheduler
from fasterTTS.models.base import BaseAsyncTTSEngine, AudioOutputGenerator
from fasterTTS.models.registry import MODEL_REGISTRY


class TTS:
    def __init__(self, scheduler_max_concurrency: int = 10):
        self.scheduler: Optional[TwoPhaseScheduler] = TwoPhaseScheduler(scheduler_max_concurrency)
        self.tts_engine: Optional[BaseAsyncTTSEngine] = None  # Initialize your TTS engine here
        self.concurrency = scheduler_max_concurrency
        self.max_vllm_memory: Optional[int] = None
        self.set_vllm_memory(scheduler_max_concurrency)

        # Create a persistent event loop and thread for background tasks
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()

    def _run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def set_vllm_memory(self, scheduler_max_concurrency: int):
        """
        Based on the expected concurrency we allocate memory for VLLM.
        This is not intended as a permanent solution but rather as a temporary workaround.
        """
        # some hardcoded values for memory allocation, had been proven to work
        match scheduler_max_concurrency:
            case n if n <= 10:
                self.max_vllm_memory = 2
            case n if n <= 20:
                self.max_vllm_memory = 2.5
            case n if n <= 30:
                self.max_vllm_memory = 3
            case n if n <= 40:
                self.max_vllm_memory = 3.5
            case _:
                self.max_vllm_memory = 6

    def from_pretrained(self, model_name_or_path: str, **kwargs):
        """Load a pretrained model compatible with HF path."""
        try:
            config_path = hf_hub_download(repo_id=model_name_or_path, filename='config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            kwargs['max_vllm_memory'] = self.max_vllm_memory
            kwargs['max_concurrency'] = self.concurrency
            self.tts_engine = MODEL_REGISTRY[config['model_type']].from_pretrained(model_name_or_path, **kwargs)
            return self
        except Exception as e:
            raise ValueError(f"Could not load model from {model_name_or_path}: {e}")


    async def _prepare_generation_context(self,
                                          input_request: TTSRequest,
                                          ):
        """Prepare the generation context for the first phase."""
        conditioning_config = self.tts_engine.conditioning_config
        audio_token_generators, speaker_embeddings, gpt_like_decoder_conditioning = None, None, None

        if conditioning_config.speaker_embeddings and conditioning_config.gpt_like_decoder_conditioning:
            (audio_token_generators, requests_ids,
             speaker_embeddings,
             gpt_like_decoder_conditioning) = await self.tts_engine.get_generation_context(input_request)
        elif conditioning_config.speaker_embeddings:
            (audio_token_generators, requests_ids,
             speaker_embeddings) = await self.tts_engine.get_generation_context(input_request)
        elif conditioning_config.gpt_like_decoder_conditioning:
            (audio_token_generators, requests_ids,
             gpt_like_decoder_conditioning) = await self.tts_engine.get_generation_context(input_request)
        else:
            audio_token_generators, requests_ids = await self.tts_engine.get_generation_context(input_request)

        # Pack everything needed for parallel processing, if some conditioning is singular we repeat it for the batch
        parallel_inputs = [
            {
                'generator': gen,
                'speaker_embedding': speaker_embeddings[i] if
                                     speaker_embeddings is not None and isinstance(speaker_embeddings, list) else
                                     speaker_embeddings if speaker_embeddings is not None else
                                     None,
                'multimodal_data': gpt_like_decoder_conditioning[i] if
                                   gpt_like_decoder_conditioning is not None and isinstance(gpt_like_decoder_conditioning, list) else
                                   gpt_like_decoder_conditioning if gpt_like_decoder_conditioning is not None else
                                   None,
                'request_id': requests_ids[i],

            }
            for i, gen in enumerate(audio_token_generators)
        ]

        return {
            'parallel_inputs': parallel_inputs,
            'request': input_request
        }

    async def _process_single_generator(self, gen_input: Dict) -> AudioOutputGenerator:
        """Process a single generator with its associated data."""
        try:
            async for chunk in self.tts_engine.process_tokens_to_speech( # type: ignore
                    generator=gen_input['generator'],
                    speaker_embeddings=gen_input['speaker_embedding'],
                    multimodal_data=gen_input['multimodal_data'],
                    request_id=gen_input['request_id'],
            ):
                yield chunk
        except Exception as e:
            raise e

    async def _second_phase_fn(self, gen_input: Dict) -> AudioOutputGenerator:
        """
        Second phase: Generate speech using the existing TTS engine.
        """

        async for chunk in self._process_single_generator(gen_input):
            yield chunk

    async def generate_speech_async(self,
                                    requests: TTSRequest
                                    ) -> Union[AsyncGenerator[TTSOutput, None], TTSOutput]:
        """Generate speech for single request asynchronously."""

        if requests.stream:
            async def async_gen():
                async for chunk in self.scheduler.run(
                    inputs=requests,
                    first_phase_fn=self._prepare_generation_context,
                    second_phase_fn=self._second_phase_fn
                ):
                    yield chunk

            return async_gen()
        else:
            complete_audio = []
            async for chunk in self.scheduler.run(
                    inputs=requests,
                    first_phase_fn=self._prepare_generation_context,
                    second_phase_fn=self._second_phase_fn
            ):
                complete_audio.append(chunk)
            return TTSOutput.combine_outputs(complete_audio)

    def generate_speech(self, request: TTSRequest) -> Union[Generator[TTSOutput, None, None], TTSOutput]:
        """Generate speech for single or multiple requests, handling long texts by splitting."""

        def split_requests(request: TTSRequest, max_length: int = 100000) -> List[TTSRequest]:
            """Split a single request into multiple requests with shorter text chunks."""
            if len(request.text) <= max_length:
                return [request]

            text_chunks = [request.text[i:i + max_length]
                           for i in range(0, len(request.text), max_length)]

            return [
                TTSRequest(
                    text=chunk,
                    language=request.language,
                    speaker_files=request.speaker_files,
                    stream=request.stream
                ) for chunk in text_chunks
            ]

        requests = split_requests(request)

        if request.stream:
            def sync_generator():
                q = queue.Queue()

                async def async_gen():
                    try:
                        for sub_request in requests:
                            async for chunk in self.scheduler.run(
                                    inputs=sub_request,
                                    first_phase_fn=self._prepare_generation_context,
                                    second_phase_fn=self._second_phase_fn
                            ):
                                q.put(chunk)
                    except Exception as e:
                        q.put(e)
                    finally:
                        q.put(None)

                asyncio.run_coroutine_threadsafe(async_gen(), self.loop)

                while True:
                    item = q.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield item

            return sync_generator()
        else:
            complete_audio = []

            async def async_combine_chunks():
                for sub_request in requests:
                    async for chunk in self.scheduler.run(
                            inputs=sub_request,
                            first_phase_fn=self._prepare_generation_context,
                            second_phase_fn=self._second_phase_fn
                    ):
                        complete_audio.append(chunk)
                return TTSOutput.combine_outputs(complete_audio)

            future = asyncio.run_coroutine_threadsafe(async_combine_chunks(), self.loop)
            result = future.result()
            return result

    async def shutdown(self):
        if self.scheduler:
            await self.scheduler.shutdown()
        if self.tts_engine and hasattr(self.tts_engine, 'shutdown'):
            await self.tts_engine.shutdown()
        self.loop.call_soon_threadsafe(self.loop.stop())
        self.loop_thread.join()
