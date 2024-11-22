import asyncio
import json
import queue
import threading
import time
import uuid
from typing import AsyncGenerator, Optional, Dict, Union, Generator, List
from huggingface_hub import hf_hub_download
from torch.onnx.symbolic_opset11 import chunk

from auralis.common.logging.logger import setup_logger
from auralis.common.definitions.output import TTSOutput
from auralis.common.definitions.requests import TTSRequest
from auralis.common.metrics.performance import track_generation
from auralis.common.scheduling.two_phase_scheduler import TwoPhaseScheduler
from auralis.models.base import BaseAsyncTTSEngine, AudioOutputGenerator



class TTS:
    def __init__(self, scheduler_max_concurrency: int = 10):
        self.scheduler: Optional[TwoPhaseScheduler] = TwoPhaseScheduler(scheduler_max_concurrency)
        self.tts_engine: Optional[BaseAsyncTTSEngine] = None  # Initialize your TTS engine here
        self.concurrency = scheduler_max_concurrency
        self.max_vllm_memory: Optional[int] = None
        self.set_vllm_memory(scheduler_max_concurrency)
        self.logger = setup_logger(__file__)

        try:
            # Try to get existing loop
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no loop exists, create new one
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
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
                self.max_vllm_memory = 7
            case n if n <= 20:
                self.max_vllm_memory = 3.2
            case n if n <= 30:
                self.max_vllm_memory = 3.75
            case n if n <= 40:
                self.max_vllm_memory = 4.3
            case _:
                self.max_vllm_memory = 6

    def from_pretrained(self, model_name_or_path: str, **kwargs):
        """Load a pretrained model compatible with HF path."""
        from auralis.models.registry import MODEL_REGISTRY # lazy import to avoid circular imports

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
        input_request.start_time = time.time()
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
                'request': input_request,

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
                    request = gen_input['request'],
            ):
                yield chunk
        except Exception as e:
            raise e

    @track_generation
    async def _second_phase_fn(self, gen_input: Dict) -> AudioOutputGenerator:
        """Second phase: Generate speech using the existing TTS engine."""
        async for chunk in self._process_single_generator(gen_input):
            yield chunk

    async def generate_speech_async(self, request: TTSRequest) -> Union[AsyncGenerator[TTSOutput, None], TTSOutput]:
        async def process_chunks():
            chunks = []
            try:
                async for chunk in self.scheduler.run(
                        inputs=request,
                        request_id=request.request_id,
                        first_phase_fn=self._prepare_generation_context,
                        second_phase_fn=self._second_phase_fn
                ):
                    if request.stream:
                        yield chunk
                    chunks.append(chunk)
            except Exception as e:
                self.logger.error(f"Error during speech generation: {e}")
                raise

            if not request.stream:
                yield TTSOutput.combine_outputs(chunks)

        if request.stream:
            return process_chunks()
        else:
            async for result in process_chunks():
                return result

    @staticmethod
    def split_requests(request: TTSRequest, max_length: int = 100000) -> List[TTSRequest]:
        """Split a single request into multiple requests with shorter text chunks to fix max tokenizer len."""
        if len(request.text) <= max_length:
            return [request]

        text_chunks = [request.text[i:i + max_length]
                       for i in range(0, len(request.text), max_length)]

        return [
            (copy := request.copy(), setattr(copy, 'text', chunk), setattr(copy, 'request_id', uuid.uuid4().hex))[0]
            for chunk in text_chunks
        ]

    async def _process_multiple_requests(self, requests: List[TTSRequest], results: Optional[List] = None) -> Optional[
        TTSOutput]:
        # Use a queue for each sub-request to maintain order while allowing parallel processing
        output_queues = [asyncio.Queue() for _ in requests] if results is not None else None

        async def process_subrequest(idx, sub_request, queue: Optional[asyncio.Queue] = None):
            chunks = []
            async for chunk in self.scheduler.run(
                    inputs=sub_request,
                    request_id=sub_request.request_id,
                    first_phase_fn=self._prepare_generation_context,
                    second_phase_fn=self._second_phase_fn
            ):
                chunks.append(chunk)
                if queue is not None:
                    await queue.put(chunk)  # Put chunks in their respective queue

            if queue is not None:
                await queue.put(None)  # Signal end of stream
            return chunks

        # Create and start all tasks in parallel for maximum throughput
        tasks = [
            asyncio.create_task(
                process_subrequest(
                    idx,
                    sub_request,
                    output_queues[idx] if output_queues else None
                )
            )
            for idx, sub_request in enumerate(requests)
        ]

        if results is not None:
            # Streaming case: consume queues in order to maintain sequence
            for idx, queue in enumerate(output_queues):
                while True:
                    chunk = await queue.get()
                    if chunk is None:  # End of stream for this sub-request
                        break
                    results[idx].append(chunk)
            return None
        else:
            # Non-streaming case: gather all results
            all_chunks = await asyncio.gather(*tasks)
            complete_audio = [chunk for chunks in all_chunks for chunk in chunks]
            return TTSOutput.combine_outputs(complete_audio)

    def generate_speech(self, request: TTSRequest) -> Union[Generator[TTSOutput, None, None], TTSOutput]:
        requests = self.split_requests(request)

        if request.stream:
            def sync_generator():
                # Initialize result buffer for each sub-request
                results = [[] for _ in requests]
                future = asyncio.run_coroutine_threadsafe(
                    self._process_multiple_requests(requests, results),
                    self.loop
                )

                current_idx = 0  # Track current sub-request being processed
                while current_idx < len(results):
                    # Yield available chunks for current sub-request
                    while results[current_idx]:
                        yield results[current_idx].pop(0)

                    # Move to next sub-request if current one is complete
                    if future.done() and not results[current_idx]:
                        current_idx += 1
                        continue

                    # Check for errors in the async processing
                    if future.done() and future.exception():
                        raise future.exception()

                    # Small sleep to prevent CPU overload
                    time.sleep(0.01)

            return sync_generator()
        else:
            future = asyncio.run_coroutine_threadsafe(self._process_multiple_requests(requests), self.loop)
            return future.result()

    async def shutdown(self):
        if self.scheduler:
            await self.scheduler.shutdown()
        if self.tts_engine and hasattr(self.tts_engine, 'shutdown'):
            await self.tts_engine.shutdown()
        self.loop.call_soon_threadsafe(self.loop.stop())
        self.loop_thread.join()
