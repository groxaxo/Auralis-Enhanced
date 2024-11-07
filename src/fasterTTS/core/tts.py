import asyncio
import json
import queue
import threading
from typing import AsyncGenerator, Optional, Dict, Union, Generator
from huggingface_hub import hf_hub_download

from src.fasterTTS.common.definitions.output import TTSOutput
from src.fasterTTS.common.definitions.requests import TTSRequest
from src.fasterTTS.common.scheduling.two_phase_scheduler import TwoPhaseScheduler
from src.fasterTTS.models.base import BaseAsyncTTSEngine, AudioOutputGenerator
from src.fasterTTS.models.registry import MODEL_REGISTRY

class TTS:
    def __init__(self):
        self.scheduler: Optional[TwoPhaseScheduler] = TwoPhaseScheduler()
        self.tts_engine: Optional[BaseAsyncTTSEngine] = None  # Initialize your TTS engine here

    def profile_tts_for_scheduler(self):
        """Profile the TTS engine to be used with the scheduler."""
        pass

    def from_pretrained(self, model_name_or_path: str, **kwargs):
        """Load a pretrained model compatible with HF path."""
        try:
            config_path = hf_hub_download(repo_id=model_name_or_path, filename='config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
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
                                   None
            }
            for i, gen in enumerate(audio_token_generators)
        ]

        return {
            'parallel_inputs': parallel_inputs,
            'request_ids': requests_ids,
            'request': input_request
        }

    async def _process_single_generator(self, gen_input: Dict) -> AudioOutputGenerator:
        """Process a single generator with its associated data."""
        try:
            async for chunk in self.tts_engine.process_tokens_to_speech( # type: ignore
                    generator=gen_input['generator'],
                    speaker_embeddings=gen_input['speaker_embedding'],
                    multimodal_data=gen_input['multimodal_data']
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
        """Generate speech for single or multiple requests."""


        if request.stream:
            def sync_generator():
                q = queue.Queue()

                async def async_gen():
                    try:
                        async for chunk in self.scheduler.run(
                            inputs=request,
                            first_phase_fn=self._prepare_generation_context,
                            second_phase_fn=self._second_phase_fn
                        ):
                            q.put(chunk)
                    except Exception as e:
                        q.put(e)
                    finally:
                        q.put(None)  # Sentinel to indicate completion

                def run_async_gen():
                    try:
                        asyncio.run(async_gen())
                    except Exception as e:
                        q.put(e)
                    finally:
                        q.put(None)

                # Start the coroutine
                threading.Thread(target=run_async_gen, daemon=True).start()

                while True:
                    item = q.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield item

            return sync_generator()
        else:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            async def generate_all():
                complete_audio = []
                async for chunk in self.scheduler.run(
                        inputs=request,
                        first_phase_fn=self._prepare_generation_context,
                        second_phase_fn=self._second_phase_fn
                ):
                    complete_audio.append(chunk)
                return TTSOutput.combine_outputs(complete_audio)

            return loop.run_until_complete(generate_all())
