import asyncio
from typing import AsyncGenerator, Optional, Dict, Any, Union, Generator, List

from fasterTTS.common.output import TTSOutput
from fasterTTS.common.requests import TTSRequest
from fasterTTS.common.scheduler import GeneratorTwoPhaseScheduler
from fasterTTS.models.base_tts_engine import BaseAsyncTTSEngine


class TTS:
    def __init__(self):
        self.scheduler: Optional[GeneratorTwoPhaseScheduler] = GeneratorTwoPhaseScheduler()
        self.tts_engine: Optional[BaseAsyncTTSEngine] = None  # Initialize your TTS engine here

    def profile_tts_for_scheduler(self):
        """Profile the TTS engine to be used with the scheduler."""
        pass

    def from_pretrained(self, model_name_or_path: str):
        """Load a pretrained model compatible with HF path."""
        pass

    async def _prepare_generation_context(self, input_request: TTSRequest, metadata: Dict[str, Any]):
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

    async def _process_single_generator(self, gen_input: Dict, metadata: Dict[str, Any]) -> AsyncGenerator[
        TTSOutput, None]:
        """Process a single generator with its associated data."""
        try:
            async for chunk in self.tts_engine.process_tokens_to_speech(
                    generator=gen_input['generator'],
                    speaker_embeddings=gen_input['speaker_embedding'],
                    multimodal_data=gen_input['multimodal_data']
            ):
                yield chunk
        except Exception as e:
            raise e

    def generate_speech(self, requests: Union[TTSRequest, List[TTSRequest]]) -> Union[Generator[TTSOutput, None, None], TTSOutput]:
        """Generate speech for single or multiple requests."""
        if not isinstance(requests, list):
            requests = [requests]

        if requests[0].stream:
            def sync_generator():
                async def async_gen():
                    metadata = [{'request': req} for req in requests]
                    async for chunk in self.scheduler.run(
                        inputs=requests,
                        metadata_list=metadata,
                        first_phase_fn=self._prepare_generation_context,
                        second_phase_fn=self._process_single_generator
                    ):
                        yield chunk

                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                agen = async_gen()
                while True:
                    try:
                        chunk = loop.run_until_complete(agen.__anext__())
                        yield chunk
                    except StopAsyncIteration:
                        break

            return sync_generator()
        else:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            async def generate_all():
                metadata = [{'request': req} for req in requests]
                complete_audio = []
                async for chunk in self.scheduler.run(
                        inputs=requests,
                        metadata_list=metadata,
                        first_phase_fn=self._prepare_generation_context,
                        second_phase_fn=self._process_single_generator
                ):
                    complete_audio.append(chunk)
                return TTSOutput.combine_outputs(complete_audio)

            return loop.run_until_complete(generate_all())