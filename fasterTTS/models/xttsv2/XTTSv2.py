import asyncio
import functools
import logging
import uuid

from pathlib import Path
from typing import Optional, List, Tuple, Union, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

import librosa
import torch
import torchaudio
from torch import nn

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams, TokensPrompt, RequestOutput
from vllm.multimodal import MultiModalDataDict
from vllm.utils import Counter

from ...common.output import TTSOutput
from ...common.requests import TTSRequest
from ...common.utilities import wav_to_mel_cloning, load_audio

from .components.vllm_mm_gpt import LearnedPositionEmbeddings
from .hf_files.tokenizer import XTTSTokenizerFast
from .hf_files.xttsv2_config import XTTSConfig
from .hf_files.xttsv2_gpt_config import XTTSGPTConfig

from .components.vllm.hidden_state_collector import HiddenStatesCollector
from .components.vllm.hijack import ExtendedSamplingParams, LogitsRepetitionPenalizer
from .components._tts.layers.xtts.hifigan_decoder import HifiDecoder
from .components._tts.layers.xtts.latent_encoder import ConditioningEncoder
from .components._tts.layers.xtts.perceiver_encoder import PerceiverResampler


class Xtts(nn.Module):
    """Async XTTS model implementation using VLLM's AsyncEngine."""

    def __init__(self, hifi_config: XTTSConfig, gpt_config: XTTSGPTConfig, tensor_parallel_size: int = 1, **kwargs):
        super().__init__()

        self.hifi_config = hifi_config
        self.gpt_config = gpt_config
        self.mel_bos_token_id = gpt_config.start_audio_token
        self.mel_eos_token_id = gpt_config.stop_audio_token
        self.tp = tensor_parallel_size
        self.tokenizer = XTTSTokenizerFast.from_pretrained("AstraMindAI/xtts2-gpt")
        self.request_counter = Counter()
        self.executor = ThreadPoolExecutor(max_workers=4)  # For CPU-bound tasks
        self.hidden_states_collector = HiddenStatesCollector()

        # Register buffer before creating modules
        self.register_buffer("mel_stats", torch.ones(80))

        # Initialize all nn.Module components
        self.conditioning_encoder = ConditioningEncoder(
            gpt_config.audio_config.mel_channels,
            gpt_config.hidden_size,
            num_attn_heads=gpt_config.num_attention_heads
        )

        self.text_embedding = nn.Embedding(
            gpt_config.number_text_tokens,
            gpt_config.hidden_size
        )

        self.text_pos_embedding = (
            LearnedPositionEmbeddings(
                gpt_config.max_text_tokens + 2,
                gpt_config.hidden_size,
                supports_pp=False
            )
            if gpt_config.max_audio_tokens != -1
            else functools.partial(gpt_config.null_position_embeddings, dim=gpt_config.hidden_size)
        )

        if gpt_config.use_perceiver_resampler:
            self.conditioning_perceiver = PerceiverResampler(
                dim=gpt_config.hidden_size,
                depth=2,
                dim_context=gpt_config.hidden_size,
                num_latents=32,
                dim_head=64,
                heads=8,
                ff_mult=4,
                use_flash_attn=False,
            )

        # Initialize HiFi-GAN decoder
        self.hifigan_decoder = HifiDecoder(
            input_sample_rate=self.hifi_config.input_sample_rate,
            output_sample_rate=self.hifi_config.output_sample_rate,
            output_hop_length=self.hifi_config.output_hop_length,
            ar_mel_length_compression=self.hifi_config.gpt_code_stride_len,
            decoder_input_dim=self.hifi_config.decoder_input_dim,
            d_vector_dim=self.hifi_config.d_vector_dim,
            cond_d_vector_in_each_upsampling_layer=self.hifi_config.cond_d_vector_in_each_upsampling_layer,
        )

        # Kept for model loading purposes
        self.text_head = nn.Linear(gpt_config.hidden_size, gpt_config.number_text_tokens, bias=True)
        self.final_norm = nn.LayerNorm(gpt_config.hidden_size, eps=1e-5, bias=True)

        # Initialize VLLM engine at the end
        self.init_vllm_engine()

        # Semaphore for concurrency control
        self.max_concurrency = 10
        self.semaphore = asyncio.BoundedSemaphore(self.max_concurrency)

    def half(self):
        # We cannot permit downcasting since it will throw an error while padding
        return

    def to(self, *args, **kwargs):
        # Block downcasting
        dtype = kwargs.get('dtype', None)
        if dtype == torch.float16 or dtype == torch.bfloat16:
            kwargs['dtype'] = torch.float32
        elif len(args) > 0 and (args[0] == torch.float16 or args[0] == torch.bfloat16):
            args = list(args)
            args[0] = torch.float32
            args = tuple(args)
        return super().to(*args, **kwargs)

    @property
    def device(self):
        """Get the current device of the model."""
        return next(self.parameters()).device

    @property
    def dtype(self):
        """Get the current dtype of the model."""
        return next(self.parameters()).dtype

    @staticmethod
    def get_memory_percentage(memory: int) -> float:
        """Get memory percentage."""
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        allocated_memory = torch.cuda.memory_allocated(0)
        available_memory = total_memory - reserved_memory - allocated_memory
        return memory / available_memory

    def init_vllm_engine(self):
        """Initialize models with AsyncVLLMEngine."""
        engine_args = AsyncEngineArgs(
            model="AstraMindAI/xtts2-gpt",
            tensor_parallel_size=self.tp,
            dtype="auto",
            disable_log_stats=True,
            max_model_len=self.gpt_config.max_text_tokens + self.gpt_config.max_audio_tokens,
            gpu_memory_utilization=self.get_memory_percentage(3 * 1024 ** 3),
            trust_remote_code=True,
            enforce_eager=True,
            limit_mm_per_prompt={"audio": 1},
            max_num_batched_tokens=7296,
        )

        self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            torch_dtype: torch.dtype = torch.float32,
            device_map: Optional[str] = "auto",
            tensor_parallel_size: int = 1,
            **kwargs,
    ) -> "Xtts":
        """Load pretrained XTTS model from HuggingFace Hub."""
        from huggingface_hub import hf_hub_download
        import json
        import os

        # Download and load configs
        if not os.path.exists(pretrained_model_name_or_path):
            config_file = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="config.json"
            )
            with open(config_file, 'r') as f:
                config = json.load(f)

        else:
            # Load from local path
            with open(os.path.join(pretrained_model_name_or_path, "config.json"), 'r') as f:
                config = json.load(f)

        # Initialize configs
        gpt_config = XTTSGPTConfig(**config['gpt_config'])
        hifi_config = XTTSConfig(**config)

        # Initialize model
        model = cls(
            hifi_config=hifi_config,
            gpt_config=gpt_config,
            tensor_parallel_size=tensor_parallel_size,
            **kwargs
        )

        # Load model weights
        if not os.path.exists(pretrained_model_name_or_path):
            hifigan_weights = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="xtts-v2.safetensors"
            )
        else:
            hifigan_weights = os.path.join(pretrained_model_name_or_path, "xtts-v2.safetensors")

        import safetensors.torch

        # Load HiFi-GAN weights
        hifigan_state = safetensors.torch.load_file(hifigan_weights)
        model.load_state_dict(hifigan_state)

        # Set model properties
        model.config = config

        # Cast model to specified dtype
        model = model.to(torch_dtype)
        model = model.to('cuda')

        return model

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

    @torch.inference_mode()
    def get_speaker_embedding(self, audio, sr):
        audio_16k = torchaudio.functional.resample(audio, sr, 16000)
        return (
            self.hifigan_decoder.speaker_encoder.forward(audio_16k.to(self.device), l2_norm=True)
            .unsqueeze(-1)
            .to(self.device)
        )

    @torch.inference_mode()
    def get_gpt_cond_latents(self, audio, sr, length: int = 30, chunk_length: int = 6):
        """Compute the conditioning latents for the GPT model from the given audio."""
        if sr != 22050:
            audio = torchaudio.functional.resample(audio, sr, 22050)
        if length > 0:
            audio = audio[:, : 22050 * length]
        if self.gpt_config.use_perceiver_resampler:
            style_embs = []
            for i in range(0, audio.shape[1], 22050 * chunk_length):
                audio_chunk = audio[:, i: i + 22050 * chunk_length]

                # if the chunk is too short ignore it
                if audio_chunk.size(-1) < 22050 * 0.33:
                    continue

                mel_chunk = wav_to_mel_cloning(
                    audio_chunk,
                    mel_norms=self.mel_stats.cpu(),
                    n_fft=2048,
                    hop_length=256,
                    win_length=1024,
                    power=2,
                    normalized=False,
                    sample_rate=22050,
                    f_min=0,
                    f_max=8000,
                    n_mels=80,
                )
                style_emb = self.get_style_emb(mel_chunk.to(self.device), None)
                style_embs.append(style_emb)

            # mean style embedding
            cond_latent = torch.stack(style_embs).mean(dim=0)
        else:
            mel = wav_to_mel_cloning(
                audio,
                mel_norms=self.mel_stats.cpu(),
                n_fft=4096,
                hop_length=1024,
                win_length=4096,
                power=2,
                normalized=False,
                sample_rate=22050,
                f_min=0,
                f_max=8000,
                n_mels=80,
            )
            cond_latent = self.get_style_emb(mel.to(self.device))
        return cond_latent.transpose(1, 2)

    @torch.inference_mode()
    def get_conditioning_latents(
            self,
            audio_path,
            max_ref_length=30,
            gpt_cond_len=6,
            gpt_cond_chunk_len=6,
            librosa_trim_db=None,
            sound_norm_refs=False,
            load_sr=22050,
    ):
        """Get the conditioning latents for the GPT model from the given audio."""
        # Deal with multiple references
        assert isinstance(audio_path, str) or isinstance(audio_path, list), "audio_path must be a string or a list."

        if not isinstance(audio_path, list):
            audio_paths = [audio_path]
        else:
            audio_paths = audio_path

        speaker_embeddings = []
        audios = []
        for file_path in audio_paths:
            audio = load_audio(file_path, load_sr)
            audio = audio[:, : load_sr * max_ref_length].to(self.device).to(self.dtype)
            if sound_norm_refs:
                audio = (audio / torch.abs(audio).max()) * 0.75
            if librosa_trim_db is not None:
                audio = librosa.effects.trim(audio, top_db=librosa_trim_db)[0]

            # Compute latents for the decoder
            speaker_embedding = self.get_speaker_embedding(audio, load_sr)
            speaker_embeddings.append(speaker_embedding)

            audios.append(audio)

        # Merge all the audios and compute the latents for the GPT
        full_audio = torch.cat(audios, dim=-1)
        gpt_cond_latents = self.get_gpt_cond_latents(
            full_audio, load_sr, length=gpt_cond_len, chunk_length=gpt_cond_chunk_len
        )  # [1, 1024, T]

        speaker_embedding = torch.stack(speaker_embeddings)
        speaker_embedding = speaker_embedding.mean(dim=0)

        return gpt_cond_latents, speaker_embedding

    def get_style_emb(self, cond_input: torch.Tensor, return_latent: Optional[bool] = False) -> torch.Tensor:
        """Get conditioning embeddings from mel spectrograms."""
        if not return_latent:
            if cond_input.ndim == 4:
                cond_input = cond_input.squeeze(1)
            conds = self.conditioning_encoder(cond_input)

            if hasattr(self, 'conditioning_perceiver'):
                conds = self.conditioning_perceiver(
                    conds.permute(0, 2, 1)
                ).transpose(1, 2)
        else:
            conds = cond_input.unsqueeze(1)
        return conds

    async def prepare_text_tokens_async(self, text: str, language: str, split_text=False) \
            -> Tuple[List[Union[int, List[int]]], List[torch.Tensor]]:
        """Prepare text tokens for the given text and language."""

        async def elaborate_tokens(text_tokens: List[int]) -> torch.Tensor:
            text_tokens.insert(0, self.tokenizer.bos_token_id)
            text_tokens.append(self.tokenizer.eos_token_id)
            return torch.tensor(text_tokens).unsqueeze(0).to(self.text_embedding.weight.device)

        async def embed_tokens(text_tokens: Union[torch.Tensor, List[torch.Tensor]]) -> List[torch.Tensor]:
            embeds = []
            if isinstance(text_tokens, list):
                for list_element in text_tokens:
                    embeds.append(self.text_embedding(list_element) + self.text_pos_embedding(list_element))
            else:
                embeds.append(self.text_embedding(text_tokens) + self.text_pos_embedding(text_tokens))
            return embeds

        fake_tokens_for_audio_generation = []
        if split_text:
            text_tokens = self.tokenizer.batch_encode_with_split(text, lang=[language])
            for idx, text_token in enumerate(text_tokens):
                text_tokens[idx] = await elaborate_tokens(text_token)
                fake_tokens_for_audio_generation.append([1] * len(text_token))
        else:
            text_tokens = self.tokenizer.batch_encode(text, lang=[language])
            text_tokens = await elaborate_tokens(text_tokens)
            fake_tokens_for_audio_generation = [1] * len(text_tokens)
        return fake_tokens_for_audio_generation, await embed_tokens(text_tokens)

    async def prepare_inputs_async(self, text: str, language: str, speaker_file: Union[str, Path],
                                   max_ref_length: int, gpt_cond_len: int, gpt_cond_chunk_len: int, split_text: bool) \
            -> Tuple[List[List[int]], List[torch.Tensor], torch.Tensor]:
        """Prepare input text with conditioning tokens. Return combined conditioning latents"""
        # Tokenize text based on the language
        text_tokens, text_embeddings = await self.prepare_text_tokens_async(text, language, split_text)

        # Load the speaker file and convert it to a tensor
        gpt_cond_latent, speaker_embeddings = await self.get_conditioning_latents_async(
            speaker_file,
            max_ref_length,
            gpt_cond_len,
            gpt_cond_chunk_len
        )

        cond_latents = []
        for text_embedding in text_embeddings:
            # Concatenate along sequence dimension
            cond_latents.append((torch.cat([gpt_cond_latent, text_embedding], dim=1).squeeze(0)
                                 .to(self.llm_engine.engine.model_config.dtype)))

        return text_tokens, cond_latents, speaker_embeddings

    async def get_conditioning_latents_async(
            self,
            audio_path,
            max_ref_length=30,
            gpt_cond_len=6,
            gpt_cond_chunk_len=6,
            librosa_trim_db=None,
            sound_norm_refs=False,
            load_sr=22050,
    ):
        """Async version of get_conditioning_latents with concurrency control."""
        async with self.semaphore:
            # Run the original get_conditioning_latents in executor
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                functools.partial(self.get_conditioning_latents,
                                  audio_path,
                                  max_ref_length,
                                  gpt_cond_len,
                                  gpt_cond_chunk_len,
                                  librosa_trim_db,
                                  sound_norm_refs,
                                  load_sr)
            ) # noqa
        return result

    async def get_model_logits(self, token_ids: List[int], conditioning: MultiModalDataDict) -> torch.Tensor:
        """Get model logits for a specific request"""
        request_id = uuid.uuid4().hex

        # Add start and end tokens
        token_ids = [self.mel_bos_token_id] + token_ids + [self.mel_eos_token_id] * 5

        engine_inputs = TokensPrompt(prompt_token_ids=token_ids)
        engine_inputs["multi_modal_data"] = conditioning

        # Bind the collector to this request
        bound_collector = self.hidden_states_collector.bind_to_request(request_id)

        # Set up sampling parameters with the bound collector
        sampling_params = ExtendedSamplingParams(
            detokenize=False,
            max_tokens=1,
            hidden_state_collector=bound_collector,
        )

        # Generate with unique request ID
        generator = self.llm_engine.generate(
            prompt=engine_inputs,
            sampling_params=sampling_params,
            request_id=request_id
        )

        # Consume the generator with a timeout
        try:
            async def consume_generator():
                async for _ in generator:
                    pass

            await asyncio.wait_for(consume_generator(), timeout=300)
        except asyncio.TimeoutError:
            raise RuntimeError("Timeout while generating logits")

        # Get the collected hidden states
        hidden_states = self.hidden_states_collector.get_hidden_states(request_id)

        if hidden_states is None:
            raise RuntimeError(f"No hidden states collected for request {request_id}")

        return hidden_states[-len(token_ids):, ...].unsqueeze(0).to(self.device).to(self.dtype)


    async def process_tokens_to_speech(
            self,
            generators: List[AsyncGenerator[RequestOutput, None]],
            speaker_embeddings: torch.Tensor,
            multimodal_data: List[torch.Tensor],
            chunk_size: int = 20,
    ) -> AsyncGenerator[TTSOutput, None]:
        """
        Process multiple token generators concurrently and emit results sequentially.
        Uses a queue-based approach to handle multiple generators reliably.
        """
        # Create a queue for each generator to store its results
        queues = [asyncio.Queue() for _ in generators]

        # Create tasks for processing each generator
        tasks = []
        for i, generator in enumerate(generators):
            task = asyncio.create_task(
                self._process_single_generator(
                    generator,
                    queues[i],
                    speaker_embeddings,
                    multimodal_data[i],
                    chunk_size
                )
            )
            tasks.append(task)

        try:
            # Process queues in sequence
            for i, queue in enumerate(queues):
                while True:
                    result = await queue.get()
                    if result is None:
                        # This generator has finished
                        break
                    else:
                        yield result

        finally:
            # Ensure all tasks are properly cleaned up
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_single_generator(
            self,
            generator: AsyncGenerator[RequestOutput, None],
            queue: asyncio.Queue,
            speaker_embeddings: torch.Tensor,
            gpt_embed_input: torch.Tensor,
            chunk_size: int
    ) -> None:
        """Process a single generator and put results in its queue."""
        try:
            last_decoded_token = 0
            accumulated_tokens = []

            async for output in generator:
                # Get new tokens
                new_tokens = output.outputs[0].token_ids[last_decoded_token:]
                accumulated_tokens.extend(new_tokens)
                last_decoded_token = len(accumulated_tokens)

                # Process tokens when we have enough or it's the final output
                if output.finished:# or len(accumulated_tokens) >= chunk_size: se lascio con acculated token mi ripete gli stesis toke, why??
                    # Process the accumulated tokens
                    hidden_states = await self.get_model_logits(
                        accumulated_tokens,
                        {
                            "audio": {
                                'embeds': gpt_embed_input,
                                "is_logits_only_mode": True
                            }
                        }
                    )

                    # Generate audio segment
                    wav = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        lambda: self.hifigan_decoder.inference(
                            hidden_states,
                            g=speaker_embeddings
                        ).cpu().numpy().squeeze()
                    ) # noqa

                    # Put result in queue
                    await queue.put(TTSOutput(
                        wav=wav
                    ))

                    # Reset accumulated tokens
                    accumulated_tokens = []

                if output.finished:
                    break

        except Exception as e:
            logging.error(f"Error in generator processing: {e}")
        finally:
            # Signal completion
            await queue.put(None)

    async def generate_speech_async_from_streaming_source(self, request: TTSRequest) -> AsyncGenerator[TTSOutput, None]:
        """Generate speech for streaming source of text, making a streaming source of audio tokens and then decoding
        and returning a streaming audio response."""
        assert isinstance(request.text, AsyncGenerator), "Text must be an AsyncGenerator for streaming source."
        # Prepare input with conditioning
        gpt_cond_latent, speaker_embeddings = await self.get_conditioning_latents_async(
            request.speaker_file,
            request.max_ref_length,
            request.gpt_cond_len,
            request.gpt_cond_chunk_len
        )
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            detokenize=False,
            top_k=request.top_k,
            logits_processors=[LogitsRepetitionPenalizer(request.repetition_penalty)],
            repetition_penalty=1.0,  # Since we're handling repetition penalty manually
            max_tokens=self.gpt_config.gpt_max_audio_tokens,
            ignore_eos=True,  # Ignore the tokenizer eos token since it is for textual generation
            stop_token_ids=[self.mel_eos_token_id],
        )

        accumulated_text = ""
        async for text in request.text:
            text = text.strip()
            accumulated_text += text

            if len(accumulated_text) > request.generate_every_n_chars:
                tokens, embeddings = await self.prepare_text_tokens_async(accumulated_text, request.language)
                gpt_embed_input = [torch.cat([gpt_cond_latent, embeddings[0]], dim=0)]

                engine_inputs = TokensPrompt(prompt_token_ids=tokens)
                if gpt_embed_input is not None:
                    engine_inputs["multi_modal_data"] = {"audio": {"embeds": gpt_embed_input, "is_logits_only_mode": False}}
                token_generator = [self.llm_engine.generate(
                    prompt=engine_inputs,
                    sampling_params=sampling_params,
                    request_id=request.request_id,
                )]
                # Process tokens to speech
                async for output in self.process_tokens_to_speech(
                        token_generator,
                        speaker_embeddings,
                        gpt_embed_input,
                        chunk_size=50
                ):
                    yield output

                accumulated_text = ""

    async def generate_speech_from_text_async(self, request: TTSRequest) -> AsyncGenerator[TTSOutput, None]:
        """Generate speech for a single request asynchronously."""
        # Prepare input with conditioning
        tokens_list, gpt_embed_inputs, speaker_embeddings = await self.prepare_inputs_async(
            request.text,
            request.language,
            request.speaker_file,
            request.max_ref_length,
            request.gpt_cond_len,
            request.gpt_cond_chunk_len,
            split_text=True  # Split text to avoid OOM on big texts
        )

        # Start all requests in parallel
        generators = []
        for seq_index, sequence in enumerate(tokens_list):
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                detokenize=False,
                top_k=request.top_k,
                logits_processors=[LogitsRepetitionPenalizer(request.repetition_penalty)],
                repetition_penalty=1.0,  # Since we're handling repetition penalty manually
                max_tokens=self.gpt_config.gpt_max_audio_tokens,
                ignore_eos=True,  # Ignore the tokenizer eos token since it is for textual generation
                stop_token_ids=[self.mel_eos_token_id],
            )

            engine_inputs = TokensPrompt(prompt_token_ids=sequence)
            if gpt_embed_inputs is not None:
                engine_inputs["multi_modal_data"] = {"audio": {"embeds": gpt_embed_inputs[seq_index], "is_logits_only_mode": False}}

            # Get audio token generator from VLLM
            token_generator = self.llm_engine.generate(
                prompt=engine_inputs,
                sampling_params=sampling_params,
                request_id=f"{request.request_id}_{seq_index}",
            )
            generators.append(token_generator)

        # Process tokens to speech
        async for output in self.process_tokens_to_speech(
                generators,
                speaker_embeddings,
                gpt_embed_inputs,
                chunk_size=50
        ):
            yield output

    def generate_speech_from_text(self, request: TTSRequest) -> List[TTSOutput]:
        """
        Synchronous wrapper for generate_speech_from_text_async.

        Args:
            request: XTTSRequest object containing generation parameters

        Returns:
            List of XTTSOutput containing the generated speech segments
        """

        async def _collect_outputs():
            outputs = []
            async for output in self.generate_speech_from_text_async(request):
                outputs.append(output)
            return outputs

        # Run the async code in an event loop
        import asyncio

        # Get or create an event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # Create a new loop if the current one is running
            new_loop = asyncio.new_event_loop()
            results = new_loop.run_until_complete(_collect_outputs())
            new_loop.close()
        else:
            results = loop.run_until_complete(_collect_outputs())

        return results
