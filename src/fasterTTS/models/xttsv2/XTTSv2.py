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
from vllm.sampling_params import RequestOutputKind
from vllm.utils import Counter

from ..base import BaseAsyncTTSEngine, ConditioningConfig, TokenGeneratorsAndPossiblyConditioning
from ...common.logging.logger import setup_logger
from ...common.definitions.output import TTSOutput
from ...common.definitions.requests import TTSRequest
from ...common.utilities import wav_to_mel_cloning, load_audio

from .components.vllm_mm_gpt import LearnedPositionEmbeddings
from .config.tokenizer import XTTSTokenizerFast
from .config.xttsv2_config import XTTSConfig
from .config.xttsv2_gpt_config import XTTSGPTConfig

from .components.vllm.hidden_state_collector import HiddenStatesCollector
from .components.vllm.hijack import ExtendedSamplingParams, LogitsRepetitionPenalizer
from .components.tts.layers.xtts.hifigan_decoder import HifiDecoder
from .components.tts.layers.xtts.latent_encoder import ConditioningEncoder
from .components.tts.layers.xtts.perceiver_encoder import PerceiverResampler


class XTTSv2Engine(BaseAsyncTTSEngine):
    """Async XTTS model implementation using VLLM's AsyncEngine."""

    model_type: "xtts"

    def __init__(self,
                 hifi_config: XTTSConfig,
                 gpt_config: XTTSGPTConfig,
                 max_gb_for_vllm_model: int = 4,
                 tensor_parallel_size: int = 1,
                 **kwargs):
        super().__init__()

        self.logger = setup_logger(__file__)
        self.logger.info("Initializing XTTSv2Engine...")

        self.hifi_config = hifi_config
        self.gpt_config = gpt_config
        self.mel_bos_token_id = gpt_config.start_audio_token
        self.mel_eos_token_id = gpt_config.stop_audio_token
        self.tp = tensor_parallel_size
        self.tokenizer = XTTSTokenizerFast.from_pretrained("AstraMindAI/xtts2-gpt")
        self.request_counter = Counter()
        self.executor = ThreadPoolExecutor(max_workers=10)  # For CPU-bound tasks

        self.max_gb_for_vllm_model = max_gb_for_vllm_model

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

        # Semaphore for concurrency control of the encoding process
        self.max_concurrency = 10
        self.semaphore = asyncio.BoundedSemaphore(self.max_concurrency)

    @property
    def conditioning_config(self) -> ConditioningConfig:
        return ConditioningConfig(
            speaker_embeddings=True, # noqa
            gpt_like_decoder_conditioning=True # noqa ?? why it is giving a warning?
        )

    def half(self):
        self.logger.warning("Cannot call .half() on XTTSv2Engine. it will be ignored.")
        # We cannot permit downcasting since it will throw an error while padding
        return

    def to(self, *args, **kwargs):
        # Block downcasting
        dtype = kwargs.get('dtype', None)
        if dtype == torch.float16 or dtype == torch.bfloat16:
            self.logger.warning("Cannot cast to half precision. Ignoring the request.")
            kwargs['dtype'] = torch.float32
        elif len(args) > 0 and (args[0] == torch.float16 or args[0] == torch.bfloat16):
            self.logger.warning("Cannot cast to half precision. Ignoring the request.")
            args = list(args)
            args[0] = torch.float32
            args = tuple(args)
        return super().to(*args, **kwargs)

    def init_vllm_engine(self):
        """Initialize models with AsyncVLLMEngine."""
        engine_args = AsyncEngineArgs(
            model="AstraMindAI/xtts2-gpt",
            tensor_parallel_size=self.tp,
            dtype="auto",
            max_model_len=self.gpt_config.max_text_tokens + self.gpt_config.max_audio_tokens,
            gpu_memory_utilization=self.get_memory_percentage(self.max_gb_for_vllm_model * 1024 ** 3),
            trust_remote_code=True,
            enforce_eager=True,
            limit_mm_per_prompt={"audio": 1},
            max_num_batched_tokens=608*4,
        )
        self.logger.info(f"Initializing VLLM engine with args: {engine_args}")
        self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            torch_dtype: torch.dtype = torch.float32,
            device_map: Optional[str] = "auto",
            tensor_parallel_size: int = 1,
            max_gpu_memory_in_gb: int = 1 ,
            **kwargs,
    ) -> nn.Module:
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
            max_gpu_memory_in_gb=max_gpu_memory_in_gb,
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
        self.logger.debug(f"Preparing text tokens for text: {text}")
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
            text_tokens = self.tokenizer(text, lang=[language])['input_ids'][0]
            text_tokens = await elaborate_tokens(text_tokens)
            fake_tokens_for_audio_generation = [1] * len(text_tokens)
        return fake_tokens_for_audio_generation, await embed_tokens(text_tokens)


    async def prepare_inputs_async(self, text: str, language: str, speaker_file: List[Union[str, Path]],
                                   max_ref_length: int, gpt_cond_len: int, gpt_cond_chunk_len: int, split_text: bool) \
            -> Tuple[List[List[int]], List[torch.Tensor], torch.Tensor]:
        """Prepare input text with conditioning tokens. Return combined conditioning latents"""
        # Tokenize text based on the language
        text_tokens, text_embeddings = await self.prepare_text_tokens_async(text, language, split_text)

        # Load the speaker file and convert it to a tensor
        gpt_cond_latent, speaker_embeddings = await self.get_audio_conditioning(
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

    async def get_audio_conditioning(
            self,
            audio_reference: [str, Path],
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
                                  audio_reference,
                                  max_ref_length,
                                  gpt_cond_len,
                                  gpt_cond_chunk_len,
                                  librosa_trim_db,
                                  sound_norm_refs,
                                  load_sr)
            ) # noqa
        return result

    async def get_model_logits(
            self,
            token_ids: List[int],
            conditioning: MultiModalDataDict,
            max_retries: int = 3,
            retry_delay: float = 0.1
    ) -> torch.Tensor:
        """
        Get model logits for a request with retry logic for empty hidden states.

        Args:
            token_ids: Input token IDs
            conditioning: Conditioning data
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        attempts = 0
        while attempts < max_retries:
            try:
                request_id = uuid.uuid4().hex

                # Add start and end tokens
                token_ids = [self.mel_bos_token_id] + list(token_ids) + [self.mel_eos_token_id] * 4

                engine_inputs = TokensPrompt(prompt_token_ids=token_ids)
                engine_inputs["multi_modal_data"] = conditioning

                hidden_states_collector = HiddenStatesCollector()
                # Bind the collector to this request
                bound_collector = hidden_states_collector.bind_to_request(request_id)

                # Set up sampling parameters with the bound collector
                sampling_params = ExtendedSamplingParams(
                    detokenize=False,
                    max_tokens=1,
                    hidden_state_collector=bound_collector,
                    output_kind=RequestOutputKind.FINAL_ONLY
                )

                # Generate with unique request ID
                generator = self.llm_engine.generate(
                    prompt=engine_inputs,
                    sampling_params=sampling_params,
                    request_id=request_id
                )

                async for output in generator:  # consume the generator
                    if output.finished:
                        pass

                # Get the collected hidden states
                hidden_states = await hidden_states_collector.get_hidden_states(request_id)

                if hidden_states is None:
                    attempts += 1
                    if attempts < max_retries:
                        self.logger.warning(
                            f"No hidden states collected for request {request_id} (attempt {attempts}/{max_retries}). "
                            f"Retrying after {retry_delay}s..."
                        )
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        raise RuntimeError(
                            f"No hidden states collected for request {request_id} after {max_retries} attempts. "
                            f"This should never happen! Please report this issue on GitHub."
                        )

                # Successfully got hidden states
                return hidden_states[-len(token_ids):-4, ...].unsqueeze(0).to(self.device).to(self.dtype)

            except Exception as e:
                attempts += 1
                if attempts < max_retries:
                    self.logger.warning(
                        f"Error getting hidden states (attempt {attempts}/{max_retries}): {str(e)}. "
                        f"Retrying after {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    self.logger.critical(f"Failed to get hidden states after {max_retries} attempts")
                    raise


    async def get_generation_context(self,
                                     request: TTSRequest,
                                     ) -> TokenGeneratorsAndPossiblyConditioning:
        # Prepare input with conditioning
        tokens_list, gpt_embed_inputs, speaker_embeddings = await self.prepare_inputs_async(
            request.text,
            request.language,
            request.speaker_files,
            request.max_ref_length,
            request.gpt_cond_len,
            request.gpt_cond_chunk_len,
            split_text=True  # Split text to avoid OOM on big texts
        )

        # Start all requests in parallel
        generators = []
        requests_id = []
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
                output_kind=RequestOutputKind.FINAL_ONLY
            )

            engine_inputs = TokensPrompt(prompt_token_ids=sequence)
            if gpt_embed_inputs is not None:
                engine_inputs["multi_modal_data"] = {
                    "audio": {"embeds": gpt_embed_inputs[seq_index], "is_logits_only_mode": False}}
            request_id =f"{request.request_id}_{seq_index}"
            # Get audio token generator from VLLM
            token_generator = self.llm_engine.generate(
                prompt=engine_inputs,
                sampling_params=sampling_params,
                request_id=request_id,
            )
            generators.append(token_generator)
            requests_id.append(request_id)

        return generators, requests_id, speaker_embeddings, gpt_embed_inputs


    async def process_tokens_to_speech(
        self,
        generator: AsyncGenerator[RequestOutput, None],
        speaker_embeddings: Optional[torch.Tensor] = None,
        multimodal_data: Optional[torch.Tensor] = None,
    ) -> AsyncGenerator[TTSOutput, None]:
        """
        Process a single token generator and emit results.
        """

        assert speaker_embeddings is not None, "Speaker embeddings must be provided for speech generation with XTTSv2."
        assert multimodal_data is not None, "Multimodal data must be provided for speech generation with XTTSv2."

        try:
            async for output in generator:
                if output.finished:
                    # get the hidden states
                    hidden_states = await self.get_model_logits(
                        list(output.outputs[0].token_ids),
                        {
                            "audio": {
                                'embeds': multimodal_data,  # Use multimodal data for conditioning
                                "is_logits_only_mode": True
                            }
                        }
                    )

                    # Genera il segmento audio
                    wav = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        lambda: self.hifigan_decoder.inference(
                            hidden_states,
                            g=speaker_embeddings
                        ).cpu().numpy().squeeze()
                    )  # noqa

                    # Yield il risultato direttamente
                    yield TTSOutput(wav=wav)

        except Exception as e:
            logging.error(f"Error in generator processing: {e}")
            raise # Re-raise the exception


