import asyncio
import functools
import hashlib
import os
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager

from pathlib import Path
from typing import Optional, List, Tuple, Union, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

import librosa
import numpy as np
import torch
import torchaudio
from torch import nn

# XTTSv2's GPT model is implemented as a custom V0-compatible model
# (vllm 0.9.2's V1 engine has no support for `prompt_embeds`, which the
# pre-computed conditioning path relies on). Force V0 before any vllm import.
os.environ.setdefault("VLLM_USE_V1", "0")

from vllm import AsyncLLMEngine, AsyncEngineArgs, RequestOutput, SamplingParams
from vllm.inputs.data import EmbedsPrompt
from vllm.sampling_params import RequestOutputKind
from vllm.utils import Counter

from ..base import (
    BaseAsyncTTSEngine,
    ConditioningConfig,
    TokenGeneratorsAndPossiblyConditioning,
)
from ...common.logging.logger import setup_logger
from ...common.definitions.output import TTSOutput
from ...common.definitions.requests import TTSRequest
from ...common.utilities import wav_to_mel_cloning, load_audio

from .components.vllm_mm_gpt import LearnedPositionEmbeddings, XttsGPT
from .config.tokenizer import XTTSTokenizerFast
from .config.xttsv2_config import XTTSConfig
from .config.xttsv2_gpt_config import XTTSGPTConfig

from .components.vllm.hijack import (
    LogitsLengthPenalizer,
    LogitsRepetitionPenalizer,
)
from .components.tts.layers.xtts.hifigan_decoder import HifiDecoder
from .components.tts.layers.xtts.latent_encoder import ConditioningEncoder
from .components.tts.layers.xtts.perceiver_encoder import PerceiverResampler


_VLLM_MEMORY_ASSERT_PATCHED = False


def _patch_vllm_memory_profile_assert():
    """Neutralise vLLM's "profile must allocate memory" assertion.

    The check at `vllm.worker.worker.Worker._assert_memory_footprint_increased
    _during_profiling` raises an `AssertionError` if the post-profile GPU
    snapshot is not strictly greater than the pre-profile baseline. For the
    XTTS GPT (a tiny ~30M-param model used with `max_num_seqs=1`) the profile
    forward frees more caching-allocator slack than it allocates, so the
    assertion produces false positives. We replace it with a no-op once per
    process; the rest of the profiling logic (KV cache block count discovery)
    still runs and stays accurate.
    """
    global _VLLM_MEMORY_ASSERT_PATCHED
    if _VLLM_MEMORY_ASSERT_PATCHED:
        return
    try:
        from vllm.worker import worker as _vllm_worker
        _vllm_worker.Worker._assert_memory_footprint_increased_during_profiling = (
            lambda self: None)
        _VLLM_MEMORY_ASSERT_PATCHED = True
    except Exception:
        # Best-effort; if the symbol moves in a future vLLM release the
        # original assertion will still fire and produce a clear traceback.
        pass

_CONDITIONING_CACHE_DIGEST_SIZE = 16


class XTTSv2Engine(BaseAsyncTTSEngine):
    """Asynchronous XTTS model implementation using VLLM's AsyncEngine.

    This class implements a high-performance text-to-speech engine based on the XTTS v2 architecture.
    It uses VLLM for efficient token generation and supports both speaker conditioning and
    GPT-like decoder conditioning for enhanced voice control. The implementation is optimized
    for inference speed through parallel processing and efficient memory management.

    Attributes:
        model_type (str): The model type identifier, set to "xtts".
    """

    model_type: "xtts"

    def __init__(
        self,
        hifi_config: XTTSConfig,
        gpt_config: XTTSGPTConfig,
        pipeline_parallel_size: int = 1,
        tensor_parallel_size: int = 1,
        **kwargs,
    ):
        """Initialize the XTTS v2 engine.

        Args:
            hifi_config (XTTSConfig): Configuration for the HiFi-GAN decoder.
            gpt_config (XTTSGPTConfig): Configuration for the GPT model.
            pipeline_parallel_size (int, optional): Number of pipeline parallel partitions. Defaults to 1.
            tensor_parallel_size (int, optional): Number of tensor parallel partitions. Defaults to 1.
            **kwargs: Additional arguments including:
                - gpt_model: Path to the GPT model
                - max_concurrency: Maximum number of concurrent requests
        """
        super().__init__()

        self.max_gb_for_vllm_model = None

        self.logger = setup_logger(__file__)
        self.logger.info("Initializing XTTSv2Engine...")

        self.gpt_model = kwargs.pop("gpt_model")
        self.device_map = kwargs.pop("device_map", "auto")
        # Increased from 0.35 to 0.5 for better GPU utilization on modern GPUs
        # This allows more concurrent requests while still leaving headroom for decoder
        self.gpu_memory_utilization = kwargs.pop("gpu_memory_utilization", 0.5)
        # ``cpu_offload_gb`` is the budget for vLLM's ``maybe_offload_to_cpu``
        # helper (see vllm.model_executor.models.utils.make_layers, which
        # wraps every transformer block with that helper). Anything > 0
        # tells vLLM "offload up to N GB of layer weights to CPU and replace
        # each layer's forward with a per-call ``functional_call`` that
        # rehydrates the params on GPU per step". That path is intended for
        # huge dense LLMs that do not fit on the GPU at all. The XTTS GPT
        # decoder is ~0.76 GB in bfloat16 so the prior default of 8.0 GB
        # silently offloaded ALL transformer blocks to CPU, and the
        # functional_call shim does not preserve the vLLM ``Attention``
        # layer's KV cache hookup correctly on V0 — generation runs but
        # the per-step KV writes go to the CPU shadow of the params, so
        # subsequent decode steps consume stale K/V and the audio output
        # decorrelates from the text prompt. Default to 0.0 so the
        # transformer blocks stay on GPU; operators with genuinely huge
        # models can opt back in explicitly.
        self.cpu_offload_gb = kwargs.pop("cpu_offload_gb", 0.0)
        self.swap_space = kwargs.pop("swap_space", 2.0)
        speaker_embedding_cache_size = kwargs.pop("speaker_embedding_cache_size", 100)
        try:
            self.speaker_embedding_cache_size = max(
                0, int(speaker_embedding_cache_size)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "speaker_embedding_cache_size must be convertible to a non-negative integer"
            ) from exc
        self.hifi_config = hifi_config
        self.gpt_config = gpt_config
        self.mel_bos_token_id = gpt_config.start_audio_token
        self.mel_eos_token_id = gpt_config.stop_audio_token
        self.tp = tensor_parallel_size
        self.pp = pipeline_parallel_size
        self.tokenizer = XTTSTokenizerFast.from_pretrained(self.gpt_model)
        self.request_counter = Counter()

        requested_concurrency = kwargs.pop("max_concurrency", 1)
        self.is_cpu = self.device_map == "cpu" or (
            self.device_map == "auto" and not torch.cuda.is_available()
        )
        self.max_concurrency = 1 if self.is_cpu else max(1, requested_concurrency)
        semaphore_concurrency = (
            1 if self.is_cpu else max(1, self.max_concurrency // 6) * self.tp
        )

        # Register buffer before creating modules
        self.register_buffer("mel_stats", torch.ones(80))

        # Cache for speaker embeddings to avoid redundant I/O and computation
        self._speaker_embedding_cache = OrderedDict()

        # Initialize all nn.Module components
        self.conditioning_encoder = ConditioningEncoder(
            gpt_config.audio_config.mel_channels,
            gpt_config.hidden_size,
            num_attn_heads=gpt_config.num_attention_heads,
        )

        self.text_embedding = nn.Embedding(
            gpt_config.number_text_tokens, gpt_config.hidden_size
        )

        self.text_pos_embedding = (
            LearnedPositionEmbeddings(
                gpt_config.max_text_tokens + 2,
                gpt_config.hidden_size,
                supports_pp=False,
            )
            if gpt_config.max_audio_tokens != -1
            else functools.partial(
                gpt_config.null_position_embeddings, dim=gpt_config.hidden_size
            )
        )

        # Keep SDP attention enabled on all CUDA devices; the attention module
        # selects flash kernels internally when the active GPU supports them.
        use_flash_attn = not self.is_cpu

        self.conditioning_perceiver = PerceiverResampler(
            dim=gpt_config.hidden_size,
            depth=2,
            dim_context=gpt_config.hidden_size,
            num_latents=32,
            dim_head=64,
            heads=8,
            ff_mult=4,
            use_flash_attn=use_flash_attn,
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

        self.final_norm = nn.LayerNorm(gpt_config.hidden_size, eps=1e-5, bias=True)

        # Kept for model loading purposes
        self.text_head = nn.Linear(
            gpt_config.hidden_size, gpt_config.number_text_tokens, bias=True
        )

        if not self.is_cpu:
            self.get_memory_usage_curve()
            # vLLM's memory profiler asserts that the GPU footprint grows
            # monotonically during its dummy forward (vllm.worker.worker:305).
            # On tiny models like the XTTS GPT the profile run actually
            # *releases* caching-allocator slack accumulated during our
            # nn.Module construction, so the post-profile snapshot dips
            # below the baseline and trips the assertion. The assertion is a
            # heuristic safety check intended for large LLMs; for XTTS it
            # produces false positives, so we replace it with a warning.
            _patch_vllm_memory_profile_assert()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Initialize VLLM engine at the end, settings its concurrency
        self.init_vllm_engine(self.max_concurrency)

        # Semaphore for concurrency control of the encoding process
        self.encoder_semaphore = asyncio.BoundedSemaphore(semaphore_concurrency)
        self.decoder_semaphore = asyncio.BoundedSemaphore(
            semaphore_concurrency
        )  # empirically found a good value

        # LRU cache for speaker conditioning to avoid recomputation.
        # Key: normalized audio references + conditioning parameters.
        self._conditioning_cache = OrderedDict()
        self._conditioning_cache_lock = asyncio.Lock()
        self._max_cache_size = 100  # Limit cache size to prevent memory issues
        self._decode_counter_lock = asyncio.Lock()
        self._decode_counter = 0

        self.eval()

    def clear_speaker_embedding_cache(self):
        """Clear the speaker embedding cache to free memory.

        This method removes all cached speaker embeddings and the associated
        CPU audio tensors, freeing up cache memory. Useful for long-running services
        or when switching between different sets of speaker voices.
        """
        self._speaker_embedding_cache.clear()
        self.logger.info("Speaker embedding cache cleared")

    def get_memory_usage_curve(self):
        """Calculate the memory usage curve based on concurrency level.

        Uses empirically determined polynomial coefficients to estimate memory requirements
        for different concurrency levels. This helps in optimizing resource allocation
        for the VLLM engine.
        """
        # thanks to NinjaPerson24119
        amd = 2.0  # AMD GPUs are less VRAM efficient than NVIDIA GPUs

        x = np.array([2, 5, 10, 16])
        y = np.array([1.25 * amd, 1.35 * amd, 1.45 * amd, 1.625 * amd])

        # polynomial fit
        coefficients = np.polyfit(x, y, 2)

        # create a polynomial object
        self.max_gb_for_vllm_model = (
            coefficients[0] * self.max_concurrency**2
            + coefficients[1] * self.max_concurrency
            + coefficients[2]
        )

    @property
    def conditioning_config(self) -> ConditioningConfig:
        return ConditioningConfig(
            speaker_embeddings=True,  # noqa
            gpt_like_decoder_conditioning=True,  # noqa
        )

    def half(self):
        self.logger.warning("Cannot call .half() on XTTSv2Engine. it will be ignored.")
        # We cannot permit downcasting since it will throw an error while padding
        return

    def to(self, *args, **kwargs):
        # Block downcasting
        dtype = kwargs.get("dtype", None)
        if dtype == torch.float16 or dtype == torch.bfloat16:
            self.logger.warning("Cannot cast to half precision. Ignoring the request.")
            kwargs["dtype"] = torch.float32
        elif len(args) > 0 and (args[0] == torch.float16 or args[0] == torch.bfloat16):
            self.logger.warning("Cannot cast to half precision. Ignoring the request.")
            args = list(args)
            args[0] = torch.float32
            args = tuple(args)
        return super().to(*args, **kwargs)

    def init_vllm_engine(self, concurrency):
        """Initialize the VLLM engine with specified concurrency.

        Args:
            concurrency (int): Maximum number of concurrent requests to handle.

        Raises:
            RuntimeError: If unable to determine memory usage for model initialization.
        """
        """Initialize models with AsyncVLLMEngine."""
        # XttsGPT now tracks per-request prefill lengths via the
        # ``HasInnerState`` hook (request_ids_to_seq_ids is forwarded into
        # ``forward``), so vLLM can co-batch concurrent requests safely. CPU
        # inference is still restricted to one sequence to avoid runaway
        # host memory use.
        max_seq_num = 1 if self.is_cpu else max(1, concurrency)
        max_model_len = (
            self.gpt_config.max_text_tokens
            + self.gpt_config.max_audio_tokens
            + 32
            + 5
            + 3
        )
        engine_kwargs = dict(
            model=self.gpt_model,
            tensor_parallel_size=1 if self.is_cpu else self.tp,
            pipeline_parallel_size=1 if self.is_cpu else self.pp,
            dtype="float32" if self.is_cpu else "auto",
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=True,
            enable_prompt_embeds=True,
            max_num_seqs=max_seq_num,
            disable_log_stats=True,
            max_num_batched_tokens=max_model_len * max_seq_num,
        )

        if self.is_cpu:
            engine_kwargs["swap_space"] = self.swap_space
        else:
            mem_utilization = self.gpu_memory_utilization
            if mem_utilization is None:
                mem_utilization = self.get_memory_percentage(
                    self.max_gb_for_vllm_model * 1024**3
                )
            if not mem_utilization:
                raise RuntimeError(
                    "Could not determine GPU memory utilization for vLLM initialization."
                )
            engine_kwargs["gpu_memory_utilization"] = mem_utilization
            engine_kwargs["cpu_offload_gb"] = self.cpu_offload_gb
            engine_kwargs["swap_space"] = self.swap_space

        engine_args = AsyncEngineArgs(
            **engine_kwargs,
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
        pipeline_parallel_size: int = 1,
        **kwargs,
    ) -> nn.Module:
        """Load a pretrained XTTS model from local path or Hugging Face Hub.

        Args:
            pretrained_model_name_or_path (str): Path to pretrained model or HF model identifier.
            torch_dtype (torch.dtype, optional): Model data type. Defaults to torch.float32.
            device_map (Optional[str], optional): Device mapping strategy. Defaults to "auto".
            tensor_parallel_size (int, optional): Number of tensor parallel partitions. Defaults to 1.
            pipeline_parallel_size (int, optional): Number of pipeline parallel partitions. Defaults to 1.
            **kwargs: Additional arguments passed to the model constructor.

        Returns:
            nn.Module: Loaded XTTS model instance.
        """
        from huggingface_hub import hf_hub_download
        import json
        import os

        # Download and load configs
        if not os.path.exists(pretrained_model_name_or_path):
            config_file = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename="config.json"
            )
            with open(config_file, "r") as f:
                config = json.load(f)

        else:
            # Load from local path
            with open(
                os.path.join(pretrained_model_name_or_path, "config.json"), "r"
            ) as f:
                config = json.load(f)

        # Initialize configs
        gpt_config = XTTSGPTConfig(**config["gpt_config"])
        hifi_config = XTTSConfig(**config)
        kwargs["device_map"] = device_map

        # Initialize model
        model = cls(
            hifi_config=hifi_config,
            gpt_config=gpt_config,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            **kwargs,
        )

        # Load model weights
        if not os.path.exists(pretrained_model_name_or_path):
            hifigan_weights = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename="xtts-v2.safetensors"
            )
        else:
            hifigan_weights = os.path.join(
                pretrained_model_name_or_path, "xtts-v2.safetensors"
            )

        # Load HiFi-GAN weights. We avoid `safetensors.torch.load_model`
        # because newer safetensors (>=0.5) rejects state dicts that contain
        # buffers sharing storage with another tensor (e.g.
        # `hifigan_decoder.speaker_encoder.torch_spec.1.spectrogram.window`
        # is a view created at construction time). Loading the raw state
        # dict and calling `load_state_dict` directly is safe — we know the
        # tensors are unique on disk and only become views in memory after
        # being assigned back into the module.
        from safetensors.torch import load_file

        state_dict = load_file(hifigan_weights, device="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict, strict=True)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "Weight mismatch loading XTTSv2 checkpoint: "
                f"missing={missing_keys}, unexpected={unexpected_keys}")

        # Set model properties
        model.config = config

        target_device = device_map
        if target_device in (None, "auto"):
            target_device = "cuda" if torch.cuda.is_available() else "cpu"

        # Cast model to specified dtype
        model = model.to(torch_dtype)
        model = model.to(target_device)

        # On Ampere and newer GPUs (SM >= 8.0) explicitly allow TF32 to maximise
        # throughput on matmul and convolution operations.  PyTorch enables this
        # by default since 1.7/1.12, but we set it explicitly so user overrides
        # in the environment cannot inadvertently disable it.
        if torch.cuda.is_available() and str(target_device).startswith("cuda"):
            prop = torch.cuda.get_device_properties(torch.device(target_device))
            if prop.major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        return model

    async def _get_speaker_embedding(self, audio, sr):
        """Extract speaker embedding from audio.

        Args:
            audio: Input audio tensor.
            sr: Sampling rate of the audio.

        Returns:
            torch.Tensor: Speaker embedding tensor.
        """
        audio_16k = torchaudio.functional.resample(audio, sr, 16000)
        async with self.decoder_semaphore:
            return (
                self.hifigan_decoder.speaker_encoder.forward(
                    audio_16k.to(self.device), l2_norm=True
                )
                .unsqueeze(-1)
                .to(self.device)
            )

    async def _merge_conditioning(
        self, text_conditioning: List[torch.Tensor], audio_conditioning: torch.Tensor
    ) -> List[torch.Tensor]:
        """Merge text and audio conditioning signals.

        Returns one ``[L_cond + L_text, hidden]`` tensor per text chunk. Text
        positional embeddings are already baked into ``text_conditioning``
        by :meth:`prepare_text_tokens_async` (see ``text_pos_embedding`` usage
        there); audio conditioning latents are perceiver outputs and carry no
        positional structure, so the resulting tensor is ready to be used as
        the leading portion of a vLLM :class:`EmbedsPrompt`.
        """
        target_dtype = self._vllm_model_dtype()
        cond_latents = []
        for text_embedding in text_conditioning:
            # Concatenate along sequence dimension
            cond_latents.append(
                torch.cat([audio_conditioning, text_embedding], dim=1)
                .squeeze(0)
                .to(target_dtype)
            )
        return cond_latents

    def _vllm_model_dtype(self) -> torch.dtype:
        """Return the dtype of the underlying vLLM-loaded GPT weights.

        vLLM 0.9.2's `AsyncLLMEngine` no longer exposes the V0 `.engine`
        attribute directly; we go through `engine.model_config.dtype` instead.
        """
        if hasattr(self.llm_engine, "engine"):
            mc = self.llm_engine.engine.model_config
        else:
            mc = self.llm_engine.model_config
        return mc.dtype

    def _get_loaded_xtts_gpt(self) -> nn.Module:
        """Reach into the vLLM engine for the loaded XttsGPT module.

        Used to look up the embedding tables (`wte`, `wpe`) when constructing
        prompt embeddings for either the autoregressive generation pass or
        the logits-only pass. The traversal follows V0 single-process layout:
        AsyncLLMEngine -> LLMEngine -> model_executor -> driver_worker ->
        model_runner -> model.
        """
        eng = self.llm_engine.engine if hasattr(self.llm_engine,
                                                "engine") else self.llm_engine
        executor = eng.model_executor
        # `driver_worker` exists for single-process executors (the typical
        # path when tensor_parallel_size == 1 == pipeline_parallel_size).
        worker = getattr(executor, "driver_worker", None)
        if worker is None:
            # Multi-worker path: fall back to the first collective worker.
            worker = executor.workers[0]
        runner = worker.model_runner
        return runner.model

    def _build_generation_prompt_embeds(
        self,
        merged_conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """Append ``wte(start_audio_token) + wpe(audio_pos=0)`` to the merged
        conditioning so vLLM can start autoregressive audio token generation.

        The XTTS GPT was trained with the start-of-audio token sitting at
        audio position 0 (immediately after the conditioning + text prefix),
        which is why ``XttsGPT._active_prefill_len`` includes this token.
        """
        gpt = self._get_loaded_xtts_gpt().gpt  # _XttsTransformer holds wte/wpe
        device = merged_conditioning.device
        dtype = merged_conditioning.dtype

        start_token_id = torch.tensor([self.mel_bos_token_id], device=device)
        start_embed = gpt.wte(start_token_id).to(dtype)
        # `LearnedPositionEmbeddings.get_fixed_embedding` returns a
        # [B, 1, D] tensor for scalar inputs; squeeze back to [1, D].
        zero_pos = torch.zeros(1, dtype=torch.long, device=device)
        start_pos_embed = gpt.wpe.get_fixed_embedding(
            zero_pos, device).view(1, -1).to(dtype)
        start_with_pos = start_embed + start_pos_embed

        return torch.cat([merged_conditioning, start_with_pos], dim=0)

    def _build_logits_prompt_embeds(
        self,
        merged_conditioning: torch.Tensor,
        token_ids: List[int],
    ) -> torch.Tensor:
        """Build a prefill-only EmbedsPrompt for the hidden-states extraction
        path.

        Layout (mirrors the original V0 logits-only flow):
            [cond_latents, text_embeds, wte(mel_bos)+wpe(0),
             wte(t_0)+wpe(1), ..., wte(t_{N-1})+wpe(N),
             wte(mel_eos)+wpe(N+1), ..., wte(mel_eos)+wpe(N+4)]

        `token_ids` is the autoregressive output from
        :meth:`get_generation_context`, so the resulting prefill captures the
        hidden state for every position the HiFiGAN decoder needs.
        """
        gpt = self._get_loaded_xtts_gpt().gpt  # _XttsTransformer holds wte/wpe
        device = merged_conditioning.device
        dtype = merged_conditioning.dtype

        full_ids = (
            [self.mel_bos_token_id]
            + list(token_ids)
            + [self.mel_eos_token_id] * 4
        )
        ids_tensor = torch.tensor(full_ids, dtype=torch.long, device=device)
        token_embeds = gpt.wte(ids_tensor).to(dtype)

        audio_positions = torch.arange(
            len(full_ids), dtype=torch.long, device=device)
        # When generation reaches close to ``gpt_max_audio_tokens`` the
        # trailing ``[mel_bos | tokens | mel_eos*4]`` positions can exceed
        # the learned wpe table (sized ``max_audio_tokens + 3``). Those
        # trailing positions are sliced off in ``get_model_logits`` via
        # ``[start_of_audio_hs:-5]``, so a clamped wpe lookup for them is
        # safe — the hidden states they contribute are discarded.
        audio_positions = torch.clamp(audio_positions, max=gpt.wpe.seq_len - 1)
        pos_embeds = gpt.wpe.get_fixed_embedding(
            audio_positions, device).view(token_embeds.shape).to(dtype)

        audio_part = token_embeds + pos_embeds
        return torch.cat([merged_conditioning, audio_part], dim=0)

    def get_gpt_cond_latents(self, audio, sr, length: int = 30, chunk_length: int = 6):
        """Generate GPT conditioning latents from audio.

        Args:
            audio: Input audio tensor.
            sr: Sampling rate of the audio.
            length (int, optional): Maximum reference length in seconds. Defaults to 30.
            chunk_length (int, optional): Length of each conditioning chunk. Defaults to 6.

        Returns:
            torch.Tensor: GPT conditioning latents.
        """
        if sr != 22050:
            audio = torchaudio.functional.resample(audio, sr, 22050)
        if length > 0:
            audio = audio[:, : 22050 * length]
        if self.gpt_config.use_perceiver_resampler:
            style_embs = []
            for i in range(0, audio.shape[1], 22050 * chunk_length):
                audio_chunk = audio[:, i : i + 22050 * chunk_length]

                # if the chunk is too short ignore it
                if audio_chunk.size(-1) < 22050 * 0.33:
                    continue

                mel_chunk = wav_to_mel_cloning(
                    audio_chunk,
                    mel_norms=self.mel_stats,
                    device=self.device,
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
                style_emb = self.get_style_emb(mel_chunk, None)
                style_embs.append(style_emb)

            # mean style embedding
            cond_latent = torch.stack(style_embs).mean(dim=0)
        else:
            mel = wav_to_mel_cloning(
                audio,
                mel_norms=self.mel_stats,
                device=self.device,
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
            cond_latent = self.get_style_emb(mel)
        return cond_latent.transpose(1, 2)

    async def get_conditioning_latents(
        self,
        audio_reference,
        max_ref_length=30,
        gpt_cond_len=6,
        gpt_cond_chunk_len=6,
        librosa_trim_db=None,
        sound_norm_refs=False,
        load_sr=22050,
    ):
        """Generate conditioning latents from reference audio.

        Args:
            audio_reference: Reference audio file path or tensor.
            max_ref_length (int, optional): Maximum reference length in seconds. Defaults to 30.
            gpt_cond_len (int, optional): Length of GPT conditioning. Defaults to 6.
            gpt_cond_chunk_len (int, optional): Length of each conditioning chunk. Defaults to 6.
            librosa_trim_db (float, optional): Trim silence below this dB threshold.
            sound_norm_refs (bool, optional): Whether to normalize reference audio. Defaults to False.
            load_sr (int, optional): Sampling rate for loading audio. Defaults to 22050.

        Returns:
            Tuple: GPT conditioning latents and speaker embeddings.
        """
        # Deal with multiple references
        assert (
            isinstance(audio_reference, bytes)
            or isinstance(audio_reference, str)
            or isinstance(audio_reference, list)
        ), (
            f"audio_reference must be a string, byte or a list but it is {type(audio_reference)}"
        )

        if not isinstance(audio_reference, list):
            audio_paths = [audio_reference]
        else:
            audio_paths = audio_reference

        speaker_embeddings = []
        audios = []
        for file_path in audio_paths:
            # Generate cache key from file path and parameters
            cache_key = (
                str(file_path),
                load_sr,
                max_ref_length,
                sound_norm_refs,
                librosa_trim_db,
            )

            # Check cache first
            if cache_key in self._speaker_embedding_cache:
                cached_data = self._speaker_embedding_cache[cache_key]
                self._speaker_embedding_cache.move_to_end(cache_key)
                speaker_embeddings.append(cached_data["speaker_embedding"])
                audios.append(cached_data["audio"])
                continue

            audio = load_audio(file_path, load_sr)
            audio = audio[:, : load_sr * max_ref_length].to(self.dtype)
            if sound_norm_refs:
                audio = (audio / torch.abs(audio).max()) * 0.75
            if librosa_trim_db is not None:
                audio = librosa.effects.trim(audio, top_db=librosa_trim_db)[0]

            # Compute latents for the decoder
            speaker_embedding = await self._get_speaker_embedding(audio, load_sr)
            speaker_embeddings.append(speaker_embedding)
            # Keep cached/reference audio on CPU until concatenation so repeated
            # speaker cache hits avoid pinning per-reference tensors on GPU.
            audios.append(audio)

            if self.speaker_embedding_cache_size > 0:
                self._speaker_embedding_cache[cache_key] = {
                    "speaker_embedding": speaker_embedding,
                    "audio": audio.cpu(),
                }
                self._speaker_embedding_cache.move_to_end(cache_key)
                if len(self._speaker_embedding_cache) > self.speaker_embedding_cache_size:
                    self._speaker_embedding_cache.popitem(last=False)

        # Merge all the audios and compute the latents for the GPT
        full_audio = torch.cat(audios, dim=-1).to(self.device)
        gpt_cond_latents = await asyncio.to_thread(
            self.get_gpt_cond_latents,
            full_audio,
            load_sr,
            length=gpt_cond_len,
            chunk_length=gpt_cond_chunk_len,
        )  # [1, 1024, T]

        speaker_embedding = torch.stack(speaker_embeddings)
        speaker_embedding = speaker_embedding.mean(dim=0)

        return gpt_cond_latents, speaker_embedding

    @asynccontextmanager
    async def cuda_memory_manager(self):
        """Context manager for CUDA memory management.

        Releases cached CUDA memory periodically (every 10 decoder calls) to avoid
        fragmentation without paying the cleanup overhead on every decode.
        ``torch.cuda.synchronize()`` and ``asyncio.sleep`` are intentionally omitted:
        the decoder runs inside ``asyncio.to_thread`` which already waits for the
        thread (and all GPU work launched within it) to complete before returning.

        Memory cleanup is performed periodically rather than on every call to reduce
        overhead while still preventing fragmentation during sustained load.
        """
        try:
            yield
        finally:
            if torch.cuda.is_available():
                # Only clear cache periodically to reduce overhead.
                async with self._decode_counter_lock:
                    self._decode_counter += 1

                    # Clear cache every 10 decoder calls to balance fragmentation vs overhead.
                    if self._decode_counter % 10 == 0:
                        torch.cuda.empty_cache()

    def get_style_emb(
        self, cond_input: torch.Tensor, return_latent: Optional[bool] = False
    ) -> torch.Tensor:
        """Extract style embedding from conditioning input.

        Args:
            cond_input (torch.Tensor): Conditioning input tensor.
            return_latent (Optional[bool], optional): Whether to return latent representation. Defaults to False.

        Returns:
            torch.Tensor: Style embedding tensor.
        """
        if not return_latent:
            if cond_input.ndim == 4:
                cond_input = cond_input.squeeze(1)
            conds = self.conditioning_encoder(cond_input)

            if hasattr(self, "conditioning_perceiver"):
                conds = self.conditioning_perceiver(conds.permute(0, 2, 1)).transpose(
                    1, 2
                )  # (b,d,32)
        else:
            conds = cond_input.unsqueeze(1)
        return conds

    async def prepare_text_tokens_async(
        self, text: str, language: str, split_text=False
    ) -> Tuple[List[Union[int, List[int]]], List[torch.Tensor]]:
        """Prepare text tokens and embeddings asynchronously.

        Args:
            text (str): Input text to tokenize.
            language (str): Language code.
            split_text (bool, optional): Whether to split text into chunks. Defaults to False.

        Returns:
            Tuple: Token IDs and text embeddings.
        """
        self.logger.debug(f"Preparing text tokens for text: {text}")

        async def elaborate_tokens(text_tokens: List[int]) -> torch.Tensor:
            text_tokens.insert(0, self.tokenizer.bos_token_id)
            text_tokens.append(self.tokenizer.eos_token_id)
            return (
                torch.tensor(text_tokens)
                .unsqueeze(0)
                .to(self.text_embedding.weight.device)
            )

        async def embed_tokens(
            text_tokens: Union[torch.Tensor, List[torch.Tensor]],
        ) -> List[torch.Tensor]:
            embeds = []
            if isinstance(text_tokens, list):
                for list_element in text_tokens:
                    embeds.append(
                        self.text_embedding(list_element)
                        + self.text_pos_embedding(list_element)
                    )
            else:
                embeds.append(
                    self.text_embedding(text_tokens)
                    + self.text_pos_embedding(text_tokens)
                )
            return embeds

        fake_tokens_for_audio_generation = []
        if split_text:
            text_tokens = self.tokenizer.batch_encode_with_split(text, lang=[language])
            for idx, text_token in enumerate(text_tokens):
                text_tokens[idx] = await elaborate_tokens(text_token)
                fake_tokens_for_audio_generation.append([1] * len(text_token))
        else:
            text_tokens = self.tokenizer(text, lang=[language])["input_ids"][0]
            text_tokens = await elaborate_tokens(text_tokens)
            fake_tokens_for_audio_generation = [1] * len(text_tokens)
        return fake_tokens_for_audio_generation, await embed_tokens(text_tokens)

    async def prepare_inputs_async(
        self,
        text: str,
        language: str,
        speaker_file: List[Union[str, Path]],
        max_ref_length: int,
        gpt_cond_len: int,
        gpt_cond_chunk_len: int,
        split_text: bool,
    ) -> Tuple[List[List[int]], List[torch.Tensor], torch.Tensor]:
        """Prepare all inputs for speech generation asynchronously.

        Args:
            text (str): Input text.
            language (str): Language code.
            speaker_file (List[Union[str, Path]]): List of speaker reference files.
            max_ref_length (int): Maximum reference length in seconds.
            gpt_cond_len (int): Length of GPT conditioning.
            gpt_cond_chunk_len (int): Length of each conditioning chunk.
            split_text (bool): Whether to split text into chunks.

        Returns:
            Tuple: Token IDs, text embeddings, and speaker embeddings.
        """
        # Tokenize text based on the language
        text_tokens, text_embeddings = await self.prepare_text_tokens_async(
            text, language, split_text
        )

        # Load the speaker file and convert it to a tensor
        gpt_cond_latent, speaker_embeddings = await self.get_audio_conditioning(
            speaker_file, max_ref_length, gpt_cond_len, gpt_cond_chunk_len
        )

        cond_latents = await self._merge_conditioning(text_embeddings, gpt_cond_latent)

        return text_tokens, cond_latents, speaker_embeddings

    @staticmethod
    def _normalize_audio_reference_for_cache(
        audio_reference: Union[str, Path, bytes]
    ) -> Tuple[str, str]:
        """Normalize a reference into a compact, type-tagged cache-key fragment.

        Paths and string references are stored as string values with a type prefix.
        Raw audio bytes are replaced with a short BLAKE2 digest so the cache does
        not retain large payloads in memory.
        """
        if isinstance(audio_reference, bytes):
            digest = hashlib.blake2b(
                audio_reference,
                digest_size=_CONDITIONING_CACHE_DIGEST_SIZE,
            ).hexdigest()
            return ("bytes", digest)
        if isinstance(audio_reference, Path):
            return ("path", str(audio_reference))
        return ("ref", str(audio_reference))

    def _get_conditioning_cache_key(
        self,
        audio_reference: Union[
            str,
            Path,
            bytes,
            List[Union[str, Path, bytes]],
            Tuple[Union[str, Path, bytes], ...],
        ],
        max_ref_length: int,
        gpt_cond_len: int,
        gpt_cond_chunk_len: int,
        librosa_trim_db: Optional[float],
        sound_norm_refs: bool,
        load_sr: int,
    ) -> Tuple[Tuple[Tuple[str, str], ...], int, int, int, Optional[float], bool, int]:
        """Build an order-preserving cache key for speaker conditioning inputs.

        Reference ordering is preserved because ``get_conditioning_latents`` uses
        the provided order when concatenating audio. The final cache key combines
        the normalized references with the conditioning parameters that affect the
        computed latents and speaker embedding.
        """
        references = (
            audio_reference
            if isinstance(audio_reference, (list, tuple))
            else (audio_reference,)
        )
        normalized_references = tuple(
            self._normalize_audio_reference_for_cache(reference)
            for reference in references
        )
        return (
            normalized_references,
            max_ref_length,
            gpt_cond_len,
            gpt_cond_chunk_len,
            librosa_trim_db,
            sound_norm_refs,
            load_sr,
        )

    async def _get_cached_conditioning_result(
        self,
        cache_key,
        cache_reference_key: Optional[Tuple[Tuple[str, str], ...]] = None,
        log_cache_hit: bool = False,
    ):
        """Return a cached conditioning tuple and refresh its LRU position."""
        async with self._conditioning_cache_lock:
            cached_result = self._conditioning_cache.get(cache_key)
            if cached_result is None:
                return None
            self._conditioning_cache.move_to_end(cache_key)

        if log_cache_hit:
            self.logger.debug(
                f"Using cached conditioning for audio reference(s): {cache_reference_key}"
            )

        return cached_result

    async def get_audio_conditioning(
        self,
        audio_reference: Union[
            str,
            Path,
            bytes,
            List[Union[str, Path, bytes]],
            Tuple[Union[str, Path, bytes], ...],
        ],
        max_ref_length=30,
        gpt_cond_len=6,
        gpt_cond_chunk_len=6,
        librosa_trim_db=None,
        sound_norm_refs=False,
        load_sr=22050,
    ):
        """Generate audio conditioning from reference files.

        Async wrapper around ``get_conditioning_latents`` with concurrency control
        and LRU caching for repeated reference audio requests.

        Args:
            audio_reference: Reference audio paths / payloads in the order they
                should be processed.
            max_ref_length (int, optional): Maximum reference length in seconds. Defaults to 30.
            gpt_cond_len (int, optional): Length of GPT conditioning. Defaults to 6.
            gpt_cond_chunk_len (int, optional): Length of each conditioning chunk. Defaults to 6.
            librosa_trim_db (float, optional): Trim silence below this dB threshold.
            sound_norm_refs (bool, optional): Whether to normalize reference audio. Defaults to False.
            load_sr (int, optional): Sampling rate for loading audio. Defaults to 22050.

        Returns:
            Tuple: GPT conditioning latents and speaker embeddings.
        """
        cache_key = self._get_conditioning_cache_key(
            audio_reference,
            max_ref_length,
            gpt_cond_len,
            gpt_cond_chunk_len,
            librosa_trim_db,
            sound_norm_refs,
            load_sr,
        )

        cached_result = await self._get_cached_conditioning_result(
            cache_key,
            cache_reference_key=cache_key[0],
            log_cache_hit=True,
        )
        if cached_result is not None:
            return cached_result

        async with self.encoder_semaphore:
            # Double-check cache after acquiring semaphore (another coroutine might have populated it)
            # Skip cache-hit logging here so requests that miss the first check do not
            # emit duplicate log lines after waiting on the semaphore.
            cached_result = await self._get_cached_conditioning_result(cache_key)
            if cached_result is not None:
                return cached_result

            # Run the original get_conditioning_latents in executor
            result = await self.get_conditioning_latents(
                audio_reference,
                max_ref_length,
                gpt_cond_len,
                gpt_cond_chunk_len,
                librosa_trim_db,
                sound_norm_refs,
                load_sr,
            )

            async with self._conditioning_cache_lock:
                self._conditioning_cache[cache_key] = result
                self._conditioning_cache.move_to_end(cache_key)
                if len(self._conditioning_cache) > self._max_cache_size:
                    self._conditioning_cache.popitem(last=False)
            return result

    async def get_model_logits(
        self,
        token_ids: List[int],
        merged_conditioning: torch.Tensor,
        request_id: str,
    ) -> torch.Tensor:
        """Run a single prefill pass to recover per-position hidden states.

        Unlike the V0 implementation this no longer relies on
        ``hidden_state_collector`` (removed in vLLM 0.9). Instead we submit a
        full EmbedsPrompt covering ``[conditioning, mel_bos, token_ids,
        mel_eos*4]`` with ``max_tokens=1`` and pick up the captured prefill
        tensor from ``XttsGPT._last_prefill_hidden_states``.

        Args:
            token_ids: Generated audio token IDs.
            merged_conditioning: ``[L_cond + L_text, hidden]`` tensor from
                :meth:`_merge_conditioning`, on the GPT model dtype/device.
            request_id: Unique request identifier (suffixed internally).

        Returns:
            torch.Tensor: ``[1, N, hidden]`` final-norm hidden states sliced
            to the ``[mel_bos, ..., token_ids[N-1]]`` positions, ready for
            the HiFiGAN decoder.
        """
        request_id = f"{request_id}_logits"
        # `merged_conditioning` already lives on the GPU; `_build_logits_*`
        # appends the [mel_bos | token_ids | mel_eos*4] block in the same
        # dtype/device.
        prompt_embeds = self._build_logits_prompt_embeds(
            merged_conditioning, token_ids)
        prefill_len = prompt_embeds.shape[0]

        XttsGPT.register_prefill_len(request_id, prefill_len)
        try:
            sampling_params = SamplingParams(
                detokenize=False,
                max_tokens=1,
                output_kind=RequestOutputKind.FINAL_ONLY,
            )

            generator = self.llm_engine.generate(
                prompt=EmbedsPrompt(prompt_embeds=prompt_embeds),
                sampling_params=sampling_params,
                request_id=request_id,
            )

            async for output in generator:  # consume the generator
                if output.finished:
                    break

            hidden_states = XttsGPT.pop_prefill_hidden_states(request_id)
        finally:
            XttsGPT.unregister_request(request_id)

        if hidden_states is None:
            raise RuntimeError(
                f"No hidden states collected for request {request_id}. "
                f"This should never happen! Please report this issue on GitHub."
            )

        start_of_audio_hs = merged_conditioning.shape[0]
        return self.final_norm(
            hidden_states[start_of_audio_hs:-5, ...]
            .unsqueeze(0)
            .to(self.device)
            .to(self.dtype)
        )

    @torch.inference_mode()
    async def get_generation_context(
        self,
        request: TTSRequest,
        gpt_cond_latent: Optional[torch.Tensor] = None,
        speaker_embeddings: Optional[torch.Tensor] = None,
    ) -> TokenGeneratorsAndPossiblyConditioning:
        """Get generation context for speech synthesis.

        Args:
            request (TTSRequest): TTS request object.
            gpt_cond_latent (Optional[torch.Tensor], optional): Pre-computed GPT conditioning latents.
            speaker_embeddings (Optional[torch.Tensor], optional): Pre-computed speaker embeddings.

        Returns:
            TokenGeneratorsAndPossiblyConditioning: Token generators and conditioning tensors.
        """
        if gpt_cond_latent is None or speaker_embeddings is None:
            # Prepare input with conditioning
            (
                tokens_list,
                gpt_embed_inputs,
                speaker_embeddings,
            ) = await self.prepare_inputs_async(
                request.text,
                request.language,
                request.speaker_files,
                request.max_ref_length,
                request.gpt_cond_len,
                request.gpt_cond_chunk_len,
                split_text=True,  # Split text to avoid OOM on big texts
            )
        else:
            tokens_list, text_embeddings = await self.prepare_text_tokens_async(
                request.text, request.language, split_text=True
            )
            gpt_embed_inputs = await self._merge_conditioning(
                text_embeddings, gpt_cond_latent
            )

        # XttsGPT bookkeeps per-request prefill lengths via the
        # ``HasInnerState`` hook, so we can submit every chunk to vLLM
        # concurrently and let the engine batch their decode steps together
        # (up to ``max_num_seqs``). Each generator just registers its
        # prefill_len once before kicking the engine and lets the worker
        # clean up via ``finished_requests_ids`` when generation completes.
        generators = []
        requests_id = []
        for seq_index, _ in enumerate(tokens_list):
            request_id = f"{request.request_id}_{seq_index}"
            requests_id.append(request_id)
            generators.append(self._submit_chunk(
                seq_index=seq_index,
                request=request,
                merged_conditioning=gpt_embed_inputs[seq_index],
                request_id=request_id,
            ))

        return generators, requests_id, speaker_embeddings, gpt_embed_inputs

    async def _submit_chunk(
        self,
        seq_index: int,
        request: TTSRequest,
        merged_conditioning: torch.Tensor,
        request_id: str,
    ):
        """Per-chunk wrapper around ``llm_engine.generate``.

        Builds the prompt embeddings, registers the prefill length under
        ``request_id`` (so :meth:`XttsGPT.forward` can rebase decode-time
        positions), then yields outputs from the engine. No external locking
        is needed: vLLM is free to co-batch this chunk's decode steps with
        any other in-flight chunk because ``XttsGPT`` carries per-request
        offsets through ``request_ids_to_seq_ids``.
        """
        prompt_embeds = self._build_generation_prompt_embeds(
            merged_conditioning)
        prefill_len = prompt_embeds.shape[0]
        XttsGPT.register_prefill_len(request_id, prefill_len)
        try:
            sampling_params = self._build_sampling_params(request)
            engine_generator = self.llm_engine.generate(
                prompt=EmbedsPrompt(prompt_embeds=prompt_embeds),
                sampling_params=sampling_params,
                request_id=request_id,
            )
            async for output in engine_generator:
                yield output
        finally:
            # vLLM will normally clear our state via finished_requests_ids,
            # but we also clear here to cover early generator close
            # (e.g. caller cancellation).
            XttsGPT.unregister_request(request_id)

    def _build_sampling_params(self, request: TTSRequest) -> SamplingParams:
        """Build per-chunk sampling params for the autoregressive pass.

        Honours every generation knob exposed on ``TTSRequest``:

        * ``temperature`` / ``top_p`` / ``top_k`` are forwarded directly.
        * ``do_sample=False`` switches the sampler to greedy by forcing
          ``temperature=0`` and ``top_k=1`` (vLLM's greedy convention).
        * ``seed`` is forwarded to ``SamplingParams.seed`` so the mel-token
          stream is reproducible for a given (text, speaker, params) tuple.
        * ``repetition_penalty`` is applied via our custom
          :class:`LogitsRepetitionPenalizer` (which can see the prompt
          tokens, unlike vLLM's built-in scalar penalty).
        * ``length_penalty`` is emulated via
          :class:`LogitsLengthPenalizer` because vLLM does not expose
          beam-search; the processor is a no-op when ``length_penalty == 1.0``.
        """
        if request.do_sample:
            temperature = request.temperature
            top_k = request.top_k
            top_p = request.top_p
        else:
            # Greedy decoding: vLLM treats temperature=0 as argmax. We also
            # collapse top_k/top_p to their no-op-for-greedy values.
            temperature = 0.0
            top_k = 1
            top_p = 1.0

        logits_processors = [
            LogitsRepetitionPenalizer(request.repetition_penalty),
        ]
        if request.length_penalty != 1.0:
            logits_processors.append(
                LogitsLengthPenalizer(request.length_penalty,
                                      eos_token_id=self.mel_eos_token_id))

        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=request.seed,
            detokenize=False,
            logits_processors=logits_processors,
            # We handle repetition penalty manually so the built-in stays
            # at its no-op value.
            repetition_penalty=1.0,
            max_tokens=self.gpt_config.gpt_max_audio_tokens,
            # The textual tokenizer's EOS is meaningless during audio
            # token generation; rely on the mel EOS token instead.
            ignore_eos=True,
            stop_token_ids=[self.mel_eos_token_id],
            output_kind=RequestOutputKind.FINAL_ONLY,
        )

    @torch.inference_mode()
    async def process_tokens_to_speech(
        self,
        generator: AsyncGenerator[RequestOutput, None],
        speaker_embeddings: Optional[torch.Tensor] = None,
        multimodal_data: Optional[torch.Tensor] = None,
        request: TTSRequest = None,
    ) -> AsyncGenerator[TTSOutput, None]:
        """Convert generated tokens to speech waveforms.

        Args:
            generator (AsyncGenerator[RequestOutput, None]): Token generator.
            speaker_embeddings (Optional[torch.Tensor], optional): Speaker embeddings.
            multimodal_data (Optional[torch.Tensor], optional): Additional multimodal data.
            request (TTSRequest, optional): Original TTS request.

        Yields:
            TTSOutput: Generated speech chunks.
        """
        assert speaker_embeddings is not None, (
            "Speaker embeddings must be provided for speech generation with XTTSv2."
        )
        assert multimodal_data is not None, (
            "Multimodal data must be provided for speech generation with XTTSv2."
        )

        async for output in generator:
            if output.finished:
                # get the hidden states
                hidden_states = await self.get_model_logits(
                    list(output.outputs[0].token_ids),
                    multimodal_data,
                    output.request_id,
                )

                async with self.decoder_semaphore:
                    async with self.cuda_memory_manager():
                        wav_tensor = await asyncio.to_thread(
                            self.hifigan_decoder,
                            hidden_states,
                            g=speaker_embeddings,
                        )
                        wav = wav_tensor.detach().cpu().numpy().squeeze()
                        del wav_tensor
                        del hidden_states

                # Per-utterance speed control via pitch-preserving phase
                # vocoder. Done before NovaSR so the 48 kHz output also
                # respects the requested rate.
                if request.speed != 1.0:
                    wav = librosa.effects.time_stretch(
                        wav.astype(np.float32), rate=float(request.speed))

                # Create the audio output
                tts_output = TTSOutput(
                    array=wav,
                    start_time=request.start_time,
                    token_length=len(output.outputs[0].token_ids),
                )

                # Apply NovaSR super-resolution if enabled
                if request.apply_novasr:
                    tts_output = tts_output.apply_super_resolution()

                yield tts_output

    async def shutdown(self):
        self.llm_engine.shutdown_background_loop()
