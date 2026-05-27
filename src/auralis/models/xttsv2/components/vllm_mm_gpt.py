"""XTTSv2 GPT decoder, vLLM-compatible (vllm >= 0.9.2, V0 engine).

The conditioning latents and text embeddings are pre-computed outside vLLM
and submitted via ``EmbedsPrompt`` / ``enable_prompt_embeds`` (see
``XTTSv2Engine``), so this module does not need vLLM's multimodal processor
machinery. It does, however, need per-request bookkeeping for two reasons:

* the XTTS ``LearnedPositionEmbeddings`` table is indexed in *audio-token*
  space (size ``max_audio_tokens + 3``), not in vLLM's absolute position
  space (size ``max_model_len``), so decode-time wpe lookups need
  ``audio_pos = absolute_pos − prefill_len + 1``;
* the hidden states for a request's prefill must be recoverable after
  ``generate(...)`` returns so the HiFiGAN decoder can be fed them.

vLLM provides exactly the hooks we need via the ``HasInnerState`` protocol:
declaring ``has_inner_state = True`` instructs ``ModelRunner.execute_model``
to forward ``request_ids_to_seq_ids`` and ``finished_requests_ids`` into
``forward``. We then use those, together with the attention metadata's
``query_start_loc``, to map every flat batch slot back to its originating
request. That removes the previous single-sequence (``max_num_seqs=1``)
restriction and restores concurrent batching.
"""

from collections.abc import Iterable
from typing import ClassVar, Dict, List, Optional, Tuple, Union

import random
import threading

import torch
import torch.nn as nn
from typing_extensions import Literal

from transformers import GPT2Config

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.gpt2 import GPT2Block
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    make_empty_intermediate_tensors_factory, make_layers)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors


class LearnedPositionEmbeddings(nn.Module):
    """XTTS' learned positional embedding table.

    Kept structurally identical to the original Coqui implementation so
    existing checkpoints load without modification. Used both inside the
    vLLM-managed ``XttsGPT`` for decode-time position lookups and by the
    outer engine when it pre-computes prompt embeddings for the text-token
    portion of the prompt (see ``XTTSv2Engine._build_*_prompt_embeds``).
    """

    def __init__(self, seq_len, model_dim, init=0.02, relative=False,
                 supports_pp=False):
        super().__init__()
        self.emb = (VocabParallelEmbedding(seq_len, model_dim)
                    if supports_pp else nn.Embedding(seq_len, model_dim))
        self.emb.weight.data.normal_(mean=0.0, std=init)
        self.relative = relative
        self.seq_len = seq_len

    def forward(self, x):
        sl = x.shape[1]
        if self.relative:
            start = random.randint(sl, self.seq_len) - sl
            indices = torch.arange(start, start + sl, device=x.device)
            assert (indices < self.seq_len).all() and (indices >= 0).all(), (
                f"Invalid position indices in forward: "
                f"min={indices.min().item()}, max={indices.max().item()}, "
                f"valid_range=[0,{self.seq_len - 1}]")
            return self.emb(indices)
        indices = torch.arange(0, sl, device=x.device)
        assert (indices < self.seq_len).all(), (
            f"Sequence length {sl} exceeds maximum position embedding length "
            f"{self.seq_len}")
        return self.emb(indices)

    def get_fixed_embedding(self, ind: torch.Tensor,
                            dev: torch.device) -> torch.Tensor:
        assert (ind < self.seq_len).all(), (
            f"Position indices out of range. Found max={ind.max().item()}, "
            f"but maximum allowed is {self.seq_len - 1}")
        assert (ind >= 0).all(), (
            f"Negative position indices found. Min value={ind.min().item()}")
        if ind.shape[0] > 1:
            return self.emb(ind)
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


class _XttsTransformer(nn.Module):
    """The GPT2 transformer stack used by XTTSv2.

    The AstraMindAI XTTS GPT checkpoint stores token and positional embedding
    tables under ``gpt.wte`` and ``gpt.wpe`` (matching the original Coqui
    layout), so we keep them here rather than promoting them to the outer
    class. The outer ``XttsGPT`` accesses them as ``self.gpt.wte`` /
    ``self.gpt.wpe`` to build prefill and decode-time embeddings.
    """

    def __init__(self, config: GPT2Config,
                 cache_config: Optional[CacheConfig],
                 quant_config: Optional[QuantizationConfig], prefix: str):
        super().__init__()
        self.config = config
        assert not config.add_cross_attention
        assert not config.scale_attn_by_inverse_layer_idx
        assert not config.reorder_and_upcast_attn
        self.embed_dim = config.hidden_size
        self.wte = VocabParallelEmbedding(config.num_audio_tokens,
                                          self.embed_dim)
        assert getattr(config, "max_audio_tokens", -1) != -1, (
            "XTTSv2 GPT config must define a positive max_audio_tokens")
        self.wpe = LearnedPositionEmbeddings(
            config.max_audio_tokens + 3,
            config.decoder_input_dim,
        )
        self.start_layer, self.end_layer, self.h = make_layers(
            config.num_hidden_layers,
            lambda prefix: GPT2Block(config, cache_config, quant_config,
                                     prefix=prefix),
            prefix=f"{prefix}.h")
        self.ln_f = nn.LayerNorm(self.embed_dim,
                                 eps=config.layer_norm_epsilon)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states"],
                                                    config.hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.h[self.start_layer:self.end_layer]:
            hidden_states = layer(hidden_states)
        if not get_pp_group().is_last_rank:
            return hidden_states  # caller wraps in IntermediateTensors
        return self.ln_f(hidden_states)


class XttsGPT(nn.Module, SupportsPP):
    """vLLM model wrapper around the XTTSv2 GPT decoder.

    ``has_inner_state`` (see :class:`vllm.model_executor.models.interfaces.HasInnerState`)
    asks the V0 ``ModelRunner`` to forward ``request_ids_to_seq_ids`` and
    ``finished_requests_ids`` into :meth:`forward`. We use those to:

    1. Look up each batch slot's prefill length when applying the XTTS
       learned position embedding to decode tokens.
    2. Slice the prefill hidden_states tensor by per-request boundaries so
       :meth:`XTTSv2Engine.get_model_logits` can recover the correct
       prefill output for each concurrently-decoded chunk.
    3. Garbage-collect per-request bookkeeping when vLLM tells us a request
       has finished.
    """

    has_inner_state: ClassVar[Literal[True]] = True

    # Concurrent-request bookkeeping. All access is single-threaded from
    # vLLM's worker thread (forward) plus the asyncio engine thread
    # (register/pop); a small lock keeps the dict consistent across those.
    _state_lock: ClassVar[threading.Lock] = threading.Lock()
    _prefill_lens_by_req: ClassVar[Dict[str, int]] = {}
    _prefill_hidden_states_by_req: ClassVar[Dict[str, torch.Tensor]] = {}

    @classmethod
    def register_prefill_len(cls, request_id: str, prefill_len: int) -> None:
        with cls._state_lock:
            cls._prefill_lens_by_req[request_id] = prefill_len

    @classmethod
    def pop_prefill_hidden_states(
            cls, request_id: str) -> Optional[torch.Tensor]:
        with cls._state_lock:
            return cls._prefill_hidden_states_by_req.pop(request_id, None)

    @classmethod
    def unregister_request(cls, request_id: str) -> None:
        with cls._state_lock:
            cls._prefill_lens_by_req.pop(request_id, None)
            cls._prefill_hidden_states_by_req.pop(request_id, None)

    @classmethod
    def _record_prefill_hidden_states(cls, request_id: str,
                                      hidden_states: torch.Tensor) -> None:
        with cls._state_lock:
            cls._prefill_hidden_states_by_req[request_id] = hidden_states

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.gpt_config = vllm_config.model_config.hf_config
        self.cache_config = vllm_config.cache_config
        self.quant_config = vllm_config.quant_config

        config = self.gpt_config
        self.audio_start_generation_token = config.start_audio_token

        self.gpt = _XttsTransformer(config, self.cache_config,
                                    self.quant_config, prefix="gpt")

        self.final_norm = nn.LayerNorm(config.hidden_size, bias=True,
                                       eps=config.layer_norm_epsilon)

        self.mel_head = ParallelLMHead(config.num_audio_tokens,
                                       config.hidden_size,
                                       bias=True,
                                       quant_config=self.quant_config,
                                       prefix="mel_head")

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(config.num_audio_tokens,
                                                config.num_audio_tokens,
                                                logit_scale)
        self.make_empty_intermediate_tensors = (
            self.gpt.make_empty_intermediate_tensors)

    # -- vLLM model interface -----------------------------------------------

    def get_input_embeddings(self,
                             input_ids: torch.Tensor) -> torch.Tensor:
        return self.gpt.wte(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        request_ids_to_seq_ids: Optional[Dict[str, List[int]]] = None,
        finished_requests_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # Garbage-collect any state for requests vLLM has just retired.
        if finished_requests_ids:
            for rid in finished_requests_ids:
                type(self).unregister_request(rid)

        is_first_rank = get_pp_group().is_first_rank
        # ``inputs_embeds`` carries real prompt embeddings only when the
        # caller submitted an ``EmbedsPrompt``; vLLM's profile / warmup run
        # uses dummy token IDs, so we cannot rely on that alone to detect
        # prefill. The attention metadata is authoritative.
        attn_md = self._get_attn_metadata()
        is_prefill_batch = self._batch_is_prefill(attn_md, positions)

        if is_first_rank:
            if is_prefill_batch and inputs_embeds is not None:
                # Real prefill via ``EmbedsPrompt``: embeddings already
                # include conditioning, text embeddings, and the
                # start-of-audio embedding with its learned positional
                # embedding baked in.
                hidden_states = inputs_embeds
            elif is_prefill_batch:
                # Warmup / profile prefill with dummy ``input_ids``. Build
                # embeddings with wpe clamped to position 0 so we don't
                # overflow the audio-space table.
                token_embeds = self.gpt.wte(input_ids)
                audio_positions = torch.zeros_like(positions)
                position_embeds = self.gpt.wpe.get_fixed_embedding(
                    audio_positions, input_ids.device).view(
                        token_embeds.shape)
                hidden_states = token_embeds + position_embeds
            else:
                # Decode batch: vLLM hands us one freshly-sampled audio
                # token per active request. Each token's absolute position
                # must be rebased into audio-token space using *its*
                # request's prefill_len before the XTTS wpe table can
                # index it.
                audio_positions = self._compute_decode_audio_positions(
                    positions, request_ids_to_seq_ids)
                # Source of the per-token embedding:
                #   ``enable_prompt_embeds=True`` flips vLLM's V0
                #   ``ModelRunner._compute_lens`` (worker/model_runner.py
                #   line 517-523 in 0.10.0) into the "prompt_embeds"
                #   branch, which fills ``inputs_embeds`` with
                #   ``seq_data.get_token_embeddings()`` — the cached
                #   embedding of every prompt token AND every previously
                #   sampled output token (the sampler runs
                #   ``self.model.get_input_embeddings(sampled_token_ids)``
                #   per step and appends to that cache) — and replaces
                #   ``input_ids`` with a placeholder of zeros. In that
                #   regime the real per-step token embedding lives in
                #   ``inputs_embeds``; doing ``self.gpt.wte(input_ids)``
                #   would silently embed token id 0 every decode step and
                #   ignore the sampled output, producing audio that is
                #   uncorrelated with the text prompt (this was the
                #   "English-shaped gibberish" failure mode). When
                #   ``inputs_embeds`` is missing (older vLLM, or any code
                #   path that does not opt into prompt_embeds), fall back
                #   to the explicit wte lookup so this remains correct on
                #   vllm < 0.10 / non-EmbedsPrompt callers.
                if inputs_embeds is not None:
                    token_embeds = inputs_embeds
                else:
                    token_embeds = self.gpt.wte(input_ids)
                position_embeds = self.gpt.wpe.get_fixed_embedding(
                    audio_positions, input_ids.device).view(
                        token_embeds.shape)
                hidden_states = token_embeds + position_embeds
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        hidden_states = self.gpt(hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        # Persist per-request prefill hidden states so
        # ``XTTSv2Engine.get_model_logits`` can recover them after
        # ``generate(...)`` returns. Only meaningful for real prefills
        # initiated via ``EmbedsPrompt`` (the profile / warmup batch has
        # no registered request_ids to look up).
        if (is_prefill_batch and inputs_embeds is not None
                and request_ids_to_seq_ids):
            self._store_prefill_hidden_states(
                hidden_states, request_ids_to_seq_ids, attn_md)

        return hidden_states

    @staticmethod
    def _batch_is_prefill(attn_md, positions: torch.Tensor) -> bool:
        """Decide whether the current forward is a prefill batch.

        Prefer the attention metadata's ``num_prefills`` / ``num_decode_tokens``
        counts; fall back to the positions tensor length heuristic if the
        metadata is missing (e.g. very old vLLM internals or a test stub).
        """
        if attn_md is not None:
            num_prefill = getattr(attn_md, "num_prefill_tokens", None)
            num_decode = getattr(attn_md, "num_decode_tokens", None)
            if num_prefill is not None and num_decode is not None:
                return num_prefill > 0 and num_decode == 0
        # Heuristic: decode batches have exactly one token per active
        # request, so positions are small. Anything bigger looks like a
        # prefill.
        return positions.numel() > 8

    def _compute_decode_audio_positions(
        self,
        positions: torch.Tensor,
        request_ids_to_seq_ids: Optional[Dict[str, List[int]]],
    ) -> torch.Tensor:
        """Compute the audio-space position for each decode token.

        For a decode batch with ``n=1`` sampling, each active request
        contributes exactly one query token. vLLM constructs
        ``request_ids_to_seq_ids`` in the same order as the seq_group
        metadata list, which in turn matches the order of flat decode tokens
        in ``positions``. We therefore build a per-token ``prefill_len``
        tensor by zipping the two together and use it to offset the
        absolute positions into the XTTS wpe table.
        """
        if not request_ids_to_seq_ids:
            # vLLM's profiling / warmup forward issues passes without going
            # through the engine wrapper, so no prefill_lens are registered.
            # Map everything to audio position 0 (the result is discarded
            # anyway during warmup).
            return torch.zeros_like(positions)

        req_ids = list(request_ids_to_seq_ids.keys())
        # Locate the decode segment within request_ids_to_seq_ids by
        # comparing batch sizes: pure-decode batches have exactly one query
        # token per request, so the *last* len(positions) request_ids belong
        # to the decoding requests when prefill+decode are co-batched. With
        # chunked_prefill disabled (our default) a forward call is either
        # all-prefill or all-decode, and len(req_ids) == positions.numel().
        if len(req_ids) != positions.numel():
            decode_req_ids = req_ids[-positions.numel():]
        else:
            decode_req_ids = req_ids

        with type(self)._state_lock:
            prefill_lens = [
                type(self)._prefill_lens_by_req.get(rid, 1)
                for rid in decode_req_ids
            ]
        prefill_lens_tensor = torch.as_tensor(
            prefill_lens, device=positions.device, dtype=positions.dtype)
        audio_positions = positions - (prefill_lens_tensor - 1)
        return torch.clamp(audio_positions, min=0,
                           max=self.gpt.wpe.seq_len - 1)

    def _store_prefill_hidden_states(
        self,
        hidden_states: torch.Tensor,
        request_ids_to_seq_ids: Dict[str, List[int]],
        attn_md,
    ) -> None:
        """Slice ``hidden_states`` per request and cache for later retrieval.

        ``query_start_loc`` from the prefill attention metadata gives the
        per-request boundaries in the flat batch; we iterate them in lock
        step with the prefill-request order in ``request_ids_to_seq_ids``.
        """
        prefill_md = self._get_prefill_metadata(attn_md)
        if prefill_md is None:
            return
        query_start_loc = getattr(prefill_md, "query_start_loc", None)
        num_prefills = getattr(attn_md, "num_prefills", None)
        if query_start_loc is None or num_prefills is None:
            return

        req_ids = list(request_ids_to_seq_ids.keys())
        # Prefill requests appear first in seq_group_metadata_list (see
        # vllm.worker.model_runner._prepare_model_input_tensors), so the
        # first ``num_prefills`` entries are prefills.
        prefill_req_ids = req_ids[:num_prefills]
        for i, rid in enumerate(prefill_req_ids):
            start = int(query_start_loc[i].item())
            end = int(query_start_loc[i + 1].item())
            type(self)._record_prefill_hidden_states(
                rid, hidden_states[start:end].detach())

    @staticmethod
    def _get_attn_metadata():
        try:
            ctx = get_forward_context()
        except Exception:
            return None
        attn_md = getattr(ctx, "attn_metadata", None)
        if isinstance(attn_md, dict):
            # vLLM V1 returns a per-layer dict; we only need one entry to
            # read the common per-request fields.
            attn_md = next(iter(attn_md.values()), None)
        return attn_md

    @staticmethod
    def _get_prefill_metadata(attn_md):
        if attn_md is None:
            return None
        # V0 FlashAttention attn_metadata splits prefill and decode into
        # sub-objects. For pure-prefill batches ``prefill_metadata`` is the
        # parent object itself; for chunked or mixed batches it's a separate
        # struct accessible via ``prefill_metadata``.
        return getattr(attn_md, "prefill_metadata", attn_md)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        # The XTTS GPT uses ``final_norm`` separately from the in-stack
        # ``ln_f``; the original Coqui implementation calls both before the
        # LM head.
        hidden_states = self.final_norm(hidden_states)
        return self.logits_processor(self.mel_head, hidden_states,
                                     sampling_metadata, self.mel_head.bias)

    # -- Weight loading -----------------------------------------------------

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> set:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_names: set = set()
        for name, loaded_weight in weights:
            if name not in params_dict:
                continue
            param = params_dict[name]
            # HF GPT-2 stores attention / projection weights via Conv1D, so
            # we transpose before copying into vLLM's parallel Linear layers.
            if ("c_attn" in name or "c_proj" in name or "c_fc" in name) \
                    and name.endswith(".weight"):
                loaded_weight = loaded_weight.t()
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_names.add(name)
        missing = set(params_dict.keys()) - loaded_names
        assert not missing, (
            f"Missing weights: {missing}. This usually means the checkpoint "
            f"does not match the XttsGPT architecture. Your model has these "
            f"params: {set(params_dict.keys())}")
        return loaded_names
