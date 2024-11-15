import functools
import random
from array import array
from dataclasses import dataclass

import torch
import torch.nn as nn
from typing import Optional, Union, Iterable, Tuple, Mapping

from torch import Tensor
from transformers import GPT2Config
from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, MultiModalConfig
from vllm.distributed import get_pp_group
from vllm.inputs import InputContext, INPUT_REGISTRY, DecoderOnlyInputs, token_inputs
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.gpt2 import GPT2Block
from vllm.model_executor.models.utils import make_layers, make_empty_intermediate_tensors_factory
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalInputs
from vllm.sequence import IntermediateTensors, SequenceData, VLLM_TOKEN_ID_ARRAY_TYPE
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP

from typing import Dict, List
from collections import defaultdict

PrefillLength= Union[int, List[int]]
TokenPosition= Union[int, List[int]]

@dataclass
class TokenPositionAndPrefillTuple:
    prefill_len: Optional[PrefillLength] = None
    pos_id: Optional[TokenPosition] = None

class PositionalEmbeddingsCorrecter:
    """Corrects positional embeddings for XTTS model,
    since they have a different length than the text embeddings.
    This class tracks tokens both by request_id and position for vLLM compatibility.
    """

    def __init__(self):
        # Maps request_id to its prefill length
        self.request_tracker_dict: Dict[str, TokenPositionAndPrefillTuple] = defaultdict(lambda: TokenPositionAndPrefillTuple())
        # Maps token_position pairs to their request_id
        self.token_to_request: Dict[str, str] = {}

    def init_request_id_prefill(self, request_id: str, prefill_len: PrefillLength, nex_token: torch.Tensor):
        """Initialize a request_id with its prefill length."""
        self.request_tracker_dict[request_id] = TokenPositionAndPrefillTuple(prefill_len, prefill_len)
        self.token_to_request[f"{nex_token}_{prefill_len}"] = request_id

    def get_by_request_id(self, request_id: str) -> TokenPositionAndPrefillTuple:
        """Retrieve the prefill length for a given request_id."""
        return self.request_tracker_dict.get(request_id, None)

    def get_by_next_token(self, next_token_ids: List[int], next_position_ids: List[int]) -> List[TokenPositionAndPrefillTuple]:
        """Retrieve prefill lengths for given token and position pairs.

        Args:
            next_token_ids: List of token IDs
            next_position_ids: List of position IDs, corresponding to token IDs

        Returns:
            List of prefill lengths for each token-position pair

        Raises:
            ValueError: If no valid token mappings are found
        """
        prefill_lengths = []
        for next_token_id, next_position_id in zip(next_token_ids, next_position_ids):
            token_key = f"{next_token_id}_{next_position_id}"
            if token_key in self.token_to_request:
                request_id = self.token_to_request[token_key]
                prefill_lengths.append(self.request_tracker_dict[request_id])

        if not prefill_lengths:
            raise ValueError(f"No valid mappings found for token pairs")
        return prefill_lengths

    def _invalidate_previous_mapping(self, request_id: str):
        """Remove all token mappings associated with a given request_id.

        This prevents memory leaks from old token mappings and ensures
        we don't have stale token-to-request associations.
        """
        # Find all token keys that map to this request_id
        keys_to_remove = [
            token_key for token_key, req_id in self.token_to_request.items()
            if req_id == request_id
        ]

        # Remove all found mappings
        for token_key in keys_to_remove:
            del self.token_to_request[token_key]

    def _get_pos_id_and_update (self, request_id: str):
        """Get the position ID for a given request_id and update it."""
        tuple_prefill_token = self.get_by_request_id(request_id)
        # Update the position ID
        self.request_tracker_dict[request_id] = TokenPositionAndPrefillTuple(tuple_prefill_token.prefill_len, tuple_prefill_token.pos_id + 1)
        return tuple_prefill_token.pos_id + 1


    def associate_new_tokens(self, request_id: str, next_token_id: int):
        """Associate a new token-position pair with a request_id.

        Before creating the new association, it removes all previous
        token mappings for this request_id to maintain consistency.

        Args:
            request_id: The request identifier
            next_token_id: The token ID to associate
        """
        pos_id = self._get_pos_id_and_update(request_id)

        # Clean up old mappings first
        self._invalidate_previous_mapping(request_id)

        # Create new mapping
        self.token_to_request[f"{next_token_id}_{pos_id}"] = request_id

    def clear_request(self, request_id: str):
        """Remove all data associated with a request_id.

        This includes both the prefill length tracking and any token mappings.
        """
        if request_id in self.request_tracker_dict:
            # First remove all token mappings
            self._invalidate_previous_mapping(request_id)
            # Then remove the request tracking
            del self.request_tracker_dict[request_id]

class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=0.02, relative=False, supports_pp=False):
        super().__init__()
        # nn.Embedding
        self.emb = VocabParallelEmbedding(seq_len, model_dim) if supports_pp else nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)
        self.relative = relative
        self.seq_len = seq_len

    def forward(self, x):
        sl = x.shape[1]
        if self.relative:
            start = random.randint(sl, self.seq_len) - sl
            indices = torch.arange(start, start + sl, device=x.device)
            # Validate indices
            assert (indices < self.seq_len).all() and (indices >= 0).all(), \
                f"Invalid position indices in forward: min={indices.min().item()}, max={indices.max().item()}, valid_range=[0,{self.seq_len-1}]"
            return self.emb(indices)
        else:
            indices = torch.arange(0, sl, device=x.device)
            # Validate indices
            assert (indices < self.seq_len).all(), \
                f"Sequence length {sl} exceeds maximum position embedding length {self.seq_len}"
            return self.emb(indices)

    def get_fixed_embedding(self, ind: torch.Tensor, dev: torch.device) -> torch.Tensor:
        """Get position embeddings with batch support.

        Args:
            ind: Position indices tensor. Can be single or batched
                 Shape: [..., seq_len] or [seq_len]
            dev: Target device for the embeddings

        Returns:
            Position embeddings tensor matching input shape plus embedding dimension
            Shape: [batch_size, seq_len, model_dim] or [1, 1, model_dim]
        """
        # Validation degli indici
        assert (ind < self.seq_len).all(), \
            f"Position indices out of range. Found max={ind.max().item()}, but maximum allowed is {self.seq_len-1}"
        assert (ind >= 0).all(), \
            f"Negative position indices found. Min value={ind.min().item()}"

        if ind.shape[0] > 1:

            return self.emb(ind)
        else:
            #assert ind.dim() <= 2, f"Single input should have 1 or 2 dimensions, got {ind.dim()}"
            return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)



def get_xtts_max_audio_tokens(ctx: InputContext) -> int:
    """Calculate maximum audio tokens based on text context and audio duration."""
    return 32 # the conditoning perciever output


def dummy_seq_data_for_xtts(
        ctx: InputContext,
        seq_len: int,
        audio_count: int,
) -> SequenceData:
    """Create dummy sequence data for XTTS profiling."""
    # Calculate audio token space needed
    audio_placeholder = array(
        VLLM_TOKEN_ID_ARRAY_TYPE,
        [1]
    ) * 32 # the conditioning perceiver output

    # Add separator between chunks
    audio_token_ids = (audio_placeholder + array(VLLM_TOKEN_ID_ARRAY_TYPE, [1])) * audio_count

    # Fill remaining sequence with padding
    other_token_ids = array(VLLM_TOKEN_ID_ARRAY_TYPE, [1]) * (seq_len - len(audio_token_ids))
    # not -1 since we add the start audio token

    return SequenceData(
        audio_token_ids +
        other_token_ids
    )

def dummy_conditioning_for_xtts(
        ctx: InputContext,
        seq_len: int,
        audio_count: int,
) -> dict:
    """Create dummy conditioning data for XTTS."""
    return {
        "audio": {
            "embeds":[
            torch.zeros(
                (seq_len, ctx.model_config.hf_config.hidden_size),
                dtype=ctx.model_config.dtype) for _ in range(audio_count)
        ],
            "is_logits_only_mode": False,
        }
    }


def dummy_data_for_xtts(
        ctx: InputContext,
        seq_len: int,
        mm_counts: Mapping[str, int],
) -> Tuple[SequenceData, dict]:
    """Create complete dummy data for XTTS profiling."""
    audio_count = mm_counts["audio"]
    seq_data = dummy_seq_data_for_xtts(ctx, seq_len, audio_count)
    cond_data = dummy_conditioning_for_xtts(ctx, seq_len, audio_count)
    return seq_data, cond_data


def input_mapper_for_xtts(ctx: InputContext, data: Union[Dict, List[Tensor]]) -> MultiModalInputs:
    """Map input data to XTTS format."""

    assert isinstance(data, dict), "XTTS MultiModal input data must be a dictionary with keys: 'embeds', 'is_logits_only_mode'"

    embeds = data.get("embeds")
    is_logits_only_mode = data.get("is_logits_only_mode", False)

    # Each item should be a torch tensor
    for audio_input in embeds:
        if not isinstance(audio_input, Tensor):
            raise NotImplementedError(f"Unsupported data type: {type(audio_input)}")

    return MultiModalInputs({"cond_latents": embeds,
                             "is_logits_only_mode": is_logits_only_mode})


def input_processor_for_xtts2_gpt(ctx: InputContext, inputs: DecoderOnlyInputs):
    """
    We'll accomodate for the extra contditioning token and for the start audio token,
    we actually insert a -1 repeated for the differecne in length between the conditioning and the tokenized text
    and then we add 1 for the start audio token
    Args:
        ctx:
        inputs:

    Returns:

    """
    multi_modal_data = inputs.get("multi_modal_data")
    audio_dict = multi_modal_data['audio']
    audio = audio_dict.get('embeds')

    is_last_decoding_pass = audio_dict.get("is_logits_only_mode", False)

    prompt_token_ids = inputs.get("prompt_token_ids")

    if not is_last_decoding_pass:
        # we fill everything with 0 since we don't actually needs text token ids, it would mess up in the sampling step
        new_token_ids = ([1] * (audio.shape[0])) + [ctx.model_config.hf_config.start_audio_token] # add the start audio generation token
    else:
        new_token_ids = ([1] * audio.shape[0]) + prompt_token_ids
    # the encoding had already been done externally to reuse the embeddings for later use but we
    # account for the new token that will be added before generation
    new_prompt = None
    return token_inputs(prompt_token_ids=new_token_ids,
                 prompt=new_prompt,
                 multi_modal_data=multi_modal_data)


@MULTIMODAL_REGISTRY.register_input_mapper("audio", input_mapper_for_xtts)
@MULTIMODAL_REGISTRY.register_max_multimodal_tokens("audio", get_xtts_max_audio_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_xtts)
@INPUT_REGISTRY.register_input_processor(input_processor_for_xtts2_gpt)
class XttsGPT(nn.Module, SupportsMultiModal, SupportsPP):
    def __init__( # type: ignore
            self,
            config: GPT2Config,
            multimodal_config: MultiModalConfig,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        self.prefix_sequence_dict: Dict[str, torch.Tensor] = {}
        # Core GPT components
        self.gpt = GPT2Model(
            config,
            cache_config,
            quant_config,
            prefix="gpt"
        )
        self.final_norm =  nn.LayerNorm(config.hidden_size, bias=True, eps=config.layer_norm_epsilon)
        # Output head for mel tokens
        self.mel_head = ParallelLMHead(
            config.num_audio_tokens,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix="mel_head"
        )
        self.audio_start_generation_token = config.start_audio_token

        # Initialize logits processor and sampler
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(config.num_audio_tokens,
                                                config.num_audio_tokens,
                                                logit_scale)
        self.sampler = Sampler()

        self.positional_embeddings_correcter = PositionalEmbeddingsCorrecter()

    @staticmethod
    def check_is_logits_only_mode(is_logits_only_mode):

        # First check if it's a boolean
        if isinstance(is_logits_only_mode, bool):
            return is_logits_only_mode

        # Then check if it's a tensor
        if torch.is_tensor(is_logits_only_mode):
            # if it's a scalar tensor, return the value
            if is_logits_only_mode.numel() == 1:
                return bool(is_logits_only_mode.item())
            # for non-scalar tensors, check if all elements are the same
            return is_logits_only_mode.any()

        # Fallback
        return bool(is_logits_only_mode)

    @staticmethod
    def _calculate_start_token_indices(cond_latents: List[torch.Tensor]) -> List[int]:
        """Calcola gli indici dove inserire i token di start.

        Args:
            cond_latents: Lista di tensori di condizionamento

        Returns:
            Lista di indici dove inserire i token di start
        """
        indices = []
        current_idx = 0

        for cond_latent in cond_latents:
            # Add
            current_idx += cond_latent.shape[0]
            # Aggiungi l'indice per il token di start dopo questo segmento
            indices.append(current_idx)
            # Incrementa per il token di start che verrà aggiunto
            current_idx += 1

        return indices

    # noinspection PyMethodOverriding
    def forward( # type: ignore
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            intermediate_tensors: Optional["IntermediateTensors"] = None,
            cond_latents: Optional[torch.Tensor] = None,
            is_logits_only_mode: bool = False,
            **kwargs,
    ) -> Union[torch.Tensor, "IntermediateTensors"]:
        """Forward pass following VLLM pattern."""
        # it is not the first iter either if the cond latents are emtpy or if the kv_caches are not empty
        is_first_iteration = len(input_ids) > 1 and torch.isin(input_ids, torch.tensor([1, 1024], device=input_ids.device)).all()

        is_logits_only_mode = self.check_is_logits_only_mode(is_logits_only_mode)

        if not is_first_iteration and not is_logits_only_mode:
            correct_positions_ids = self.positional_embeddings_correcter.get_by_next_token(input_ids.tolist(), positions.tolist())
            positions = 1 + positions - torch.tensor([correct_positions_id.prefill_len for correct_positions_id in correct_positions_ids], device=positions.device)


        hidden_states = self.gpt(
            input_ids=input_ids,
            position_ids=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            # this is the conditioning input ( voice conditioning + text_embeds )
            input_embeds=cond_latents,
            is_first_iteration=is_first_iteration,
            is_logits_only_mode=is_logits_only_mode
        )

        return hidden_states

    # noinspection PyUnresolvedReferences
    def compute_logits(
            self,
            hidden_states: torch.Tensor,
            sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:

        # normalize the hidden states
        hidden_states = self.final_norm(hidden_states)

        for seq in sampling_metadata.seq_groups:
            # Check if we need to collect hidden states
            sampling_params = seq.sampling_params
            if (hasattr(sampling_params, 'hidden_state_collector')
                    and sampling_params.hidden_state_collector is not None):
                self.positional_embeddings_correcter.clear_request(sampling_params.request_id)
                # Call the collector directly with the hidden states
                sampling_params.hidden_state_collector(hidden_states, sampling_params.request_id)  # The request_id is already bound

        # Compute logits using the mel_head
        logits = self.logits_processor(self.mel_head, hidden_states, sampling_metadata)
        return logits

    # noinspection PyUnresolvedReferences
    def sample(
            self,
            logits: torch.Tensor,
            sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        for seq_id, seq_groups in enumerate(sampling_metadata.seq_groups):
            if hasattr(seq_groups.sampling_params, 'request_id') and seq_groups.sampling_params.request_id is not None:
                idx = seq_groups.seq_ids[0]
                # Call the collector directly with the next tokens
                if not self.positional_embeddings_correcter.get_by_request_id(seq_groups.sampling_params.request_id):
                    self.positional_embeddings_correcter.init_request_id_prefill(
                        request_id = seq_groups.sampling_params.request_id,
                        prefill_len=len(seq_groups.seq_data[idx].prompt_token_ids),
                        nex_token=next_tokens.outputs[seq_id].samples[0].output_token # index out of error
                    )
                else:
                    self.positional_embeddings_correcter.associate_new_tokens(
                        request_id=seq_groups.sampling_params.request_id,
                        next_token_id=next_tokens.outputs[seq_id].samples[0].output_token)

        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights following VLLM pattern."""
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_names = set()
        for name, loaded_weight in weights:
            if name not in params_dict:
                continue

            param = params_dict[name]
            if "c_attn" in name or "c_proj" in name or "c_fc" in name:
                if name.endswith(".weight"):
                    loaded_weight = loaded_weight.t()

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_names.add(name)
        # used to check if all weights were loaded
        assert set(params_dict.keys()) - loaded_names == set(), \
            (f"Missing weights: {set(params_dict.keys()) - loaded_names}, "
             f"this probably means you are using an incompatible model ")

class GPT2Model(nn.Module):

    def __init__(
            self,
            config: GPT2Config,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
    ):
        super().__init__()
        self.config = config
        assert not config.add_cross_attention
        assert not config.scale_attn_by_inverse_layer_idx
        assert not config.reorder_and_upcast_attn
        self.embed_dim = config.hidden_size
        self.wte = VocabParallelEmbedding(config.num_audio_tokens, self.embed_dim)
        self.wpe = (
            LearnedPositionEmbeddings(config.max_audio_tokens + 3, config.decoder_input_dim)
            if config.max_audio_tokens != -1
            else functools.partial(config.null_position_embeddings, dim=config.decoder_input_dim)
        )
        self.start_layer, self.end_layer, self.h = make_layers(
            config.num_hidden_layers,
            lambda prefix: GPT2Block(
                config, cache_config, quant_config, prefix=prefix),
            prefix=f"{prefix}.h")
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states"],
                                                    config.hidden_size))

    #def is_tensor_errored(self, tensor: torch.Tensor):
    #    try:
    #        if not torch.is_tensor(tensor):
    #            return False
    #        if tensor[0] == 0:
    #            return False
    #    except Exception as e:
    #        print(f"Error: {e}")
    #        return True
    def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            intermediate_tensors: Optional[IntermediateTensors],
            input_embeds: Optional[torch.Tensor] = None,
            is_first_iteration: bool = False,
            is_logits_only_mode: bool = False,
    ) -> Union[torch.Tensor, IntermediateTensors]:

        if get_pp_group().is_first_rank:
            if isinstance(input_embeds, torch.Tensor) and len(input_embeds) > 1 and len(input_embeds.shape) < 4:
                # if two equal tensors are passed, vllm aggregate them in a new (batched) tensor
                input_embeds = list(input_embeds)  # so we unbacth them :) (unless we are in the profiling run)
            if is_first_iteration and not is_logits_only_mode:
                input_ids = input_ids[-1].reshape(1, 1)
            elif is_logits_only_mode:
                if isinstance(input_embeds, list):
                    starting_idx = []
                    for input_embed in input_embeds:
                        starting_idx.append(input_embed.shape[0])

                    ending_ids = attn_metadata.seq_lens

                    cumulative_starts = [starting_idx[0]]
                    cumulative_ends = [ending_ids[0]]

                    for i in range(1, len(starting_idx)):
                        next_start = cumulative_ends[i - 1] + starting_idx[i]
                        next_end = cumulative_ends[i - 1] + ending_ids[i]
                        cumulative_starts.append(next_start)
                        cumulative_ends.append(next_end)

                    ids_for_unpacking = [end - start for start, end in zip(cumulative_starts, cumulative_ends)]

                    input_ids = torch.cat([
                        input_ids[start:end].reshape(1, -1)
                        for start, end in zip(cumulative_starts, cumulative_ends)
                    ], dim=-1)
                    position_ids = torch.cat([
                        torch.arange(0, end - start, device=input_ids.device).reshape(1, -1)
                        for start, end in zip(cumulative_starts, cumulative_ends)
                    ], dim=-1).squeeze(0)

                else:
                    input_ids = input_ids[input_embeds.shape[1]:].reshape(1, -1)
                    position_ids = torch.arange(0, input_ids.shape[1], device=input_ids.device)

            else:
                input_ids = input_ids

            audio_inputs_embeds = self.wte(input_ids).squeeze(0)

            position_embeds = self.wpe.get_fixed_embedding(
                position_ids, input_ids.device
            ) if not is_first_iteration \
                else self.wpe(audio_inputs_embeds.reshape(-1, 1))

            hidden_states = audio_inputs_embeds + position_embeds

            if is_first_iteration or is_logits_only_mode:
                if isinstance(input_embeds, list):
                    input_embeds = [input_embed.view(-1, input_embed.shape[-1]) for input_embed in input_embeds]

                    if is_logits_only_mode:
                        hidden_states = list(hidden_states.split(ids_for_unpacking, dim=0))

                    hidden_states = torch.cat([
                        tensor for pair in zip(input_embeds, [hidden_states] * len(input_embeds)
                        if not isinstance(hidden_states, list) else hidden_states)
                        for tensor in pair
                    ], dim=0)
                else:
                    input_embeds = input_embeds.view(-1, input_embeds.shape[-1])
                    if input_embeds.shape[0] == attn_metadata.num_prefill_tokens: # this is the profiling run
                        input_embeds = input_embeds[:-1]
                    hidden_states = torch.cat([input_embeds, hidden_states], dim=0)

            hidden_states = hidden_states.view(-1, self.embed_dim)
            if hidden_states.shape[0] != (attn_metadata.num_prefill_tokens+attn_metadata.num_decode_tokens):
                print('Errore')
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.h[i]
            hidden_states = layer(hidden_states,
                                  kv_caches[i - self.start_layer],
                                  attn_metadata)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.ln_f(hidden_states)
        return hidden_states

