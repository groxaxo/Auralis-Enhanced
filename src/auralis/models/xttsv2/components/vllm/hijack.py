import math
from typing import List

import torch


class LogitsLengthPenalizer:
    """Soft per-step emulation of HuggingFace's beam-search ``length_penalty``.

    vLLM does not implement beam search, so ``SamplingParams.length_penalty``
    is unavailable. To still honour ``TTSRequest.length_penalty`` we install
    this processor and bias the mel-EOS token's logit at each decode step:

        EOS_logit += (1.0 - length_penalty) * sqrt(n + 1)

    where ``n`` is the number of audio tokens already generated. The
    ``sqrt`` keeps the effect growing slowly with length so the bias never
    swamps the model's actual confidence in EOS.

    Sign matches HF's documented behaviour: ``length_penalty > 1`` makes
    the EOS coefficient negative, suppressing premature ending and
    yielding longer audio; ``length_penalty < 1`` boosts EOS and ends
    sooner. ``length_penalty == 1.0`` is a no-op.
    """

    def __init__(self, length_penalty: float, eos_token_id: int):
        self.length_penalty = float(length_penalty)
        self.eos_token_id = int(eos_token_id)

    def __call__(self, prompt_token_ids: List[int], token_ids: List[int],
                 logits: torch.Tensor) -> torch.Tensor:
        if self.length_penalty == 1.0:
            return logits
        n = len(token_ids)
        adjustment = (1.0 - self.length_penalty) * math.sqrt(n + 1)
        logits[self.eos_token_id] = logits[self.eos_token_id] + adjustment
        return logits


class LogitsRepetitionPenalizer:
    """Logits processor for preventing repetitive text generation.

    Implements a repetition penalty that modifies token logits based on prior
    occurrences in the prompt and generated sequence. Used by the XTTSv2
    engine via the vLLM `SamplingParams.logits_processors` hook to discourage
    the audio decoder from getting stuck on a token.
    """

    def __init__(self, repetition_penalty: float):
        if repetition_penalty < 0:
            raise ValueError("Repetition penalty must be non-negative")
        self.repetition_penalty = repetition_penalty

    def __call__(self, prompt_token_ids: List[int], token_ids: List[int],
                 logits: torch.Tensor) -> torch.Tensor:
        if self.repetition_penalty == 1.0 or (not token_ids
                                              and not prompt_token_ids):
            return logits

        repeated_tokens = torch.tensor(prompt_token_ids + token_ids,
                                       device=logits.device,
                                       dtype=torch.long)
        repeated_logits = logits[repeated_tokens]
        repeated_logits = torch.where(
            repeated_logits > 0,
            repeated_logits / self.repetition_penalty,
            repeated_logits * self.repetition_penalty,
        )
        logits[repeated_tokens] = repeated_logits
        return logits
