from typing import List

import torch


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
