from dataclasses import dataclass
from typing import Union, AsyncGenerator, Optional


@dataclass
class TTSRequest:
    """Container for XTTS inference request data"""
    request_id: str
    text: Union[AsyncGenerator[str, None], str]
    language: str
    speaker_file: str  # Path to the speaker audio file
    generate_every_n_chars: Optional[int] = None
    temperature: float = 0.75
    top_p: float = 0.85
    top_k: int = 50
    repetition_penalty: float = 5.0
    length_penalty: float = 1.0
    do_sample: bool = True
    max_ref_length: int = 60
    gpt_cond_len: int = 30
    gpt_cond_chunk_len: int = 4

