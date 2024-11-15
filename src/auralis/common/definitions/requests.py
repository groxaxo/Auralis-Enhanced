import uuid
from dataclasses import dataclass
from typing import Union, AsyncGenerator, Optional, List, Literal


@dataclass
class TTSRequest:
    """Container for TTS inference request data"""
    # Request metadata
    text: Union[AsyncGenerator[str, None], str, List[str]]
    language: str
    speaker_files: Union[List[str], bytes]  # Path to the speaker audio file
    request_id: str = uuid.uuid4().hex
    load_sample_rate: int = 22050
    sound_norm_refs: bool = False

    generate_every_n_chars: Optional[int] = None

    # Voice conditioning parameters
    max_ref_length: int = 60
    gpt_cond_len: int = 30
    gpt_cond_chunk_len: int = 4

    # Generation parameters
    stream: bool = False
    temperature: float = 0.75
    top_p: float = 0.85
    top_k: int = 50
    repetition_penalty: float = 5.0
    length_penalty: float = 1.0
    do_sample: bool = True

    def copy(self):
        return TTSRequest(
            **self.__dict__
        )
