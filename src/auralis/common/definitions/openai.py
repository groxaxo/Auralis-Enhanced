import base64
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from auralis.common.definitions.requests import TTSRequest


def _dataclass_default(field):
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:  # type: ignore[comparison-overlap]
        return field.default_factory()
    return None


tts_defaults = {field.name: _dataclass_default(field) for field in fields(TTSRequest)}


class ChatCompletionMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class VoiceChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatCompletionMessage]
    speaker_files: List[str] = Field(
        ..., description="List of base64-encoded reference audio files"
    )
    modalities: List[Literal["text", "audio"]] = Field(
        default_factory=lambda: ["text", "audio"],
        description="Output modalities to return",
    )
    openai_api_url: Optional[str] = Field(
        default=None, description="OpenAI-compatible text-generation endpoint"
    )
    vocalize_at_every_n_words: int = Field(default=100, ge=1)
    stream: bool = Field(default=True)

    enhance_speech: bool = Field(default=tts_defaults["enhance_speech"])
    language: str = Field(default=tts_defaults["language"])
    max_ref_length: int = Field(default=tts_defaults["max_ref_length"])
    gpt_cond_len: int = Field(default=tts_defaults["gpt_cond_len"])
    gpt_cond_chunk_len: int = Field(default=tts_defaults["gpt_cond_chunk_len"])
    temperature: float = Field(default=tts_defaults["temperature"])
    top_p: float = Field(default=tts_defaults["top_p"])
    top_k: int = Field(default=tts_defaults["top_k"])
    repetition_penalty: float = Field(default=tts_defaults["repetition_penalty"])
    length_penalty: float = Field(default=tts_defaults["length_penalty"])
    do_sample: bool = Field(default=tts_defaults["do_sample"])
    ref_text: Optional[Union[str, List[str]]] = None

    @field_validator("openai_api_url")
    def validate_oai_url(cls, value):
        if value is None:
            raise ValueError("An OpenAI-compatible text generation URL is required")
        return value

    @field_validator("stream")
    def validate_stream(cls, value):
        if not value:
            raise ValueError(
                "Streaming must be enabled; use /v1/audio/speech for non-streaming audio"
            )
        return value

    @field_validator("speaker_files")
    def validate_speaker_files(cls, value):
        if not value:
            raise ValueError("At least one speaker file is required")
        for encoded_file in value:
            try:
                base64.b64decode(encoded_file, validate=True)
            except Exception as exc:
                raise ValueError("Invalid base64 encoding in speaker file") from exc
        return value

    def to_tts_request(self, text: str = "") -> TTSRequest:
        speaker_data_list = [base64.b64decode(item) for item in self.speaker_files]
        return TTSRequest(
            text=text,
            stream=False,
            speaker_files=speaker_data_list,
            enhance_speech=self.enhance_speech,
            language=self.language,
            max_ref_length=self.max_ref_length,
            gpt_cond_len=self.gpt_cond_len,
            gpt_cond_chunk_len=self.gpt_cond_chunk_len,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
            do_sample=self.do_sample,
            ref_text=self.ref_text,
        )

    def to_openai_request(self) -> Dict[str, Any]:
        excluded = {
            "speaker_files",
            "openai_api_url",
            "vocalize_at_every_n_words",
            "modalities",
            "ref_text",
            *tts_defaults.keys(),
        }
        payload = {
            key: value
            for key, value in self.model_dump().items()
            if key not in excluded
        }
        payload["stream"] = True
        return payload


def _resolve_vllm_voice(voice: str) -> List[str]:
    """Resolve bundled XTTS reference voices without machine-specific paths."""

    repository_root = Path(__file__).resolve().parents[4]
    voice_mapping = {
        "alloy": repository_root / "voice_library/colombiana/sample_0.wav",
        "echo": repository_root / "voice_library/facu/sample_0.mp3",
        "fable": repository_root / "voice_library/luisiana/sample_0.opus",
        "onyx": repository_root / "voice_library/elgriego/sample_0.mp3",
        "nova": repository_root / "tests/resources/audio_samples/female.wav",
        "shimmer": repository_root / "tests/resources/audio_samples/female.wav",
    }
    candidate = voice_mapping.get(voice.lower())
    if candidate and candidate.exists():
        return [str(candidate)]

    direct_path = Path(voice).expanduser()
    if direct_path.exists():
        return [str(direct_path)]

    available = ", ".join(sorted(voice_mapping))
    raise ValueError(
        f"Voice {voice!r} is not a bundled XTTS reference. Available aliases: "
        f"{available}; alternatively provide base64 audio files."
    )


class AudioSpeechGenerationRequest(BaseModel):
    input: str = Field(..., description="The textual input to convert")
    model: str = Field(..., description="The model requested by the client")
    voice: Union[str, List[str]] = Field(
        default="alloy",
        description=(
            "MLX voice/speaker name, a local XTTS voice alias/path, or a list of "
            "base64-encoded reference audio files"
        ),
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = (
        Field(default="mp3")
    )
    speed: float = Field(default=1.0, ge=0.25, le=4.0)

    enhance_speech: bool = Field(default=tts_defaults["enhance_speech"])
    language: str = Field(default="auto")
    max_ref_length: int = Field(default=tts_defaults["max_ref_length"])
    gpt_cond_len: int = Field(default=tts_defaults["gpt_cond_len"])
    gpt_cond_chunk_len: int = Field(default=tts_defaults["gpt_cond_chunk_len"])
    temperature: float = Field(default=tts_defaults["temperature"])
    top_p: float = Field(default=tts_defaults["top_p"])
    top_k: int = Field(default=tts_defaults["top_k"])
    repetition_penalty: float = Field(default=tts_defaults["repetition_penalty"])
    length_penalty: float = Field(default=tts_defaults["length_penalty"])
    do_sample: bool = Field(default=tts_defaults["do_sample"])
    apply_novasr: bool = Field(default=False)
    ref_text: Optional[Union[str, List[str]]] = None
    instruct: Optional[str] = None
    max_tokens: Optional[int] = 1200

    @field_validator("voice")
    def validate_voice(cls, value):
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("Voice cannot be empty")
            return value
        if not value:
            raise ValueError("At least one voice file is required")
        for encoded_file in value:
            try:
                base64.b64decode(encoded_file, validate=True)
            except Exception as exc:
                raise ValueError("Invalid base64 encoding in voice file") from exc
        return value

    def to_tts_request(self, backend: str = "vllm") -> TTSRequest:
        voice_name = None
        if isinstance(self.voice, str):
            if backend == "mlx":
                speaker_data_list = None
                # ``alloy`` is the OpenAI schema default, not a universal MLX
                # speaker. Let the server/model default apply unless the client
                # explicitly supplies another MLX voice name.
                voice_name = (
                    None if self.voice.lower() == "alloy" else self.voice
                )
            else:
                speaker_data_list = _resolve_vllm_voice(self.voice)
        else:
            speaker_data_list = [base64.b64decode(item) for item in self.voice]

        return TTSRequest(
            text=self.input,
            stream=False,
            speaker_files=speaker_data_list,
            enhance_speech=self.enhance_speech,
            language=self.language,
            max_ref_length=self.max_ref_length,
            gpt_cond_len=self.gpt_cond_len,
            gpt_cond_chunk_len=self.gpt_cond_chunk_len,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
            do_sample=self.do_sample,
            apply_novasr=self.apply_novasr,
            voice=voice_name,
            ref_text=self.ref_text,
            instruct=self.instruct,
            speed=self.speed,
            max_tokens=self.max_tokens,
        )
