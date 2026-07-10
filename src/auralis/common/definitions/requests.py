from __future__ import annotations

import functools
import hashlib
import io
import json
import uuid
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Literal, Optional, Union, get_args

import langid
import librosa
import soundfile as sf
from cachetools import LRUCache

from auralis.common.definitions.enhancer import (
    AudioPreprocessingConfig,
    EnhancedAudioProcessor,
)


def hash_params(*args, **kwargs):
    params_str = json.dumps([str(arg) for arg in args], sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()


def cached_processing(maxsize=128):
    def decorator(func):
        cache = LRUCache(maxsize=maxsize)

        @functools.wraps(func)
        def wrapper(
            self,
            audio_path: str,
            audio_config: AudioPreprocessingConfig,
            *args,
            **kwargs,
        ):
            params_dict = {"audio_path": audio_path, "config": asdict(audio_config)}
            cache_key = hash_params(params_dict)
            result = cache.get(cache_key)
            if result is not None:
                return result
            result = func(self, audio_path, audio_config, *args, **kwargs)
            cache[cache_key] = result
            return result

        return wrapper

    return decorator


SupportedLanguages = Literal[
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "pl",
    "tr",
    "ru",
    "nl",
    "cs",
    "ar",
    "zh-cn",
    "hu",
    "ko",
    "ja",
    "hi",
    "da",
    "fi",
    "sv",
    "auto",
    "",
]


@lru_cache(maxsize=1024)
def get_language(text: str):
    detected_language = langid.classify(text)[0].strip()
    if detected_language == "zh":
        detected_language = "zh-cn"
    return detected_language


def validate_language(language: str) -> SupportedLanguages:
    supported = get_args(SupportedLanguages)
    if language not in supported:
        raise ValueError(f"Language {language} not supported. Must be one of {supported}")
    return language  # type: ignore[return-value]


@dataclass
class TTSRequest:
    """Container shared by the vLLM and optional MLX TTS backends.

    Backend-neutral fields preserve the established XTTS interface. The MLX
    fields at the end are ignored by the vLLM backend and make model-specific
    controls available without creating a second public request type.
    """

    text: Union[AsyncGenerator[str, None], str, List[str]]
    speaker_files: Optional[
        Union[str, List[str], bytes, List[bytes]]
    ] = None
    context_partial_function: Optional[Callable] = None

    start_time: Optional[float] = None
    enhance_speech: bool = False
    audio_config: AudioPreprocessingConfig = field(
        default_factory=AudioPreprocessingConfig
    )
    language: SupportedLanguages = "auto"
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    load_sample_rate: int = 22050
    sound_norm_refs: bool = False

    max_ref_length: int = 60
    gpt_cond_len: int = 30
    gpt_cond_chunk_len: int = 4

    stream: bool = False
    temperature: float = 0.75
    top_p: float = 0.85
    top_k: int = 50
    repetition_penalty: float = 5.0
    length_penalty: float = 1.0
    do_sample: bool = True

    apply_novasr: bool = False

    # Optional backend-neutral controls used by MLX-native models.
    voice: Optional[str] = None
    ref_text: Optional[Union[str, List[str]]] = None
    instruct: Optional[str] = None
    speed: float = 1.0
    max_tokens: Optional[int] = 1200
    streaming_interval: float = 2.0
    backend_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.language == "auto" and isinstance(self.text, str) and self.text:
            self.language = get_language(self.text)

        validate_language(self.language)
        self.processor = EnhancedAudioProcessor(self.audio_config)
        if isinstance(self.speaker_files, list) and self.enhance_speech:
            self.speaker_files = [
                self.preprocess_audio(file, self.audio_config)
                for file in self.speaker_files
            ]

    def infer_language(self):
        if self.language == "auto" and isinstance(self.text, str) and self.text:
            self.language = get_language(self.text)

    @cached_processing()
    def preprocess_audio(
        self,
        audio_source: Union[str, bytes],
        audio_config: AudioPreprocessingConfig,
    ) -> str:
        try:
            temp_dir = Path("/tmp/auralis")
            temp_dir.mkdir(exist_ok=True)
            if isinstance(audio_source, str):
                source_path = Path(audio_source)
                audio, sample_rate = librosa.load(
                    source_path, sr=self.audio_config.sample_rate
                )
            else:
                source_path = None
                audio, sample_rate = librosa.load(
                    io.BytesIO(audio_source), sr=self.audio_config.sample_rate
                )
            processed = self.processor.process(audio)

            stem = (
                str(hash(audio_source))
                if isinstance(audio_source, bytes)
                else source_path.stem
            )
            suffix = (
                ".wav"
                if isinstance(audio_source, bytes)
                else source_path.suffix or ".wav"
            )
            output_path = temp_dir / f"{stem}-{uuid.uuid4().hex}{suffix}"
            sf.write(output_path, processed, sample_rate)
            return str(output_path)
        except Exception as exc:
            print(f"Error processing audio: {exc}. Using original file.")
            return audio_source  # type: ignore[return-value]

    def copy(self):
        copy_fields = {
            "text": self.text,
            "speaker_files": self.speaker_files,
            "context_partial_function": self.context_partial_function,
            "start_time": self.start_time,
            "enhance_speech": self.enhance_speech,
            "audio_config": self.audio_config,
            "language": self.language,
            "request_id": self.request_id,
            "load_sample_rate": self.load_sample_rate,
            "sound_norm_refs": self.sound_norm_refs,
            "max_ref_length": self.max_ref_length,
            "gpt_cond_len": self.gpt_cond_len,
            "gpt_cond_chunk_len": self.gpt_cond_chunk_len,
            "stream": self.stream,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
            "do_sample": self.do_sample,
            "apply_novasr": self.apply_novasr,
            "voice": self.voice,
            "ref_text": self.ref_text,
            "instruct": self.instruct,
            "speed": self.speed,
            "max_tokens": self.max_tokens,
            "streaming_interval": self.streaming_interval,
            "backend_kwargs": dict(self.backend_kwargs),
        }
        return TTSRequest(**copy_fields)
