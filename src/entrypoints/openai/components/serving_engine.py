from dataclasses import dataclass
from typing import Literal, Optional, Union, AsyncGenerator, BinaryIO
from auralis.common.definitions.requests import TTSRequest
from auralis.common.definitions.output import TTSOutput
import re
import io
import aiofiles
from fastapi.responses import StreamingResponse
import asyncio
import numpy as np
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor
import tempfile
import os
from pathlib import Path
import uuid
import ffmpeg


@dataclass
class OpenAITTSRequest:
    model: str
    input: str
    voice: BinaryIO
    response_format: Literal['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm'] = 'mp3'
    speed: float = 1.0

    def __post_init__(self):
        if len(self.input) > 4096:
            raise ValueError("Input text exceeds maximum length of 4096 characters")
        if not (0.25 <= self.speed <= 4.0):
            raise ValueError("Speed must be between 0.25 and 4.0")


class AudioProcessor:
    """Handle audio processing with efficient buffering and conversion."""

    def __init__(self, format: str, speed: float, temp_dir: str):
        self.format = format
        self.speed = speed
        self.temp_dir = temp_dir
        self.temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")
        self.processed_path = os.path.join(temp_dir, f"{uuid.uuid4()}.{format}")
        self.buffer = io.BytesIO()
        self.is_first_chunk = True
        self.ffmpeg_process = None

    async def process_chunk(self, chunk: bytes) -> None:
        """Add chunk to buffer."""
        self.buffer.write(chunk)

    async def finalize(self) -> AsyncGenerator[bytes, None]:
        """Process complete audio and stream it."""
        try:
            # Write buffered data to temporary WAV
            self.buffer.seek(0)
            with open(self.temp_path, 'wb') as f:
                f.write(self.buffer.getvalue())

            # Setup ffmpeg command for processing
            stream = ffmpeg.input(self.temp_path)

            # Apply speed adjustment if needed
            if self.speed != 1.0:
                stream = stream.filter('atempo', self.speed)

            # Setup output format
            stream = stream.output(
                self.processed_path,
                acodec='libmp3lame' if self.format == 'mp3' else self.format,
                audio_bitrate='192k',
                **self._get_format_options()
            )

            # Run ffmpeg processing
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: stream.overwrite_output().run(capture_stdout=True, capture_stderr=True)
            )

            # Stream processed file in chunks
            chunk_size = 8192  # Optimal chunk size for streaming
            async with aiofiles.open(self.processed_path, 'rb') as f:
                while True:
                    chunk = await f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

        finally:
            # Cleanup
            self._cleanup()

    def _get_format_options(self) -> dict:
        """Get format-specific ffmpeg options."""
        options = {
            'mp3': {'acodec': 'libmp3lame', 'audio_bitrate': '192k'},
            'opus': {'acodec': 'libopus', 'audio_bitrate': '128k'},
            'aac': {'acodec': 'aac', 'audio_bitrate': '192k'},
            'flac': {'acodec': 'flac'},
            'wav': {'acodec': 'pcm_s16le'},
            'pcm': {'acodec': 'pcm_s16le', 'f': 'raw'}
        }
        return options.get(self.format, {})

    def _cleanup(self):
        """Clean up temporary files."""
        for path in [self.temp_path, self.processed_path]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass


class OpenAICompatibleTTS:
    def __init__(self, tts_engine: 'TTS'):
        self.tts_engine = tts_engine
        self._validate_engine_compatibility()
        self.temp_dir = tempfile.mkdtemp(prefix="tts_")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

    async def cleanup(self):
        """Cleanup temporary directory."""
        try:
            if os.path.exists(self.temp_dir):
                for file in os.listdir(self.temp_dir):
                    os.remove(os.path.join(self.temp_dir, file))
                os.rmdir(self.temp_dir)
        except Exception:
            pass

    def _validate_engine_compatibility(self):
        if not hasattr(self.tts_engine, 'tts_engine') or self.tts_engine.tts_engine is None:
            raise ValueError("TTS engine not properly initialized")
        if not hasattr(self.tts_engine.tts_engine, 'conditioning_config'):
            raise ValueError("TTS engine missing conditioning_config")

    def _validate_and_clean_text(self, text: str) -> str:
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        text = text.strip()
        if not text:
            raise ValueError("Input text cannot be empty")
        return text

    async def _read_voice_data(self, voice: BinaryIO) -> bytes:
        if isinstance(voice, (bytes, bytearray)):
            return voice
        elif isinstance(voice, io.BytesIO):
            return voice.read()
        elif hasattr(voice, 'read'):
            return await voice.read()
        raise ValueError("Unsupported voice data type")

    async def _create_tts_request(self, openai_request: OpenAITTSRequest) -> TTSRequest:
        cleaned_text = self._validate_and_clean_text(openai_request.input)
        voice_data = await self._read_voice_data(openai_request.voice)

        return TTSRequest(
            text=cleaned_text,
            voice_data=voice_data,
            stream=True,
            output_format='wav',  # Always use WAV initially for best quality
            speed_factor=1.0  # Handle speed in post-processing
        )

    async def _stream_audio_response(self,
                                     audio_stream: AsyncGenerator[TTSOutput, None],
                                     speed: float,
                                     format: str) -> AsyncGenerator[bytes, None]:
        """Stream audio with efficient processing."""
        processor = AudioProcessor(format, speed, self.temp_dir)

        # Collect and process chunks
        async for chunk in audio_stream:
            await processor.process_chunk(chunk.audio_data)

        # Stream processed audio
        async for processed_chunk in processor.finalize():
            yield processed_chunk

    async def generate_speech(self,
                              model: str,
                              input_text: str,
                              voice: BinaryIO,
                              response_format: str = 'mp3',
                              speed: float = 1.0) -> StreamingResponse:
        request = OpenAITTSRequest(
            model=model,
            input=input_text,
            voice=voice,
            response_format=response_format,
            speed=speed
        )

        tts_request = await self._create_tts_request(request)
        audio_stream = await self.tts_engine.generate_speech_async(tts_request)

        return StreamingResponse(
            self._stream_audio_response(
                audio_stream,
                speed=speed,
                format=response_format
            ),
            media_type=f"audio/{response_format}",
            headers={
                "Content-Type": f"audio/{response_format}",
                "Transfer-Encoding": "chunked"
            }
        )

