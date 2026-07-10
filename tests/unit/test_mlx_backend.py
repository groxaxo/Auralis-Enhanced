from dataclasses import dataclass

import numpy as np
import pytest

from auralis.backends.mlx import MLXTTSEngine
from auralis.common.definitions.output import TTSOutput
from auralis.common.definitions.requests import TTSRequest


@dataclass
class FakeResult:
    audio: np.ndarray
    sample_rate: int = 24000
    token_count: int = 4


class FakeModel:
    sample_rate = 24000

    def __init__(self):
        self.calls = []

    def generate(
        self,
        text,
        voice=None,
        speed=1.0,
        lang_code="auto",
        ref_audio=None,
        temperature=0.7,
        stream=False,
    ):
        self.calls.append(
            {
                "text": text,
                "voice": voice,
                "speed": speed,
                "lang_code": lang_code,
                "ref_audio": ref_audio,
                "temperature": temperature,
                "stream": stream,
            }
        )
        yield FakeResult(np.array([0.1, 0.2], dtype=np.float32))
        yield FakeResult(np.array([0.3], dtype=np.float32))


def make_engine():
    model = FakeModel()
    return MLXTTSEngine(model, "fake", voice="Chelsie"), model


def test_non_streaming_generation_combines_chunks():
    engine, model = make_engine()
    request = TTSRequest(
        text="hello",
        speaker_files=None,
        language="en",
        speed=1.1,
        temperature=0.5,
    )

    output = engine.generate_speech(request)

    assert isinstance(output, TTSOutput)
    np.testing.assert_allclose(output.array, [0.1, 0.2, 0.3])
    assert output.sample_rate == 24000
    assert model.calls[0]["voice"] == "Chelsie"
    assert model.calls[0]["speed"] == 1.1
    assert "top_k" not in model.calls[0]


def test_streaming_generation_yields_auralis_outputs():
    engine, _ = make_engine()
    request = TTSRequest(
        text="hello",
        speaker_files=None,
        language="en",
        stream=True,
    )

    chunks = list(engine.generate_speech(request))

    assert [len(chunk.array) for chunk in chunks] == [2, 1]


@pytest.mark.asyncio
async def test_async_non_streaming_generation():
    engine, _ = make_engine()
    request = TTSRequest(text="hello", speaker_files=None, language="en")

    output = await engine.generate_speech_async(request)

    assert isinstance(output, TTSOutput)
    assert len(output.array) == 3
