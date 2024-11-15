from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse

from entrypoints.openai.components.serving_engine import OpenAICompatibleTTS
from auralis import TTS

app = FastAPI()



@app.post("/v1/audio/speech")
async def create_speech(
    model: str,
    input: str,
    voice: UploadFile = File(...),
    response_format: str = "mp3",
    speed: float = 1.0
):
    async with OpenAICompatibleTTS(tts_engine) as tts:
        return await tts.generate_speech(
            model=model,
            input_text=input,
            voice=voice.file,
            response_format=response_format,
            speed=speed
        )