import argparse
import asyncio
import base64
import json
import uuid
from typing import List, Optional

import aiohttp
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from auralis import TTS, TTSRequest

app = FastAPI()

# Initialize the TTS engine
tts_engine = TTS().from_pretrained(
    "AstraMindAI/xttsv2", gpt_model="AstraMindAI/xtts2-gpt"
)


@app.post("/generate_audio")
async def generate_audio(request: Request):
    try:
        data = await request.json()
        text = data.get("text")
        speaker_files = data.get("speaker_files", None)

        # Extract auralis parameters from data
        auralis_params = {
            k.replace('auralis_', ''): v
            for k, v in data.items()
            if k.startswith('auralis_')
        }

        if not text:
            return JSONResponse(status_code=400, content={"error": "Text is required"})

        if speaker_files:
            speaker_data_list = []
            for base64_str in speaker_files:
                try:
                    content = base64.b64decode(base64_str)
                    speaker_data_list.append(content)
                except Exception as e:
                    return JSONResponse(status_code=400, content={"error": f"Invalid base64 encoding: {str(e)}"})
        else:
            return JSONResponse(status_code=400, content={"error": "Speaker files are required"})

        # Create TTSRequest with default params and auralis overrides
        tts_request = TTSRequest(
            text=text,
            speaker_files=speaker_data_list,
            stream=False,
            **auralis_params  # Unpack any auralis parameters that were provided
        )

        output = await tts_engine.generate_speech_async(tts_request)
        audio_bytes = output.to_bytes()

        if hasattr(tts_request, 'cleanup'):
            tts_request.cleanup()

        return Response(content=audio_bytes, media_type="audio/wav")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error generating audio: {str(e)}"})


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        return JSONResponse(
            status_code=400,
            content={"error": "Authorization header with Bearer token is required"}
        )

    try:
        incoming_data = await request.json()

        # Extract auralis parameters
        auralis_params = {
            k.replace('auralis_', ''): v
            for k, v in incoming_data.items()
            if k.startswith('auralis_')
        }

        # Rest of the parameters
        openai_api_key = authorization[len("Bearer "):]
        openai_api_url = incoming_data.get("openai_api_url", "https://api.openai.com/v1/chat/completions")
        incoming_data["stream"] = True
        num_of_token_to_vocalize = incoming_data.get("vocalize_at_every_n_words", 100)

        # Process speaker files
        speaker_files = incoming_data.get("speaker_files")
        if not speaker_files:
            return JSONResponse(status_code=400, content={"error": "Speaker files are required"})

        speaker_data_list = []
        for base64_str in speaker_files:
            try:
                content = base64.b64decode(base64_str)
                speaker_data_list.append(content)
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid base64 encoding: {str(e)}"})

        # Initialize TTS request with auralis parameters
        tts_request = TTSRequest(
            text="",
            language=auralis_params.get('language', "auto"),
            speaker_files=speaker_data_list,
            stream=False,
            **auralis_params  # Unpack any additional auralis parameters
        )

        # Prepare OpenAI request
        openai_request_data = {
            k: v for k, v in incoming_data.items()
            if k not in ["speaker_files", "openai_api_url", "vocalize_at_every_n_words"]
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }

        tts_request.context_partial_function = await tts_engine.prepare_for_streaming_generation(tts_request)
        request_id = uuid.uuid4().hex

        async def stream_generator():
            accumulated_content = ""

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(openai_api_url, json=openai_request_data, headers=headers) as resp:
                        if resp.status != 200:
                            error_response = await resp.text()
                            raise HTTPException(status_code=resp.status, detail=error_response)

                        async for line in resp.content:
                            if not line:
                                continue

                            line = line.decode("utf-8").strip()
                            if not line.startswith("data:"):
                                continue

                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                break

                            try:
                                data = json.loads(data_str)
                                content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")

                                if content:
                                    accumulated_content += content
                                    yield f"data: {json.dumps(data)}\n\n"

                                    if len(accumulated_content.split()) >= num_of_token_to_vocalize:
                                        # Generate audio
                                        tts_request.text = accumulated_content
                                        tts_request.infer_language()
                                        audio_output = await tts_engine.generate_speech_async(tts_request)

                                        audio_base64 = base64.b64encode(audio_output.to_bytes()).decode("utf-8")
                                        yield f"data: {json.dumps({ 'id': request_id, 'object': 'audio.chunk', 'data': audio_base64 } ) }\n\n"

                                        accumulated_content = ""
                                else:
                                    yield f"data: {json.dumps(data)}\n\n"

                            except json.JSONDecodeError:
                                continue

                # Process any remaining content
                if accumulated_content:
                    tts_request.text = accumulated_content
                    tts_request.infer_language()
                    audio_output = await tts_engine.generate_speech_async(tts_request)

                    audio_base64 = base64.b64encode(audio_output.to_bytes()).decode("utf-8")
                    yield f"data: {json.dumps({'id': request_id, 'object': 'audio.chunk', 'data': audio_base64})}\n\n"

                # Send completion messages
                yield f"data: {json.dumps({ 'id': request_id, 'object': 'chat.completion.chunk', 'choices': [{'delta': {}, 'index': 0, 'finish_reason': 'stop'}] })}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                if hasattr(tts_request, 'cleanup'):
                    tts_request.cleanup()

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error in chat completions: {str(e)}"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auralis TTS FastAPI Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    uvicorn.run(
        "auralis.openai:app",  # Make sure this matches your filename
        host=args.host,
        port=args.port,
        reload=args.reload
    )