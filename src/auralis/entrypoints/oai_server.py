import argparse
import asyncio
import base64
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import aiohttp
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from auralis.core.tts import TTS
from auralis.common.definitions.openai import (
    VoiceChatCompletionRequest,
    AudioSpeechGenerationRequest,
)

# Global state for lazy initialization with inactivity timeout
tts_engine: Optional[TTS] = None
last_activity_time: float = 0.0
cleanup_task: Optional[asyncio.Task] = None
init_lock: asyncio.Lock = asyncio.Lock()
shutdown_lock: asyncio.Lock = asyncio.Lock()
initializing: bool = False
shutting_down: bool = False

# Configuration defaults
INACTIVITY_TIMEOUT = 300  # 5 minutes in seconds
CLEANUP_CHECK_INTERVAL = 30  # Check every 30 seconds

# Global arguments
server_args = None

# Mapping of logging level strings to their corresponding logging constants
logger_str_to_logging = {
    "info": logging.INFO,
    "warn": logging.WARNING,
    "err": logging.ERROR,
}


async def ensure_tts_engine(args=None) -> TTS:
    """Ensure TTS engine is initialized (lazy loading with thread safety).

    Args:
        args: Optional argparse.Namespace with model configuration

    Returns:
        Initialized TTS engine

    Raises:
        RuntimeError: If initialization fails
    """
    global tts_engine, last_activity_time, initializing

    # Fast path: engine already exists
    if tts_engine is not None:
        return tts_engine

    # Acquire lock for initialization
    async with init_lock:
        # Double-check after acquiring lock
        if tts_engine is not None:
            return tts_engine

        if initializing:
            # Wait for another task to complete initialization
            while initializing:
                await asyncio.sleep(0.01)
            return tts_engine

        initializing = True
        try:
            # Use provided args or defaults
            if args is None:
                if server_args is not None:
                    args = server_args
                else:
                    args = argparse.Namespace(
                        model="AstraMindAI/xttsv2",
                        gpt_model="AstraMindAI/xtts2-gpt",
                        max_concurrency=8,
                        vllm_logging_level="warn",
                    )

            logging_level = logger_str_to_logging.get(
                args.vllm_logging_level, logging.WARNING
            )

            # Initialize TTS engine in a thread to avoid event loop conflict
            logging.info("Initializing TTS engine (lazy loading in thread)...")

            def create_tts_engine():
                """Synchronous TTS engine creation."""
                return TTS(
                    scheduler_max_concurrency=args.max_concurrency,
                    vllm_logging_level=logging_level,
                ).from_pretrained(
                    args.model,
                    gpt_model=args.gpt_model,
                    max_concurrency=args.max_concurrency,  # Pass to model for VLLM config
                )

            # Run synchronous initialization in thread pool
            tts_engine = await asyncio.to_thread(create_tts_engine)

            # Update activity time
            last_activity_time = time.time()
            logging.info(f"TTS engine initialized successfully at {last_activity_time}")

            return tts_engine
        finally:
            initializing = False


async def shutdown_tts_engine():
    """Shutdown TTS engine and release resources."""
    global tts_engine, last_activity_time, shutting_down

    async with shutdown_lock:
        if tts_engine is None or shutting_down:
            return

        shutting_down = True
        try:
            logging.info("Shutting down TTS engine due to inactivity...")
            await tts_engine.shutdown()
            tts_engine = None
            last_activity_time = 0.0
            logging.info("TTS engine shut down successfully, VRAM released")
        except Exception as e:
            logging.error(f"Error during TTS engine shutdown: {e}")
            # Even if shutdown fails, clear the reference to allow reinitialization
            tts_engine = None
        finally:
            shutting_down = False


async def cleanup_inactive_engine():
    """Check for inactivity and shutdown TTS engine if timeout exceeded."""
    global tts_engine, last_activity_time

    if tts_engine is None or last_activity_time == 0:
        return

    current_time = time.time()
    time_since_activity = current_time - last_activity_time

    if time_since_activity > INACTIVITY_TIMEOUT:
        logging.info(
            f"TTS engine inactive for {time_since_activity:.1f}s (> {INACTIVITY_TIMEOUT}s), triggering shutdown"
        )
        await shutdown_tts_engine()


async def cleanup_worker():
    """Background task that periodically checks for inactivity."""
    while True:
        try:
            await cleanup_inactive_engine()
        except Exception as e:
            logging.error(f"Error in cleanup worker: {e}")

        await asyncio.sleep(CLEANUP_CHECK_INTERVAL)


@asynccontextmanager
async def lifecycle_manager(app: FastAPI):
    """FastAPI lifespan manager for startup/shutdown.

    Starts cleanup task on startup, shuts down TTS engine on shutdown.
    """
    global cleanup_task

    # Start cleanup task
    cleanup_task = asyncio.create_task(cleanup_worker())
    logging.info(
        f"Started inactivity cleanup task (timeout={INACTIVITY_TIMEOUT}s, check_interval={CLEANUP_CHECK_INTERVAL}s)"
    )

    yield  # App runs here

    # Shutdown cleanup task
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        cleanup_task = None

    # Shutdown TTS engine if exists
    await shutdown_tts_engine()


# Initialize FastAPI application
app = FastAPI(lifespan=lifecycle_manager)


@app.get("/health")
async def health_check():
    """Health check endpoint with detailed status."""
    global tts_engine, last_activity_time

    try:
        current_time = time.time()
        time_since_activity = (
            current_time - last_activity_time if last_activity_time > 0 else 0
        )
        time_until_shutdown = (
            max(0, INACTIVITY_TIMEOUT - time_since_activity) if tts_engine else 0
        )

        return {
            "status": "healthy",
            "tts_engine": {
                "initialized": tts_engine is not None,
                "status": "active" if tts_engine else "inactive",
                "time_since_last_activity": f"{time_since_activity:.1f}s",
                "time_until_shutdown": f"{time_until_shutdown:.1f}s"
                if tts_engine
                else "N/A",
                "initializing": initializing,
                "shutting_down": shutting_down,
            },
            "server": {
                "inactivity_timeout": f"{INACTIVITY_TIMEOUT}s",
                "cleanup_check_interval": f"{CLEANUP_CHECK_INTERVAL}s",
            },
        }
    except NameError as e:
        return {
            "status": "error",
            "error": f"NameError: {e}",
            "message": "Global variables not found in scope",
        }


def start_tts_engine(args, logging_level):
    """Initialize the Text-to-Speech engine with specified parameters

    Args:
        args: Parsed command line arguments containing model configurations
        logging_level: Logging level for the TTS engine
    """
    global tts_engine
    tts_engine = TTS(
        scheduler_max_concurrency=args.max_concurrency, vllm_logging_level=logging_level
    ).from_pretrained(args.model, gpt_model=args.gpt_model)


@app.post("/v1/audio/speech")
async def generate_audio(request: AudioSpeechGenerationRequest):
    """Generate audio from text using the TTS engine

    Args:
        request: Audio speech generation request containing text and parameters

    Returns:
        Response containing generated audio in requested format

    Raises:
        HTTPException: If TTS engine is not initialized or generation fails
    """
    global last_activity_time

    try:
        # Lazy initialize TTS engine if needed (uses global server_args)
        tts = await ensure_tts_engine()

        # Update activity time
        last_activity_time = time.time()

        print(f"DEBUG: tts_engine = {tts}, type = {type(tts)}")
        print("DEBUG: Creating TTS request...")

        # Create TTSRequest with default params and auralis overrides
        tts_request = request.to_tts_request()
        print("DEBUG: TTS request created successfully")

        # Generate speech and adjust speed
        output = await tts.generate_speech_async(tts_request)
        if request.speed != 1.0:
            output.change_speed(request.speed)
        audio_bytes = output.to_bytes(request.response_format)

        return Response(
            content=audio_bytes, media_type=f"audio/{request.response_format}"
        )

    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        print(f"ERROR in generate_audio: {e}")
        print(f"Traceback:\n{tb}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error generating audio: {str(e)}", "traceback": tb},
        )


@app.post("/v1/chat/completions")
async def chat_completions(
    request: VoiceChatCompletionRequest, authorization: Optional[str] = Header(None)
):
    """Handle chat completions with optional audio generation

    Args:
        request: Voice chat completion request containing prompt and parameters
        authorization: Bearer token for API authentication

    Returns:
        StreamingResponse containing text and/or audio chunks

    Raises:
        HTTPException: If TTS engine is not initialized or request fails
    """
    global last_activity_time

    # Lazy initialize TTS engine if needed (uses global server_args)
    tts = await ensure_tts_engine()

    # Update activity time
    last_activity_time = time.time()

    # Validate authorization header
    if not authorization or not authorization.startswith("Bearer "):
        return JSONResponse(
            status_code=400,
            content={"error": "Authorization header with Bearer token is required"},
        )
    try:
        # Extract request parameters
        openai_api_key = authorization[len("Bearer ") :]
        modalities = request.modalities
        num_of_token_to_vocalize = request.vocalize_at_every_n_words

        # Initialize TTS request with auralis parameters
        tts_request = request.to_tts_request(text="")

        # Prepare OpenAI request
        openai_request_data = request.to_openai_request()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}",
        }

        request_id = uuid.uuid4().hex

        # Validate requested modalities
        valid_modalities = ["text", "audio"]
        if not all(m in valid_modalities for m in modalities):
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Invalid modalities. Must be one or more of {valid_modalities}"
                },
            )

        async def stream_generator():
            """Generator function for streaming text and audio responses

            Yields:
                JSON-formatted strings containing text chunks and/or audio data
            """
            accumulated_content = ""

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        request.openai_api_url,
                        json=openai_request_data,
                        headers=headers,
                    ) as resp:
                        if resp.status != 200:
                            error_response = await resp.text()
                            raise HTTPException(
                                status_code=resp.status, detail=error_response
                            )

                        # Process streaming response line by line
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
                                content = (
                                    data.get("choices", [{}])[0]
                                    .get("delta", {})
                                    .get("content", "")
                                )

                                if content:
                                    # Accumulate content and handle text/audio generation
                                    accumulated_content += content
                                    # Stream text if requested
                                    if "text" in modalities:
                                        yield f"data: {json.dumps(data)}\n\n"

                                    # Generate audio when word threshold is reached
                                    if (
                                        len(accumulated_content.split())
                                        >= num_of_token_to_vocalize
                                    ):
                                        if "audio" in modalities:
                                            tts_request.text = accumulated_content
                                            tts_request.infer_language()
                                            audio_output = (
                                                await tts.generate_speech_async(
                                                    tts_request
                                                )
                                            )
                                            audio_base64 = base64.b64encode(
                                                audio_output.to_bytes()
                                            ).decode("utf-8")
                                            yield f"data: {json.dumps({'id': request_id, 'object': 'audio.chunk', 'data': audio_base64})}\n\n"

                                        accumulated_content = ""
                                elif "text" in modalities:
                                    # Stream non-content text events
                                    yield f"data: {json.dumps(data)}\n\n"

                            except json.JSONDecodeError:
                                continue

                # Process any remaining content for audio
                if accumulated_content and "audio" in modalities:
                    tts_request.text = accumulated_content
                    tts_request.infer_language()
                    audio_output = await tts.generate_speech_async(tts_request)
                    audio_base64 = base64.b64encode(audio_output.to_bytes()).decode(
                        "utf-8"
                    )
                    yield f"data: {json.dumps({'id': request_id, 'object': 'audio.chunk', 'data': audio_base64})}\n\n"

                # Send completion messages for text modality
                if "text" in modalities:
                    yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'choices': [{'delta': {}, 'index': 0, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                pass

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Error in chat completions: {str(e)}"}
        )


def main():
    """Main function to configure and start the FastAPI server"""
    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        description="Auralis TTS FastAPI Server with inactivity timeout"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the server on"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9950,
        help="Port to run the server on (default: 9950)",
    )
    parser.add_argument(
        "--model", type=str, default="AstraMindAI/xttsv2", help="The base model to run"
    )
    parser.add_argument(
        "--gpt_model",
        type=str,
        default="AstraMindAI/xtts2-gpt",
        help="The gpt model to load alongside the base model, if present",
    )
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=8,
        help="The concurrency value that is used in the TTS Engine, it is directly connected to the memory consumption",
    )
    parser.add_argument(
        "--vllm_logging_level",
        type=str,
        default="warn",
        help="The vllm logging level, could be one of [info | warn | err]",
    )
    parser.add_argument(
        "--inactivity_timeout",
        type=int,
        default=300,
        help="Seconds of inactivity before TTS engine shuts down (default: 300 = 5 minutes)",
    )
    parser.add_argument(
        "--cleanup_interval",
        type=int,
        default=30,
        help="Seconds between inactivity checks (default: 30)",
    )

    args = parser.parse_args()

    # Store args globally
    global server_args
    server_args = args

    # Update global configuration
    global INACTIVITY_TIMEOUT, CLEANUP_CHECK_INTERVAL
    INACTIVITY_TIMEOUT = args.inactivity_timeout
    CLEANUP_CHECK_INTERVAL = args.cleanup_interval

    # Log configuration
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Starting Auralis TTS server on port {args.port}")
    logging.info(
        f"Inactivity timeout: {INACTIVITY_TIMEOUT}s, Cleanup interval: {CLEANUP_CHECK_INTERVAL}s"
    )
    logging.info(
        "TTS engine will load on first request and auto-shutdown after inactivity"
    )

    # Start the FastAPI server (TTS engine loads lazily on first request)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
