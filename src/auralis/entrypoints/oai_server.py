"""OpenAI-compatible Auralis server with optional vLLM and MLX backends."""

from __future__ import annotations

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
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from auralis.common.definitions.openai import (
    AudioSpeechGenerationRequest,
    VoiceChatCompletionRequest,
)
from auralis.core.tts import TTS

DEFAULT_VLLM_MODEL = "AstraMindAI/xttsv2"
DEFAULT_VLLM_GPT_MODEL = "AstraMindAI/xtts2-gpt"
DEFAULT_MLX_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"

# Lazy global state. The model is released after an inactivity timeout so the
# process can coexist with other GPU/Metal workloads.
tts_engine: Optional[TTS] = None
last_activity_time: float = 0.0
cleanup_task: Optional[asyncio.Task] = None
init_lock: asyncio.Lock = asyncio.Lock()
shutdown_lock: asyncio.Lock = asyncio.Lock()
initializing: bool = False
shutting_down: bool = False
active_model_name: Optional[str] = None

INACTIVITY_TIMEOUT = 300
CLEANUP_CHECK_INTERVAL = 30
server_args = None

logger_str_to_logging = {
    "info": logging.INFO,
    "warn": logging.WARNING,
    "err": logging.ERROR,
}


def _default_args() -> argparse.Namespace:
    return argparse.Namespace(
        backend="auto",
        model=None,
        gpt_model=DEFAULT_VLLM_GPT_MODEL,
        max_concurrency=1,
        device="auto",
        gpu_memory_utilization=0.65,
        cpu_offload_gb=0.0,
        swap_space=0.0,
        vllm_logging_level="warn",
        mlx_voice="Chelsie",
        mlx_ref_text=None,
        mlx_instruct=None,
        mlx_max_tokens=1200,
        mlx_lazy=False,
    )


async def ensure_tts_engine(args=None) -> TTS:
    """Create the selected engine on first use, with concurrent init protection."""

    global tts_engine, last_activity_time, initializing, active_model_name

    if tts_engine is not None:
        return tts_engine

    async with init_lock:
        if tts_engine is not None:
            return tts_engine

        if initializing:
            while initializing:
                await asyncio.sleep(0.01)
            if tts_engine is None:
                raise RuntimeError("TTS initialization failed in another request")
            return tts_engine

        initializing = True
        try:
            args = args or server_args or _default_args()
            logging_level = logger_str_to_logging.get(
                args.vllm_logging_level, logging.WARNING
            )

            def create_tts_engine() -> tuple[TTS, str]:
                engine = TTS(
                    scheduler_max_concurrency=max(1, args.max_concurrency),
                    vllm_logging_level=logging_level,
                    backend=args.backend,
                )

                if engine.backend_name == "mlx":
                    model_name = args.model or DEFAULT_MLX_MODEL
                    loaded = engine.from_pretrained(
                        model_name,
                        voice=args.mlx_voice,
                        ref_text=args.mlx_ref_text,
                        instruct=args.mlx_instruct,
                        max_tokens=args.mlx_max_tokens,
                        lazy=args.mlx_lazy,
                    )
                else:
                    model_name = args.model or DEFAULT_VLLM_MODEL
                    scheduler_concurrency = (
                        1 if args.device == "cpu" else max(1, args.max_concurrency)
                    )
                    loaded = engine.from_pretrained(
                        model_name,
                        gpt_model=args.gpt_model,
                        device_map=args.device,
                        max_concurrency=scheduler_concurrency,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                        cpu_offload_gb=args.cpu_offload_gb,
                        swap_space=args.swap_space,
                    )
                return loaded, model_name

            logging.info("Initializing Auralis TTS backend lazily...")
            tts_engine, active_model_name = await asyncio.to_thread(create_tts_engine)
            last_activity_time = time.time()
            logging.info(
                "Auralis initialized: backend=%s model=%s",
                tts_engine.backend_name,
                active_model_name,
            )
            return tts_engine
        finally:
            initializing = False


async def shutdown_tts_engine() -> None:
    """Release the active model and backend caches."""

    global tts_engine, last_activity_time, shutting_down, active_model_name

    async with shutdown_lock:
        if tts_engine is None or shutting_down:
            return

        shutting_down = True
        engine = tts_engine
        try:
            logging.info("Shutting down inactive Auralis engine...")
            await engine.shutdown()
        except Exception:
            logging.exception("Error while shutting down the TTS engine")
        finally:
            tts_engine = None
            active_model_name = None
            last_activity_time = 0.0
            shutting_down = False


async def cleanup_inactive_engine() -> None:
    if tts_engine is None or last_activity_time == 0:
        return
    if time.time() - last_activity_time > INACTIVITY_TIMEOUT:
        await shutdown_tts_engine()


async def cleanup_worker() -> None:
    while True:
        try:
            await cleanup_inactive_engine()
        except Exception:
            logging.exception("Error in Auralis inactivity cleanup")
        await asyncio.sleep(CLEANUP_CHECK_INTERVAL)


@asynccontextmanager
async def lifecycle_manager(app: FastAPI):
    del app
    global cleanup_task

    cleanup_task = asyncio.create_task(cleanup_worker())
    try:
        yield
    finally:
        if cleanup_task:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass
            cleanup_task = None
        await shutdown_tts_engine()


app = FastAPI(
    title="Auralis Enhanced",
    description="OpenAI-compatible TTS with optional CUDA/vLLM and Apple MLX backends",
    lifespan=lifecycle_manager,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    current_time = time.time()
    elapsed = current_time - last_activity_time if last_activity_time else 0.0
    remaining = max(0.0, INACTIVITY_TIMEOUT - elapsed) if tts_engine else 0.0
    configured_backend = getattr(server_args, "backend", "auto")

    return {
        "status": "healthy",
        "tts_engine": {
            "initialized": tts_engine is not None,
            "status": "active" if tts_engine else "inactive",
            "backend": tts_engine.backend_name if tts_engine else configured_backend,
            "model": active_model_name,
            "time_since_last_activity_seconds": round(elapsed, 1),
            "time_until_shutdown_seconds": round(remaining, 1) if tts_engine else None,
            "initializing": initializing,
            "shutting_down": shutting_down,
        },
        "server": {
            "inactivity_timeout_seconds": INACTIVITY_TIMEOUT,
            "cleanup_check_interval_seconds": CLEANUP_CHECK_INTERVAL,
        },
    }


@app.post("/v1/audio/speech")
async def generate_audio(request: AudioSpeechGenerationRequest):
    global last_activity_time

    try:
        tts = await ensure_tts_engine()
        last_activity_time = time.time()
        tts_request = request.to_tts_request(backend=tts.backend_name)
        output = await tts.generate_speech_async(tts_request)
        last_activity_time = time.time()

        if request.speed != 1.0 and tts.backend_name != "mlx":
            output = output.change_speed(request.speed)

        return Response(
            content=output.to_bytes(request.response_format),
            media_type=f"audio/{request.response_format}",
        )
    except Exception as exc:
        logging.exception("Audio generation failed")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error generating audio: {exc}"},
        )


@app.post("/v1/chat/completions")
async def chat_completions(
    request: VoiceChatCompletionRequest,
    authorization: Optional[str] = Header(None),
):
    """Proxy an OpenAI-compatible text stream and optionally vocalize chunks."""

    global last_activity_time

    if not authorization or not authorization.startswith("Bearer "):
        return JSONResponse(
            status_code=400,
            content={"error": "Authorization header with Bearer token is required"},
        )

    try:
        tts = await ensure_tts_engine()
        last_activity_time = time.time()
        openai_api_key = authorization[len("Bearer ") :]
        modalities = request.modalities
        vocalize_every = request.vocalize_at_every_n_words
        tts_request = request.to_tts_request(text="")
        openai_request_data = request.to_openai_request()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}",
        }
        request_id = uuid.uuid4().hex

        async def stream_generator():
            accumulated_content = ""
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        request.openai_api_url,
                        json=openai_request_data,
                        headers=headers,
                    ) as response:
                        if response.status != 200:
                            error_response = await response.text()
                            raise HTTPException(
                                status_code=response.status, detail=error_response
                            )

                        async for raw_line in response.content:
                            if not raw_line:
                                continue
                            line = raw_line.decode("utf-8").strip()
                            if not line.startswith("data:"):
                                continue
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                break

                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            content = (
                                data.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            if content:
                                accumulated_content += content
                                if "text" in modalities:
                                    yield f"data: {json.dumps(data)}\n\n"

                                if len(accumulated_content.split()) >= vocalize_every:
                                    if "audio" in modalities:
                                        tts_request.text = accumulated_content
                                        tts_request.infer_language()
                                        audio_output = await tts.generate_speech_async(
                                            tts_request
                                        )
                                        audio_base64 = base64.b64encode(
                                            audio_output.to_bytes()
                                        ).decode("utf-8")
                                        payload = {
                                            "id": request_id,
                                            "object": "audio.chunk",
                                            "data": audio_base64,
                                        }
                                        yield f"data: {json.dumps(payload)}\n\n"
                                    accumulated_content = ""
                            elif "text" in modalities:
                                yield f"data: {json.dumps(data)}\n\n"

                if accumulated_content and "audio" in modalities:
                    tts_request.text = accumulated_content
                    tts_request.infer_language()
                    audio_output = await tts.generate_speech_async(tts_request)
                    audio_base64 = base64.b64encode(
                        audio_output.to_bytes()
                    ).decode("utf-8")
                    payload = {
                        "id": request_id,
                        "object": "audio.chunk",
                        "data": audio_base64,
                    }
                    yield f"data: {json.dumps(payload)}\n\n"

                if "text" in modalities:
                    completion = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "choices": [
                            {"delta": {}, "index": 0, "finish_reason": "stop"}
                        ],
                    }
                    yield f"data: {json.dumps(completion)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                logging.exception("Chat/audio streaming failed")
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    except Exception as exc:
        logging.exception("Chat completion setup failed")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error in chat completions: {exc}"},
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Auralis Enhanced OpenAI-compatible TTS server"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9950)
    parser.add_argument(
        "--backend",
        choices=["auto", "vllm", "cuda", "mlx"],
        default="auto",
        help="Inference backend. auto chooses MLX on Apple Silicon and vLLM elsewhere.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model path or Hugging Face repo. Defaults to XTTS for vLLM and an "
            "8-bit Qwen3-TTS model for MLX."
        ),
    )
    parser.add_argument("--gpt_model", default=DEFAULT_VLLM_GPT_MODEL)
    parser.add_argument("--max_concurrency", type=int, default=1)
    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], default="auto"
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.65)
    parser.add_argument("--cpu_offload_gb", type=float, default=0.0)
    parser.add_argument("--swap_space", type=float, default=0.0)
    parser.add_argument("--vllm_logging_level", default="warn")

    parser.add_argument(
        "--mlx_voice",
        default="Chelsie",
        help="Default speaker/voice for MLX models that expose named voices.",
    )
    parser.add_argument(
        "--mlx_ref_text",
        default=None,
        help="Default transcript used with MLX reference-audio voice cloning.",
    )
    parser.add_argument(
        "--mlx_instruct",
        default=None,
        help="Default style or voice-design instruction for compatible MLX models.",
    )
    parser.add_argument("--mlx_max_tokens", type=int, default=1200)
    parser.add_argument(
        "--mlx_lazy",
        action="store_true",
        help="Defer MLX parameter evaluation until first generation.",
    )

    parser.add_argument("--inactivity_timeout", type=int, default=300)
    parser.add_argument("--cleanup_interval", type=int, default=30)
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    global server_args, INACTIVITY_TIMEOUT, CLEANUP_CHECK_INTERVAL
    server_args = args
    INACTIVITY_TIMEOUT = max(0, args.inactivity_timeout)
    CLEANUP_CHECK_INTERVAL = max(1, args.cleanup_interval)

    logging.basicConfig(level=logging.INFO)
    logging.info(
        "Starting Auralis on %s:%s (backend=%s, lazy model loading enabled)",
        args.host,
        args.port,
        args.backend,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
