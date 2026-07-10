<div align="center">

# 🌌 Auralis Enhanced
### Production text-to-speech across NVIDIA CUDA and Apple Silicon MLX

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-MLX-black.svg)](https://github.com/ml-explore/mlx)
[![NovaSR](https://img.shields.io/badge/Audio-48kHz%20NovaSR-brightgreen.svg)](https://huggingface.co/YatharthS/NovaSR)

**A local, model-flexible TTS platform with voice cloning, streaming, an OpenAI-compatible API, and optional native MLX inference for M-series Macs.**

[Apple Silicon](#apple-silicon-mlx-quick-start-) • [NVIDIA](#nvidia-cudavllm-quick-start-) • [Python API](#python-api) • [OpenAI API](#openai-compatible-server) • [Architecture](#backend-architecture)

</div>

---

## What changed

Auralis Enhanced now has an explicit backend layer:

- **`mlx`** — native Apple Silicon inference through `mlx-audio`; designed for M1 through M5 Macs.
- **`vllm`** — the established high-throughput XTTSv2 path for NVIDIA/Linux.
- **`auto`** — selects MLX on Apple Silicon and vLLM elsewhere.

MLX is an **optional dependency**. Installing the Mac backend does not install vLLM or NVIDIA packages, and installing the CUDA backend does not install MLX. The public `TTS`, `TTSRequest`, `TTSOutput`, and OpenAI-compatible API remain shared across both paths.

> [!IMPORTANT]
> MLX uses MLX-native checkpoints supported by `mlx-audio`; it does not execute the existing XTTS/vLLM checkpoint. The CUDA XTTS path remains available and unchanged through `backend="vllm"`.

## Key features

- **Apple Silicon support:** Metal-accelerated MLX inference with quantized model support.
- **NVIDIA production path:** Async XTTSv2 generation through vLLM.
- **Voice cloning:** Reference-audio conditioning on compatible models.
- **Named voices and voice design:** Available when supported by the selected MLX model.
- **Streaming and async APIs:** Synchronous, asynchronous, and streaming generation.
- **OpenAI-compatible endpoint:** `/v1/audio/speech` with the same server on either backend.
- **Portable deployment:** No machine-specific `/home/...` paths in API voice resolution.
- **Optional 48 kHz enhancement:** NovaSR remains available as a post-processing step.
- **Automatic idle cleanup:** Releases model memory after configurable inactivity.

---

## Apple Silicon MLX quick start 🍎

### Requirements

- Apple Silicon Mac (M1, M2, M3, M4, M5, or later)
- macOS
- Python 3.10 or newer; Python 3.11 is recommended

### Install

```bash
git clone https://github.com/groxaxo/Auralis-Enhanced.git
cd Auralis-Enhanced

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-mlx.txt
```

Equivalent editable install:

```bash
pip install -e ".[mlx,server,examples]"
```

### Generate speech

```python
from auralis import TTS, TTSRequest

MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"

tts = TTS(backend="mlx").from_pretrained(
    MODEL,
    voice="Chelsie",
)

request = TTSRequest(
    text="Hello from Auralis Enhanced running natively on Apple Silicon.",
    speaker_files=None,
    language="en",
)

output = tts.generate_speech(request)
output.save("output.wav")
```

`TTS(backend="auto")` resolves to MLX automatically on a native Apple Silicon Python process. Use `backend="mlx"` in production configuration when an explicit backend is preferable.

### Voice cloning on a compatible MLX model

```python
request = TTSRequest(
    text="This sentence will follow the reference speaker.",
    speaker_files=["reference.wav"],
    ref_text="The exact transcript spoken in reference.wav.",
    language="en",
    temperature=0.75,
)

output = tts.generate_speech(request)
output.save("cloned.wav")
```

Reference transcripts materially improve in-context voice cloning for models such as Qwen3-TTS Base. Model capabilities vary; named voices, reference audio, and voice-design instructions are passed only through the selected MLX model interface.

### Streaming on Mac

```python
request = TTSRequest(
    text="A longer passage that should begin playing while it is generated.",
    speaker_files=None,
    language="en",
    stream=True,
    streaming_interval=1.0,
)

for chunk in tts.generate_speech(request):
    print(chunk.get_info())
```

### MLX model controls

Common controls are first-class `TTSRequest` fields:

```python
request = TTSRequest(
    text="Speak calmly and clearly.",
    speaker_files=None,
    language="en",
    voice="Chelsie",
    instruct="A calm, warm, reassuring delivery",
    speed=0.95,
    max_tokens=1600,
    top_k=50,
    top_p=0.9,
    backend_kwargs={"min_p": 0.02},
)
```

`backend_kwargs` is the escape hatch for model-specific `mlx-audio` parameters without coupling Auralis to one model family.

---

## NVIDIA CUDA/vLLM quick start 🟢

The existing XTTSv2 backend remains the high-throughput NVIDIA/Linux option.

```bash
git clone https://github.com/groxaxo/Auralis-Enhanced.git
cd Auralis-Enhanced

conda create -n auralis_env python=3.10 -y
conda activate auralis_env
pip install -r requirements.txt
```

```python
from auralis import TTS, TTSRequest

tts = TTS(backend="vllm").from_pretrained(
    "AstraMindAI/xttsv2",
    gpt_model="AstraMindAI/xtts2-gpt",
)

request = TTSRequest(
    text="Hello from the CUDA backend.",
    speaker_files=["reference.wav"],
    language="en",
)

output = tts.generate_speech(request)
output.save("output.wav")
```

The legacy `TTS()` call still resolves to vLLM on Linux and other non-Apple-Silicon platforms.

---

## Python API

### Backend selection

```python
TTS(backend="auto")   # MLX on Apple Silicon; vLLM elsewhere
TTS(backend="mlx")    # Explicit Apple MLX backend
TTS(backend="vllm")   # Explicit NVIDIA/vLLM backend
TTS(backend="cuda")   # Alias for vLLM
```

The `AURALIS_BACKEND` environment variable can override `auto`:

```bash
export AURALIS_BACKEND=mlx
```

### Async generation

```python
output = await tts.generate_speech_async(request)
```

For streaming requests, `generate_speech_async` returns an async generator:

```python
request.stream = True
stream = await tts.generate_speech_async(request)
async for chunk in stream:
    process(chunk)
```

### Optional NovaSR enhancement

```python
request = TTSRequest(
    text="Generate and enhance this speech.",
    speaker_files=None,
    language="en",
    apply_novasr=True,
)
output = tts.generate_speech(request)
```

NovaSR is a PyTorch post-processing component, separate from the MLX inference engine. It gracefully falls back if the enhancer is unavailable on the selected platform.

---

## OpenAI-compatible server

### Run on Apple Silicon

```bash
auralis.openai \
  --backend mlx \
  --model mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit \
  --mlx_voice Chelsie \
  --host 0.0.0.0 \
  --port 8000
```

### Run on NVIDIA

```bash
auralis.openai \
  --backend vllm \
  --model AstraMindAI/xttsv2 \
  --gpt_model AstraMindAI/xtts2-gpt \
  --host 0.0.0.0 \
  --port 8000
```

### Generate through the API

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local-tts",
    "input": "Auralis now runs on Apple Silicon and NVIDIA.",
    "voice": "Chelsie",
    "response_format": "wav",
    "language": "en"
  }' \
  --output speech.wav
```

Check runtime state:

```bash
curl http://127.0.0.1:8000/health
```

The health response reports the resolved backend, loaded model, initialization state, and idle-shutdown timing. Models load lazily on the first request.

---

## Backend architecture

```text
TTS / TTSRequest / TTSOutput
              │
       backend resolver
        ┌─────┴─────┐
        │           │
  MLX adapter   vLLM scheduler
   mlx-audio      XTTSv2
        │           │
 Apple Metal    NVIDIA CUDA
```

Design rules:

1. Backend-specific libraries are imported lazily.
2. `mlx-audio` and vLLM are optional extras, never unconditional dependencies.
3. Backend selection happens once when `TTS` is constructed.
4. Model-specific MLX parameters pass through `backend_kwargs`.
5. Existing XTTS callers retain the same request and output types.

This boundary makes it possible to add future inference engines without duplicating the API or making every installation carry every accelerator stack.

---

## Docker GPU deployment

Docker deployment is intended for the NVIDIA backend. MLX requires native macOS and is not available inside a standard Linux Docker container.

```bash
docker build -t auralis-enhanced:latest .

docker run -d --name auralis-enhanced \
  --gpus all \
  -p 8000:8000 \
  auralis-enhanced:latest \
  --backend vllm --host 0.0.0.0 --port 8000
```

---

## Performance notes

The established NVIDIA benchmark scripts remain available:

```bash
python bench_speed.py
```

Performance on Apple Silicon depends heavily on the selected model, quantization, text length, streaming interval, and unified-memory capacity. No M5 benchmark is claimed until it is measured on physical M5 hardware; the implementation uses the native MLX execution path and supports quantized MLX checkpoints.

---

## Development and validation

```bash
pip install -e ".[dev,mlx]"   # Apple Silicon
# or
pip install -e ".[dev,cuda]"  # NVIDIA/Linux

pytest -q tests/unit
```

Backend-selection and MLX-adapter tests use a fake model, so their contract can be validated without downloading model weights.

---

## Acknowledgments

- [AstraMind AI](https://github.com/astramind-ai) for the original Auralis project.
- [Coqui AI](https://coqui.ai/) for XTTSv2.
- [Apple MLX](https://github.com/ml-explore/mlx) for Apple Silicon machine-learning primitives.
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) for MLX-native audio model implementations.
- [NovaSR](https://github.com/ysharma3501/NovaSR) for audio super-resolution.

---

<div align="center">

### [Give Auralis Enhanced a star](https://github.com/groxaxo/Auralis-Enhanced)

Built for private, local, accelerator-aware speech generation.

</div>
