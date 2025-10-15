# Auralis Repository Components

A concise map of the codebase: top-level modules, directories, and key classes/functions to help you navigate quickly.

## Overview
- **Purpose**: High-performance text-to-speech (TTS) engine with voice cloning, streaming, batching, enhancement, and OpenAI-compatible server.
- **Primary package**: `src/auralis/`
- **Entrypoints**: OpenAI-compatible FastAPI server and VLLM helper.

## Directory Structure
- **`src/auralis/`**
  - **`common/`**: Shared definitions, logging, metrics, scheduling utilities
  - **`core/`**: High-level TTS interface
  - **`entrypoints/`**: CLI/server entrypoints
  - **`models/`**: Model abstractions and XTTSv2 implementation
- **`docs/`**: User and developer documentation
- **`examples/`**: Usage examples (e.g., Gradio, OpenAI server client)
- **`tests/`**: Test suite

## Core Components
- **`src/auralis/core/tts.py`**
  - **`class TTS`**: Public API for TTS. Provides `from_pretrained(...)`, `generate_speech(...)`, and async/streaming variants. Orchestrates scheduling and models.

## Model Abstractions
- **`src/auralis/models/base.py`**
  - **`@dataclass class ConditioningConfig`**: Model conditioning config.
  - **`class BaseAsyncTTSEngine(torch.nn.Module)`**: Abstract interface for asynchronous two-phase generation engines.
  - **`class AudioOutputGenerator`**: Helper/protocol for generating audio output (used by engines).

- **`src/auralis/models/registry.py`**
  - Minimal registry hooks for models.

## XTTSv2 Model Implementation
- **`src/auralis/models/xttsv2/XTTSv2.py`**
  - **`class XTTSv2Engine(BaseAsyncTTSEngine)`**: Main XTTSv2 async engine implementation leveraging VLLM AsyncEngine and the layered TTS stack.

- **`src/auralis/models/xttsv2/components/`**
  - **`vllm_mm_gpt.py`**: Multimodal GPT integration utilities for VLLM-based pipelines.
  - **`vllm/`**
    - `hidden_state_collector.py`: Hidden state capture hooks.
    - `hijack.py`: VLLM behavior overrides/hijacking helpers.
  - **`tts/`**
    - `layers/xtts/`
      - `hifigan_decoder.py`: HiFi-GAN decoder components.
      - `perceiver_encoder.py`: Perceiver resampler/encoder.
      - `latent_encoder.py`: Conditioning encoder for style/voice.
      - `zh_num2words.py`: Chinese numerals to words conversion.

- **`src/auralis/models/xttsv2/config/`**
  - `xttsv2_config.py`: Model configuration dataclasses.
  - `xttsv2_gpt_config.py`: GPT-side configuration.
  - `tokenizer.py`: Tokenizer configuration/utilities.

- **`src/auralis/models/xttsv2/utils/`**
  - Checkpoint conversion and related helpers (see `docs/` for usage).

## Common Definitions and Utilities
- **`src/auralis/common/definitions/requests.py`**
  - **`@dataclass class TTSRequest`**: Unified request container supporting strings/lists/generators, speaker refs, language, streaming and generation params.

- **`src/auralis/common/definitions/output.py`**
  - **`@dataclass class TTSOutput`**: Output container with utilities
    - Conversion: `to_tensor()`, `to_bytes()`, `from_tensor()`, `from_file()`
    - Processing: `combine_outputs()`, `resample()`, `get_info()`, `change_speed()`
    - IO/Playback: `save()`, `play()`, `preview()`

- **`src/auralis/common/definitions/scheduler.py`**
  - **`enum TaskState`**: `QUEUED`, `PROCESSING_FIRST`, `PROCESSING_SECOND`, `DONE`
  - **`@dataclass class QueuedRequest`**: Queue element container.

- **`src/auralis/common/definitions/enhancer.py`**
  - **`@dataclass class AudioPreprocessingConfig`**: Preprocessing flags/params (normalize, trim, enhance, etc.).
  - **`class EnhancedAudioProcessor`**: Torch-based audio enhancement pipeline.

- **`src/auralis/common/definitions/openai.py`**
  - **`class ChatCompletionMessage(BaseModel)`**
  - **`class VoiceChatCompletionRequest(BaseModel)`**
  - **`class AudioSpeechGenerationRequest(BaseModel)`**
  - OpenAI-compatible payload schemas mapping onto `TTSRequest` defaults.

- **`src/auralis/common/logging/logger.py`**
  - **`class VLLMLogOverrider`**: Redirect/format VLLM logs.
  - **`class ColoredFormatter(logging.Formatter)`**: Rich log formatting.
  - **`def setup_logger(name=None, level=...)`**: Project logger factory.
  - **`def set_vllm_logging_level(level)`**: Adjust VLLM logging.

- **`src/auralis/common/metrics/performance.py`**
  - **`@dataclass class TTSMetricsTracker`**: Tracks performance metrics.
  - **`def track_generation(...)`**: Decorator to instrument generation.

- **`src/auralis/common/scheduling/two_phase_scheduler.py`**
  - **`class TwoPhaseScheduler`**: Two-phase async task scheduler with parallelism.

- **`src/auralis/common/utilities.py`**
  - Shared utility helpers used across modules.

## Entrypoints (Servers/CLI)
- **`src/auralis/entrypoints/oai_server.py`**
  - **`def start_tts_engine(args, logging_level)`**: Initialize TTS engine.
  - **`def main()`**: FastAPI server bootstrap, OpenAI-compatible routes for chat/audio.

- **`src/auralis/entrypoints/llm_server.py`**
  - **`def start_vllm_server()`**: Convenience launcher for `vllm serve`.

## Package Init
- **`src/auralis/__init__.py`**
  - Re-exports for user-friendly API, e.g., `TTS`, `TTSRequest`, `TTSOutput`.

## Tests and Examples
- **`tests/`**: Unit tests for core, models, and utilities.
- **`examples/`**: Scripts demonstrating sync/async usage, OpenAI server, and Gradio UI.

## Notes
- XTTSv2 components under `src/auralis/models/xttsv2/components/tts` are under the Coqui AI License (see `README.md`).
- Performance and memory characteristics are documented in `README.md`.
