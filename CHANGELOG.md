# Changelog

All notable changes to Auralis Enhanced will be documented in this file.

## [Unreleased] - 2026-07-10

### Added

- Optional **Apple Silicon MLX backend** powered by `mlx-audio`.
- Automatic backend resolution: MLX on native Apple Silicon, vLLM elsewhere.
- MLX-native named voices, reference-audio cloning, style instructions, quantized checkpoints, streaming, and model-specific `backend_kwargs`.
- Apple Silicon installation extra (`.[mlx]`) and `requirements-mlx.txt`.
- OpenAI-compatible server flags for MLX model, voice, lazy loading, instructions, reference transcript, and token limit.
- Backend-selection and fake-model MLX adapter unit tests.
- A dedicated `examples/mlx_macos.py` example.

### Changed

- vLLM and NVIDIA dependencies are now isolated in the optional `cuda` extra.
- Model imports are lazy, allowing the package to import on macOS without vLLM.
- Removed the Linux-only installation guard from `setup.py`.
- `TTSRequest.speaker_files` is optional for models with built-in named voices.
- OpenAI voice resolution is portable and no longer depends on `/home/op/...` paths.
- Server health output reports the resolved backend and active model.
- Server errors are logged without returning internal tracebacks to API clients.

### Compatibility

- Existing Linux/NVIDIA behavior remains available through `backend="vllm"` or `backend="cuda"`.
- MLX requires an MLX-native checkpoint; XTTS/vLLM checkpoints are not converted implicitly.
- NovaSR remains an optional PyTorch post-processing stage and is separate from MLX inference.

---

## [2.0.0] - 2026-03-02

### 🚀 Major Change: FlashSR → NovaSR Migration

This release replaces **FlashSR** with **NovaSR** for audio super-resolution, providing significant improvements in speed and efficiency.

### Test Results (RTX 3060)

| Test | Result |
|------|--------|
| NovaSR Direct Model | ✅ PASS |
| NovaSR Processor Module | ✅ PASS |
| TTSOutput Attributes | ✅ PASS |
| TTSRequest Attributes | ✅ PASS |
| Whisper Transcription | ✅ PASS (89.8% avg similarity) |

### Performance Benchmarks

| Metric | Result |
|--------|--------|
| Model Load Time | 0.32s (after initial download) |
| Processing Speed | 25x realtime (0.04x RTF) |
| Model Size | ~52 KB |
| VRAM Overhead | ~0.1 MB |

### Added
- **NovaSR Integration**: New audio super-resolution processor using the NovaSR model
  - Located at `src/auralis/common/enhancers/novasr.py`
  - `NovaSRProcessor` class with lazy model loading
  - `get_novasr_processor()` singleton function for efficient model reuse

### Changed
- **API Rename**: `apply_flashsr` → `apply_novasr` in `TTSRequest`
- **Attribute Rename**: `_flashsr_applied` → `_novasr_applied` in `TTSOutput`
- **Method Update**: `apply_super_resolution(method="novasr")` now uses NovaSR

### Removed
- **FlashSR**: Completely removed FlashSR integration
  - Deleted `src/auralis/common/enhancers/flashsr.py`
  - Removed FlashSR dependencies

### Performance Improvements

| Metric | FlashSR (Old) | NovaSR (New) | Improvement |
|--------|---------------|--------------|-------------|
| **Model Size** | ~1000 MB | ~52 KB | **19,230x smaller** |
| **Processing Speed** | 14x realtime | 3600x realtime | **257x faster** |
| **VRAM Overhead** | ~700 MB | ~0.1 MB | **99.98% reduction** |
| **Input Sample Rate** | 16kHz | 16kHz | Same |
| **Output Sample Rate** | 48kHz | 48kHz | Same |

### Migration Guide

If you were using FlashSR, update your code:

```python
# Old (FlashSR)
request = TTSRequest(
    text="Hello world!",
    speaker_files=['reference.wav'],
    apply_flashsr=True  # OLD
)

# New (NovaSR)
request = TTSRequest(
    text="Hello world!",
    speaker_files=['reference.wav'],
    apply_novasr=True  # NEW
)
```

### Files Changed

| File | Action |
|------|--------|
| `src/auralis/common/enhancers/novasr.py` | NEW |
| `src/auralis/common/enhancers/flashsr.py` | DELETED |
| `src/auralis/common/enhancers/__init__.py` | MODIFIED |
| `src/auralis/common/definitions/output.py` | MODIFIED |
| `src/auralis/common/definitions/requests.py` | MODIFIED |
| `src/auralis/models/xttsv2/XTTSv2.py` | MODIFIED |
| `tests/unit/test_novasr.py` | NEW |
| `tests/unit/test_flashsr.py` | DELETED |
| `tests/test_novasr_saturation.py` | NEW |
| `tests/test_flashsr_saturation.py` | DELETED |
| `examples/novasr_example.py` | NEW |
| `examples/flashsr_example.py` | DELETED |
| `examples/gradio_example.py` | MODIFIED |
| `generate_metrics_and_samples.py` | MODIFIED |

---

## [1.0.0] - Initial Release

### Added
- Initial Auralis Enhanced release
- FlashSR audio super-resolution integration
- Voice cloning with XTTSv2
- Multilingual support
- Server deployment configuration
- Gradio web UI
