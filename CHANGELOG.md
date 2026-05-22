# Changelog

All notable changes to Auralis Enhanced will be documented in this file.

## [2.1.0] - 2026-05-22

### Blackwell (RTX 50-series) Port

This release replaces the vLLM 0.6.4 multimodal model integration with a
V0-compatible custom GPT2 model on top of vLLM 0.9.2, so Auralis runs
natively on Blackwell (SM 12.0) GPUs such as the RTX 5080 / 5090.

### Throughput on RTX 5080 Laptop (16 GiB, SM 12.0)

| Configuration | Audio | Wall | RTF | Realtime |
| :--- | :--- | :--- | :--- | :--- |
| Serial baseline | 268 s | 125 s | 0.468x | 2.14x |
| Concurrent, conc=8, n=16 | 282 s | 27.2 s | 0.096x | 10.4x |
| Concurrent, conc=16, n=32 | 568 s | 38.4 s | 0.068x | 14.8x |
| Concurrent, conc=32, n=64 | 1103 s | 57.7 s | 0.052x | 19.1x |
| Concurrent, conc=64, n=64 | 1103 s | 48.9 s | **0.044x** | **22.6x** |

A 10-hour audiobook now renders in approximately 26 minutes on a single
RTX 5080 Laptop.

### Added

- **Concurrent batching for the XTTS GPT.** `XttsGPT` now implements
  vLLM's `HasInnerState` protocol; the V0 model runner forwards
  `request_ids_to_seq_ids` and `finished_requests_ids` into `forward`
  and the model maintains per-request `prefill_lens` and prefill hidden
  states. This lifts the single-sequence restriction from the initial port
  and restores vLLM's continuous batching.
- **Throughput benchmark.** New `scripts/benchmark.py` measures serial vs.
  concurrent RTF for any concurrency / request count combination.

### Changed

- **GPU stack.** Bumped to PyTorch 2.7.0+cu128, torchaudio 2.7.0+cu128,
  triton 3.3.0, xformers 0.0.30 (matching dependencies of vLLM 0.9.2).
  Removed the `vllm==0.6.4.post1` and `spacy==3.7.5` pins.
- **vLLM integration.** Replaced the V0 multimodal model in
  `vllm_mm_gpt.py` with a custom GPT2 variant that consumes
  `EmbedsPrompt` (`enable_prompt_embeds=True`) so conditioning latents and
  text embeddings can be pre-computed in PyTorch and submitted directly.
  Hidden states for the HiFiGAN feed are captured per request via a class
  dict instead of the removed `hidden_state_collector` hook.
- **SDPA fallback.** `PerceiverResampler` now enables flash + mem-efficient
  + math backends together, because Blackwell's flash kernel does not
  cover every q/k/v shape the perceiver feeds it.
- **Checkpoint loading.** Switched `safetensors.load_model` to
  `safetensors.load_file` + `nn.Module.load_state_dict`, sidestepping
  newer safetensors' rejection of shared-storage buffers in the HiFiGAN
  decoder's speaker encoder.

### Removed

- `src/auralis/models/xttsv2/components/vllm/hidden_state_collector.py`
  and the `ExtendedSamplingParams` wrapper in `hijack.py` — both replaced
  by per-request bookkeeping on `XttsGPT`.
- Stale `git+https://github.com/ysharma3501/FlashSR.git@6f90f8d`
  requirement (the code had already migrated to NovaSR).

### Migration

If you previously ran with the `vllm==0.6.4.post1` pin, recreate the
environment so the new GPU stack is installed:

```bash
conda create -n auralis python=3.10 -y
conda activate auralis
pip install -r requirements.txt
pip install -e .
pip install git+https://github.com/ysharma3501/NovaSR.git  # NovaSR weights
```

### Files Changed

| File | Action |
|------|--------|
| `src/auralis/models/xttsv2/components/vllm_mm_gpt.py` | REWRITTEN |
| `src/auralis/models/xttsv2/XTTSv2.py` | MODIFIED |
| `src/auralis/models/xttsv2/components/tts/layers/xtts/perceiver_encoder.py` | MODIFIED |
| `src/auralis/models/xttsv2/components/vllm/hijack.py` | TRIMMED |
| `src/auralis/models/xttsv2/components/vllm/hidden_state_collector.py` | DELETED |
| `setup.py`, `requirements.txt` | MODIFIED |
| `scripts/benchmark.py` | NEW |
| `INSTALL.md`, `README.md`, `CHANGELOG.md` | UPDATED |

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
