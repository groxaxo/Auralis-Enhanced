# Performance Bottlenecks (RTX 3090 / Ampere)

This page tracks bottlenecks observed in the current inference pipeline and optimization directions that preserve output quality and avoid increasing memory usage.

## Bottlenecks identified

1. **Phase-1 text/token generation latency (VLLM GPT path)**
   - Location: `src/auralis/models/xttsv2/components/vllm_mm_gpt.py` and `src/auralis/models/xttsv2/XTTSv2.py`
   - Impact: Dominates time-to-first-audio for long prompts.

2. **Phase-2 token→wave decode cost (XTTS decoder / vocoder path)**
   - Location: `src/auralis/models/xttsv2/components/tts/layers/xtts/`
   - Impact: Main throughput limiter under sustained load.

3. **Scheduler polling delay (`sleep(0.01)`) during ordered output draining**
   - Location: `src/auralis/common/scheduling/two_phase_scheduler.py`
   - Impact: Adds avoidable CPU-side scheduling overhead and micro-latency.

4. **Per-chunk synchronization object allocations in scheduler output buffers**
   - Location: `src/auralis/common/scheduling/two_phase_scheduler.py`
   - Impact: Extra Python object creation on every generated chunk.
   - **Status:** optimized by storing chunks directly (no per-chunk `asyncio.Event` allocation/wait).

5. **Insufficient per-stage visibility during request execution**
   - Location: `src/auralis/common/scheduling/two_phase_scheduler.py`
   - Impact: Harder to isolate whether slowdowns come from phase 1 or phase 2 in production traces.
   - **Status:** request logs now include `total`, `phase1`, and `phase2` durations for bottleneck attribution.

6. **Speaker conditioning preparation for cloning requests**
   - Location: `src/auralis/core/tts.py` (`prepare_for_streaming_generation`, `_prepare_generation_context`)
   - Impact: Added front-loaded latency when speaker embeddings and GPT-like conditioning are both enabled.
   - **Status:** optimized with LRU caching (100 entries by default) to avoid redundant I/O and computation for repeated speakers without pinning cached audio on GPU.

7. **Cross-phase handoff pressure (parallel input materialization)**
   - Location: `src/auralis/core/tts.py` (`parallel_inputs` construction)
   - Impact: Python-side orchestration overhead increases with request fan-out.

8. **Repeated CPU transfers of mel_stats in mel-spectrogram computation**
   - Location: `src/auralis/models/xttsv2/XTTSv2.py` (lines 443, 462)
   - Impact: Thousands of unnecessary PCIe transfers per inference on long audio, blocking GPU and CPU.
   - **Status:** FIXED - mel_stats now stays on GPU, passed directly to transform with explicit device parameter.

9. **Scheduler wait timeout inefficiency**
   - Location: `src/auralis/common/scheduling/two_phase_scheduler.py` (line 357)
   - Impact: Default 30-second wait timeout could cause stalls in edge cases.
   - **Status:** FIXED - reduced default timeout from 30s to 5s for faster recovery while respecting explicit timeout settings.

10. **Error handling delays in scheduler**
    - Location: `src/auralis/common/scheduling/two_phase_scheduler.py` (line 104)
    - Impact: A fixed 1-second retry delay slowed transient recovery while creating trade-offs under persistent failures.
    - **Status:** FIXED - replaced with capped exponential backoff starting at 0.1s.

## 3090-specific optimization priorities (no quality/resource increase)

1. **Keep GPU saturated with existing concurrency budget**
   Tune only existing `scheduler_max_concurrency` to the best no-OOM point (usually small increments around the current default).

2. **Reduce CPU orchestration overhead before model changes**
   Prioritize low-risk scheduler/path overhead reductions (like the per-chunk event removal already applied).

3. **Use existing metrics hooks to validate changes**
   Track `tokens/s`, `req/s`, and `ms per second of audio` from `auralis.common.metrics.performance` and keep quality checks unchanged.

4. **Optimize cloning path usage pattern**
   Reuse prepared conditioning context when possible for repeated same-speaker requests to avoid recomputation.
   - **Enhancement:** Use `clear_speaker_embedding_cache()` method to manually clear cache when switching voice sets or managing memory.

## Recent Optimizations (Latest)

### mel_stats GPU Caching
- **Before:** `mel_stats.cpu()` called repeatedly in loops during mel-spectrogram computation
- **After:** `mel_stats` stays on GPU, passed with explicit `device=self.device` parameter
- **Impact:** Eliminates thousands of PCIe transfers per inference, reducing GPU/CPU blocking
- **Files:** `src/auralis/models/xttsv2/XTTSv2.py`

### Speaker Embedding Caching
- **Before:** Every speaker file loaded from disk, resampled, and computed fresh embeddings
- **After:** LRU cache (100 entries by default) stores computed embeddings plus CPU audio keyed by (file_path, load_sr, max_ref_length, sound_norm_refs, librosa_trim_db)
- **Impact:** Eliminates redundant I/O and embedding computation for repeated speaker files without retaining per-entry audio on GPU
- **API:** Use `engine.clear_speaker_embedding_cache()` to manually clear when needed
- **Files:** `src/auralis/models/xttsv2/XTTSv2.py`

### Scheduler Timeout Optimization
- **Before:** Default 30-second wait timeout in event-based progress signaling
- **After:** Reduced to 5-second default (still respects explicit `request_timeout` if set)
- **Impact:** Faster recovery in edge case stalls without affecting normal operation
- **Files:** `src/auralis/common/scheduling/two_phase_scheduler.py`

### Error Recovery Speed
- **Before:** 1-second sleep on queue processing errors
- **After:** capped exponential backoff starting at 0.1s for faster transient recovery without retry spam under persistent failures
- **Impact:** Fast transient recovery while reducing repeated error churn
- **Files:** `src/auralis/common/scheduling/two_phase_scheduler.py`
