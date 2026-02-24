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

7. **Cross-phase handoff pressure (parallel input materialization)**  
   - Location: `src/auralis/core/tts.py` (`parallel_inputs` construction)  
   - Impact: Python-side orchestration overhead increases with request fan-out.

## 3090-specific optimization priorities (no quality/resource increase)

1. **Keep GPU saturated with existing concurrency budget**  
   Tune only existing `scheduler_max_concurrency` to the best no-OOM point (usually small increments around the current default).

2. **Reduce CPU orchestration overhead before model changes**  
   Prioritize low-risk scheduler/path overhead reductions (like the per-chunk event removal already applied).

3. **Use existing metrics hooks to validate changes**  
   Track `tokens/s`, `req/s`, and `ms per second of audio` from `auralis.common.metrics.performance` and keep quality checks unchanged.

4. **Optimize cloning path usage pattern**  
   Reuse prepared conditioning context when possible for repeated same-speaker requests to avoid recomputation.
