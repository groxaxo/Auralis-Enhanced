<div align="center">

# 🌌 Auralis Enhanced
### *Production-Ready Text-to-Speech with Voice Cloning, NovaSR Audio Enhancement & Network Deployment*

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub](https://img.shields.io/badge/GitHub-Auralis--Enhanced-blue.svg)](https://github.com/akumaburn/Auralis-Enhanced-Blackwell)
[![NovaSR](https://img.shields.io/badge/Audio-48kHz%20NovaSR-brightgreen.svg)](https://huggingface.co/YatharthS/NovaSR)

**Process an entire novel in minutes, not hours. Convert books to speech with professional 48kHz audio quality powered by NovaSR — now running natively on Blackwell (RTX 50-series) GPUs.**

[Quick Start](#quick-start-) • [Blackwell Notes](#-blackwell-rtx-50-series-support) • [Benchmark](#-throughput-benchmark) • [Docker GPU Deploy](#-docker-gpu-deploy) • [Features](#key-features-) • [Performance](#-performance--benchmarks) • [Credits](#-acknowledgments)

</div>

---

## 🚀 What is Auralis Enhanced?

**Auralis Enhanced** is a production-ready fork of the original Auralis TTS engine, optimized for server deployment and high-quality audio production. It features **NovaSR audio super-resolution**, converting standard 24kHz synthesis into professional **48kHz broadcast-quality output** with negligible overhead.

### ⚡ Key Highlights
- **🎵 Professional Output**: Automatic 48kHz output via ultra-fast NovaSR (3600x realtime).
- **🎤 Voice Cloning**: Clone any voice from a short reference audio sample.
- **🌍 Multilingual**: Native support for English, Spanish, and many other languages.
- **⚙️ Production Ready**: Pre-configured for network accessibility (`0.0.0.0`) and multi-GPU setups.
- **⚡ Super Fast**: Optimized inference loop with streaming support.

---

## 📊 Performance & Benchmarks

### End-to-end throughput on RTX 5080 Laptop (Blackwell SM 12.0, 16 GiB)

Measured with `scripts/benchmark.py` on an idle GPU; the 64 short
sentences total ~11 minutes of synthesized speech.

| Configuration | Audio | Wall | RTF | Realtime |
| :--- | ---: | ---: | ---: | ---: |
| Serial baseline, n=16 | 166 s | 26.7 s | 0.161x | 6.2x |
| Concurrent, conc=8, n=16 | 161 s | 8.7 s | 0.054x | 18.6x |
| Concurrent, conc=16, n=32 | 336 s | 15.1 s | 0.045x | 22.3x |
| Concurrent, conc=32, n=64 | 628 s | 22.9 s | 0.036x | 27.5x |
| Concurrent, conc=64, n=64 | 670 s | 23.0 s | **0.034x** | **29.2x** |

> [!TIP]
> Use `python scripts/benchmark.py --conc 32 --n 64 --no-serial` to
> reproduce the throughput numbers on your own GPU.

> [!NOTE]
> Numbers updated after fixes for two latent regressions on
> vLLM 0.10's V0 engine: (1) the default `cpu_offload_gb=8.0` was
> silently offloading the entire GPT decoder to CPU and round-tripping
> per-decode-step transfers through `functional_call`, and (2) the
> decode forward path embedded the placeholder `input_ids` instead of
> using the `inputs_embeds` that vLLM precomputes from each sampled
> token under `enable_prompt_embeds=True`. Both fixes ship in commits
> [`9403ff7`](https://github.com/akumaburn/Auralis-Enhanced-Blackwell/commit/9403ff7)
> and [`d9a3d75`](https://github.com/akumaburn/Auralis-Enhanced-Blackwell/commit/d9a3d75).
> Audio totals shrank ~40% because the model now generates the prompt
> sentence and stops, rather than rambling on uncorrelated mel tokens
> until a random EOS sample.

### NovaSR super-resolution overhead

| Metric | Base (24kHz) | Enhanced (NovaSR 48kHz) | Impact |
| :--- | :--- | :--- | :--- |
| **NovaSR Processing RTF** | - | **0.04x** | **25x realtime** |
| **Model Load Time** | - | **0.32s** | Negligible overhead |
| **VRAM Usage** | ~4.2 GB | ~4.3 GB | +0.1 GB Overhead |
| **Model Size** | - | **~52 KB** | Tiny footprint |

### NovaSR vs FlashSR Comparison

| Metric | FlashSR (Old) | NovaSR (New) | Improvement |
| :--- | :--- | :--- | :--- |
| **Model Size** | ~1000 MB | ~52 KB | **19,230x smaller** |
| **Processing Speed** | 14x realtime | 25-3600x realtime | **Up to 257x faster** |
| **VRAM Overhead** | ~700 MB | ~0.1 MB | **99.98% reduction** |

### Audio Quality Verification

Transcription similarity verified using OpenAI Whisper on RTX 3060:

| Sample | Base (24kHz) | Enhanced (48kHz) | Similarity |
| :--- | :--- | :--- | :--- |
| Sample 1 | ✅ Correct | ✅ Correct | 69.2% |
| Sample 2 | ✅ Correct | ✅ Correct | 80.0% |
| Sample 3 | ✅ Correct | ✅ Correct | **100%** |
| Sample 4 | ✅ Correct | ✅ Correct | **100%** |
| Sample 5 | ✅ Correct | ✅ Correct | **100%** |
| **Average** | - | - | **89.8%** |

> [!TIP]
> **RTF < 1.0** means audio is generated faster than it is spoken. At **0.04x**, upsampling 1 minute of audio takes only **2.4 seconds**.

---

## 🎧 Audio Showcase

Experience the difference between standard synthesis and professional **48kHz Audio Enrichment**.

### 🇺🇸 English Comparison
| ID | Text Segment | Standard (24kHz) | NovaSR Enhanced (48kHz) |
| :-- | :--- | :--- | :--- |
| 1 | *Auralis Enhanced: High-performance TTS engine for production...* | <audio controls src="samples/sample_en_1_base.wav"></audio> | <audio controls src="samples/sample_en_1_enhanced.wav"></audio> |
| 2 | *Seamless voice cloning and audio super-resolution...* | <audio controls src="samples/sample_en_2_base.wav"></audio> | <audio controls src="samples/sample_en_2_enhanced.wav"></audio> |
| 3 | *Professional 48kHz quality with NovaSR technology...* | <audio controls src="samples/sample_en_3_base.wav"></audio> | <audio controls src="samples/sample_en_3_enhanced.wav"></audio> |
| 4 | *Full data privacy with local infrastructure deployment...* | <audio controls src="samples/sample_en_4_base.wav"></audio> | <audio controls src="samples/sample_en_4_enhanced.wav"></audio> |
| 5 | *Where speed meets impeccable audio fidelity...* | <audio controls src="samples/sample_en_5_base.wav"></audio> | <audio controls src="samples/sample_en_5_enhanced.wav"></audio> |

### 🇪🇸 Spanish Comparison
| ID | Segmento de Texto | Base (24kHz) | NovaSR Aumentado (48kHz) |
| :-- | :--- | :--- | :--- |
| 1 | *Auralis Enhanced: Motor de texto a voz de alto rendimiento...* | <audio controls src="samples/sample_es_1_base.wav"></audio> | <audio controls src="samples/sample_es_1_enhanced.wav"></audio> |
| 2 | *Clonación de voz e inferencia ultrarrápida sin interrupciones...* | <audio controls src="samples/sample_es_2_base.wav"></audio> | <audio controls src="samples/sample_es_2_enhanced.wav"></audio> |
| 3 | *Síntesis escalada instantáneamente a calidad profesional...* | <audio controls src="samples/sample_es_3_base.wav"></audio> | <audio controls src="samples/sample_es_3_enhanced.wav"></audio> |
| 4 | *Privacidad de datos y escalabilidad ilimitada en su infraestructura...* | <audio controls src="samples/sample_es_4_base.wav"></audio> | <audio controls src="samples/sample_es_4_enhanced.wav"></audio> |
| 5 | *Fidelidad de audio impecable donde la velocidad es clave...* | <audio controls src="samples/sample_es_5_base.wav"></audio> | <audio controls src="samples/sample_es_5_enhanced.wav"></audio> |
---
## 🟢 Blackwell (RTX 50-series) Support

This fork ships a custom `XttsGPT` model on top of vLLM 0.9.2 + PyTorch
2.7+cu128, so it runs natively on Blackwell GPUs (RTX 5080, 5090, RTX
PRO 6000). Key implementation notes:

- The GPT decoder uses vLLM's `HasInnerState` protocol to track per-
  request prefill lengths and hidden states, so vLLM's continuous
  batching is fully restored (no `max_num_seqs=1` workaround).
- Conditioning latents and text embeddings are pre-computed in PyTorch
  and submitted as `EmbedsPrompt(prompt_embeds=...)`, sidestepping the
  multimodal-processor refactor needed for vLLM ≥ 0.7.
- The perceiver resampler enables flash + mem-efficient + math SDPA
  backends together because Blackwell's flash kernel does not yet cover
  every q/k/v shape XTTS feeds it.
- `safetensors` checkpoint loading uses `load_file` + `load_state_dict`
  to avoid newer safetensors' rejection of shared-storage buffers in the
  HiFiGAN decoder's speaker encoder.

See [`CHANGELOG.md`](CHANGELOG.md#210---2026-05-22) for the full porting
notes.

---

## Quick Start ⭐

### 1. Installation
```bash
git clone https://github.com/akumaburn/Auralis-Enhanced-Blackwell.git
cd Auralis-Enhanced-Blackwell
conda create -n auralis python=3.10 -y
conda activate auralis

# PyTorch 2.7 + CUDA 12.8 wheels (Blackwell-capable)
pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0

# Auralis Enhanced + vLLM 0.9.2 + transitive deps
pip install -r requirements.txt
pip install -e .

# NovaSR audio super-resolution (optional, enables 48 kHz output)
pip install git+https://github.com/ysharma3501/NovaSR.git
```

See [`INSTALL.md`](INSTALL.md) for system packages, CPU-only setup, and
troubleshooting.

### 2. Basic Usage (Python)
```python
from auralis import TTS, TTSRequest

# Initialize
tts = TTS().from_pretrained("AstraMindAI/xttsv2", gpt_model='AstraMindAI/xtts2-gpt')

# Generate 48kHz audio with NovaSR
request = TTSRequest(
    text="Hello world! This is Auralis Enhanced with NovaSR.",
    speaker_files=['reference.wav'],
    language="en",
    apply_novasr=True  # Enable 48kHz super-resolution
)
output = tts.generate_speech(request)
output.save('output.wav')
```

### 3. Manual Super-Resolution
```python
# Generate at 24kHz, then enhance to 48kHz
request = TTSRequest(
    text="Manual enhancement example.",
    speaker_files=['reference.wav'],
    apply_novasr=False
)
output = tts.generate_speech(request)
print(f"Sample rate: {output.sample_rate}")  # 24000

# Apply NovaSR manually
enhanced = output.apply_super_resolution()
print(f"Enhanced rate: {enhanced.sample_rate}")  # 48000
```

### 4. Reproducible / styled generation

`TTSRequest` exposes the same generation knobs upstream Coqui XTTS does:

| Field | Default | Effect |
| :--- | :--- | :--- |
| `seed` | `None` | When set, makes the autoregressive sampler deterministic; identical `(text, speaker, seed)` yields a byte-identical waveform. |
| `do_sample` | `True` | `False` switches to greedy decoding (`temperature=0`, `top_k=1`), also fully deterministic. |
| `temperature`, `top_p`, `top_k` | `0.75 / 0.85 / 50` | Standard sampling knobs, forwarded to vLLM's sampler. |
| `repetition_penalty` | `5.0` | Penalises tokens already in the prompt or generated so far (see `LogitsRepetitionPenalizer`). |
| `length_penalty` | `1.0` | Biases the mel-EOS logit by `(1 − length_penalty) * sqrt(n + 1)`. `> 1.0` yields longer audio, `< 1.0` shorter. |
| `speed` | `1.0` | Pitch-preserving time stretch on the HiFiGAN output. `1.5` is 1.5× faster, `0.75` is slower. |
| `apply_novasr` | `False` | 16 → 48 kHz NovaSR super-resolution. |

```python
# Deterministic generation
request = TTSRequest(
    text="The same seed produces the same waveform.",
    speaker_files=['reference.wav'],
    seed=42,
)

# Greedy decoding
request = TTSRequest(
    text="No sampling randomness here.",
    speaker_files=['reference.wav'],
    do_sample=False,
)

# Slow down the output (pitch preserved)
request = TTSRequest(
    text="Read this slowly.",
    speaker_files=['reference.wav'],
    speed=0.75,
)
```

---

## 📈 Throughput benchmark

Reproduce the table at the top of the README on your own GPU:

```bash
# Concurrent throughput sweep (no serial baseline, ~1 min on RTX 5080)
python scripts/benchmark.py --conc 64 --n 64 --no-serial

# Include the serial baseline for a side-by-side speedup number
python scripts/benchmark.py --conc 16 --n 16

# 48 kHz output (NovaSR upsampling on every request)
python scripts/benchmark.py --conc 16 --n 16 --novasr
```

The script prints a single line starting with `RESULT` per run, so it is
easy to log or grep in CI.

### Auto-concurrency

`TTS()` and `TTS().from_pretrained(...)` both pick a sensible
`max_concurrency` (and `scheduler_max_concurrency`) from your
currently-free VRAM if you do not pass an explicit value:

```python
# Auto-detected; on a 16 GiB GPU this typically lands at 32.
tts = TTS().from_pretrained("AstraMindAI/xttsv2",
                            gpt_model="AstraMindAI/xtts2-gpt")
```

The formula divides the smaller of `gpu_memory_utilization *
total_vram` and `free_vram − 3 GiB` (HiFiGAN + conditioning headroom)
by the per-slot cost (~155 MiB at the XTTS GPT config), and clamps
to `[1, 32]`. Lowering `gpu_memory_utilization` scales the suggestion
down naturally; passing `max_concurrency` explicitly disables the
auto-detect. See `auralis.common.utilities.suggest_max_concurrency`
for the full memory model.

---

## 🚀 Docker GPU Deploy

Run the backend with NVIDIA GPU support in one command.

### 1. Build image
```bash
docker build -t auralis-enhanced:latest .
```

### 2. Start API container (GPU)
```bash
docker run -d --name auralis-enhanced \
  --gpus all \
  -p 8000:8000 \
  auralis-enhanced:latest \
  --host 0.0.0.0 --port 8000
```

### 3. Verify service
```bash
curl http://127.0.0.1:8000/health
```

### 4. Optional: Run the Next.js frontend
```bash
cd frontend
npm install
npm run dev
```

The frontend defaults to `http://localhost:8000` for API requests.

## 🌐 Local Server Deployment (without Docker)

### Start backend API
```bash
python -m auralis.entrypoints.oai_server --host 0.0.0.0 --port 8000
```

### Start frontend UI
```bash
cd frontend
npm install
npm run dev
```

---

## 🔄 Migration from FlashSR

If upgrading from a previous version using FlashSR:

```python
# Old code (FlashSR)
request = TTSRequest(
    text="Hello world!",
    speaker_files=['reference.wav'],
    apply_flashsr=True  # OLD - no longer works
)

# New code (NovaSR)
request = TTSRequest(
    text="Hello world!",
    speaker_files=['reference.wav'],
    apply_novasr=True  # NEW
)
```

See [CHANGELOG.md](CHANGELOG.md) for full migration details.

---

## 🤝 Acknowledgments

Special thanks to:
- **[AstraMind AI](https://github.com/astramind-ai)** for the original Auralis project.
- **[Coqui AI](https://coqui.ai/)** for the XTTSv2 model.
- **[NovaSR](https://github.com/ysharma3501/NovaSR)** for the ultra-fast audio super-resolution technology.

---

<div align="center">

### 🌟 [Give us a star on GitHub!](https://github.com/akumaburn/Auralis-Enhanced-Blackwell)

Built with ❤️ by the open-source community.

</div>
