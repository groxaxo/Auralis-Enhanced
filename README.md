<div align="center">

# 🌌 Auralis Enhanced
### *Production-Ready Text-to-Speech with Voice Cloning, NovaSR Audio Enhancement & Network Deployment*

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub](https://img.shields.io/badge/GitHub-Auralis--Enhanced-blue.svg)](https://github.com/groxaxo/Auralis-Enhanced)
[![NovaSR](https://img.shields.io/badge/Audio-48kHz%20NovaSR-brightgreen.svg)](https://huggingface.co/YatharthS/NovaSR)

**Process an entire novel in minutes, not hours. Convert books to speech with professional 48kHz audio quality powered by NovaSR!**

[Quick Start](#quick-start-) • [Deployment](#-server-deployment) • [Features](#key-features-) • [Performance](#-performance--benchmarks) • [Credits](#-acknowledgments)

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

Measured on an **NVIDIA RTX 3060 (12GB)**, comparing standard 24kHz synthesis against 48kHz production output with NovaSR.

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
## Quick Start ⭐

### 1. Installation
```bash
git clone https://github.com/groxaxo/Auralis-Enhanced.git
cd Auralis-Enhanced
conda create -n auralis_env python=3.10 -y
conda activate auralis_env
pip install -r requirements.txt
pip install -e .
```

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

---

## 🚀 Server Deployment

This fork is built for the network. Both the API and UI are accessible from `0.0.0.0` by default.

### Start Backend API
```bash
python -m auralis.entrypoints.oai_server --host 0.0.0.0 --port 8000
```

### Start Frontend UI
```bash
cd examples && python gradio_example.py
```

For advanced deployment (systemd, Docker, Nginx), see the **[Server Deployment Guide](docs/deployment/server-setup.md)**.

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

### 🌟 [Give us a star on GitHub!](https://github.com/groxaxo/Auralis-Enhanced)

Built with ❤️ by the open-source community.

</div>
