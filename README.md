<div align="center">

# 🌌 Auralis Enhanced
### *Production-Ready Text-to-Speech with Voice Cloning, FlashSR Audio Enhancement & Network Deployment*

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub](https://img.shields.io/badge/GitHub-Auralis--Enhanced-blue.svg)](https://github.com/groxaxo/Auralis-Enhanced)
[![FlashSR](https://img.shields.io/badge/Audio-48kHz%20FlashSR-brightgreen.svg)](https://huggingface.co/YatharthS/FlashSR)

**Process an entire novel in minutes, not hours. Convert books to speech with professional 48kHz audio quality powered by FlashSR!**

[Quick Start](#quick-start-) • [Deployment](#-server-deployment) • [Features](#key-features-) • [Performance](#-performance--benchmarks) • [Credits](#-acknowledgments)

</div>

---

## 🚀 What is Auralis Enhanced?

**Auralis Enhanced** is a production-ready fork of the original Auralis TTS engine, optimized for server deployment and high-quality audio production. It features **FlashSR audio super-resolution**, converting standard 24kHz synthesis into professional **48kHz broadcast-quality output** with negligible overhead.

### ⚡ Key Highlights
- **🎵 Professional Output**: Automatic 48kHz output via ultra-fast FlashSR.
- **🎤 Voice Cloning**: Clone any voice from a short reference audio sample.
- **🌍 Multilingual**: Native support for English, Spanish, and many other languages.
- **⚙️ Production Ready**: Pre-configured for network accessibility (`0.0.0.0`) and multi-GPU setups.
- **⚡ Super Fast**: Optimized inference loop with streaming support.

---

## 📊 Performance & Benchmarks

Measured on an **NVIDIA RTX 3090**, comparing standard 24kHz synthesis against 48kHz production output with FlashSR.

| Metric | Base (24kHz) | Enhanced (FlashSR 48kHz) | Impact |
| :--- | :--- | :--- | :--- |
| **Peak RTF (Real-Time Factor)** | **0.19x** | **0.26x** | Negligible (+0.07x) |
| **Max Generation Speed** | ~8000 chars/min | ~6500 chars/min | Ultra-Fast Processing |
| **VRAM Usage** | ~4.4 GB | ~5.1 GB | +0.7 GB Overhead |
| **System RAM** | ~9.2 GB | ~9.6 GB | Low Memory Footprint |

> [!TIP]
> **RTF < 1.0** means audio is generated faster than it is spoken. At **0.19x**, generating 1 minute of speech takes only **11.4 seconds**.

---

## 🎧 Audio Showcase

Experience the difference between standard synthesis and professional **48kHz Audio Enrichment**.

### 🇺🇸 English Comparison
| ID | Text Segment | Standard (24kHz) | FlashSR Enhanced (48kHz) |
| :-- | :--- | :--- | :--- |
| 1 | *Auralis Enhanced: High-performance TTS engine for production...* | <audio controls src="samples/sample_en_1_base.wav"></audio> | <audio controls src="samples/sample_en_1_enhanced.wav"></audio> |
| 2 | *Seamless voice cloning and audio super-resolution...* | <audio controls src="samples/sample_en_2_base.wav"></audio> | <audio controls src="samples/sample_en_2_enhanced.wav"></audio> |
| 3 | *Professional 48kHz quality with FlashSR technology...* | <audio controls src="samples/sample_en_3_base.wav"></audio> | <audio controls src="samples/sample_en_3_enhanced.wav"></audio> |
| 4 | *Full data privacy with local infrastructure deployment...* | <audio controls src="samples/sample_en_4_base.wav"></audio> | <audio controls src="samples/sample_en_4_enhanced.wav"></audio> |
| 5 | *Where speed meets impeccable audio fidelity...* | <audio controls src="samples/sample_en_5_base.wav"></audio> | <audio controls src="samples/sample_en_5_enhanced.wav"></audio> |

### 🇪🇸 Spanish Comparison
| ID | Segmento de Texto | Base (24kHz) | FlashSR Aumentado (48kHz) |
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

# Generate 48kHz audio
request = TTSRequest(
    text="Hello world! This is Auralis Enhanced.",
    speaker_files=['reference.wav'],
    language="en"
)
output = tts.generate_speech(request)
output.save('output.wav')
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

## 🤝 Acknowledgments

Special thanks to:
- **[AstraMind AI](https://github.com/astramind-ai)** for the original Auralis project.
- **[Coqui AI](https://coqui.ai/)** for the XTTSv2 model.
- **[FlashSR](https://github.com/ysharma3501/FlashSR)** for the audio super-resolution technology.

---

<div align="center">

### 🌟 [Give us a star on GitHub!](https://github.com/groxaxo/Auralis-Enhanced)

Built with ❤️ by the open-source community.

</div>
