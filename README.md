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

## 🎧 Audio Samples (Comparison)

Experience the difference between standard synthesis and **FlashSR 48kHz enhancement**.

### 🇺🇸 English Comparison
> "Auralis Enhanced is a high-performance text-to-speech engine designed for production environments. It offers seamless voice cloning, audio super-resolution, and lightning-fast inference for a global audience."

| Base (24kHz) | Enhanced (FlashSR 48kHz) |
| :--- | :--- |
| <audio controls src="samples/sample_en_base.wav"></audio> | <audio controls src="samples/sample_en_enhanced.wav"></audio> |

### 🇪🇸 Spanish Comparison
> "Auralis Enhanced es un motor de texto a voz de alto rendimiento diseñado para entornos de producción. Ofrece clonación de voz sin interrupciones, súper resolución de audio e inferencia ultrarrápida para una audiencia global."

| Base (24kHz) | Enhanced (FlashSR 48kHz) |
| :--- | :--- |
| <audio controls src="samples/sample_es_base.wav"></audio> | <audio controls src="samples/sample_es_enhanced.wav"></audio> |

---

## 📊 Performance & Benchmarks

Measured on an **NVIDIA RTX 3090** (Single Stream Inference with FlashSR 48kHz).

### Generation Speed (RTF)
| Language | Audio Duration | Generation Time | Real-Time Factor (RTF) |
| :--- | :--- | :--- | :--- |
| **English** 🇺🇸 | 12.06s | 13.53s | **1.12x** |
| **Spanish** 🇪🇸 | 17.68s | 11.99s | **0.68x** |

> [!NOTE]
> RTF (Real-Time Factor) < 1.0 means the system generates audio faster than it is spoken. For book-scale processing with batching, RTF can reach as low as **0.02x**.

### Resource Usage
| Component | Usage (Base) | Usage (Peak) |
| :--- | :--- | :--- |
| **VRAM (GPU)** | ~4.1 GB | **4.6 GB** |
| **RAM (System)** | ~9.2 GB | **9.6 GB** |
| **CPU** | Moderate | Peak during FlashSR |

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
