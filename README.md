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

## 📊 Performance Metrics

We measured the performance of Auralis Enhanced on an **NVIDIA RTX 3090**, comparing standard 24kHz synthesis against 48kHz production output with FlashSR.

| Metric | Base (24kHz) | Enhanced (FlashSR 48kHz) | Impact |
| :--- | :--- | :--- | :--- |
| **Real-Time Factor (RTF)** | **0.87x** | **0.91x** | Negligible (~0.04x) |
| **Avg Generation Speed** | ~1700 chars/min | ~1600 chars/min | High Speed Maintained |
| **VRAM Usage** | ~4.4 GB | ~5.1 GB | +0.7 GB |

> **Note:** Lower RTF is better. An RTF of 0.91x means generating 10 seconds of audio takes only 9.1 seconds.

---

## 🎧 Audio Samples (Comparison)

Below are 5 examples per language demonstrating the upgrade from standard TTS to **FlashSR 48kHz Studio Quality**.

### 🇺🇸 English Comparison

| ID | Text Segment | Base (24kHz) | Enhanced (48kHz) |
| :--- | :--- | :--- | :--- |
| 1 | *"Auralis Enhanced is a high-performance..."* | <audio controls src="samples/sample_en_1_base.wav"></audio> | <audio controls src="samples/sample_en_1_enhanced.wav"></audio> |
| 2 | *"It offers seamless voice cloning..."* | <audio controls src="samples/sample_en_2_base.wav"></audio> | <audio controls src="samples/sample_en_2_enhanced.wav"></audio> |
| 3 | *"With FlashSR technology..."* | <audio controls src="samples/sample_en_3_base.wav"></audio> | <audio controls src="samples/sample_en_3_enhanced.wav"></audio> |
| 4 | *"Deploy it on your own infrastructure..."* | <audio controls src="samples/sample_en_4_base.wav"></audio> | <audio controls src="samples/sample_en_4_enhanced.wav"></audio> |
| 5 | *"Experience the future of voice..."* | <audio controls src="samples/sample_en_5_base.wav"></audio> | <audio controls src="samples/sample_en_5_enhanced.wav"></audio> |

### 🇪🇸 Spanish Comparison

| ID | Text Segment | Base (24kHz) | Enhanced (48kHz) |
| :--- | :--- | :--- | :--- |
| 1 | *"Auralis Enhanced es un motor..."* | <audio controls src="samples/sample_es_1_base.wav"></audio> | <audio controls src="samples/sample_es_1_enhanced.wav"></audio> |
| 2 | *"Ofrece clonación de voz..."* | <audio controls src="samples/sample_es_2_base.wav"></audio> | <audio controls src="samples/sample_es_2_enhanced.wav"></audio> |
| 3 | *"Con la tecnología FlashSR..."* | <audio controls src="samples/sample_es_3_base.wav"></audio> | <audio controls src="samples/sample_es_3_enhanced.wav"></audio> |
| 4 | *"Implementelo en su propia..."* | <audio controls src="samples/sample_es_4_base.wav"></audio> | <audio controls src="samples/sample_es_4_enhanced.wav"></audio> |
| 5 | *"Experimente el futuro..."* | <audio controls src="samples/sample_es_5_base.wav"></audio> | <audio controls src="samples/sample_es_5_enhanced.wav"></audio> |

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
