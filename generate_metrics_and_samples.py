import os
import time
import json
import asyncio
import torch
import numpy as np
from auralis import TTS, TTSRequest

# Config
LANGUAGES = ["en", "es"]
NUM_SAMPLES = 5
SAMPLES_DIR = "samples"
METRICS_FILE = os.path.join(SAMPLES_DIR, "metrics_report.json")

# Sample texts (English)
EN_TEXTS = [
    "Auralis Enhanced provides high-fidelity speech synthesis with ultra-low latency.",
    "The integration of NovaSR allows for real-time upscaling from 24 kilohertz to 48 kilohertz.",
    "Experience the power of advanced neural text-to-speech technology today.",
    "Our system is optimized for performance, ensuring smooth and natural-sounding audio.",
    "This is a demonstration of the robust audio saturation guards in the latest update.",
]

# Sample texts (Spanish)
ES_TEXTS = [
    "Auralis Enhanced proporciona síntesis de voz de alta fidelidad con una latencia ultra baja.",
    "La integración de NovaSR permite el escalado en tiempo real de 24 kilohercios a 48 kilohercios.",
    "Experimente el poder de la tecnología avanzada de texto a voz neuronal hoy mismo.",
    "Nuestro sistema está optimizado para el rendimiento, garantizando un audio fluido y natural.",
    "Esta es una demostración de las protecciones contra la saturación de audio en la última actualización.",
]


async def generate_samples(tts):
    if not os.path.exists(SAMPLES_DIR):
        os.makedirs(SAMPLES_DIR)

    metrics = []

    # Find speaker files
    en_speaker = "samples/benchmark_en.wav"
    es_speaker = "samples/benchmark_es.wav"

    if not os.path.exists(en_speaker) or not os.path.exists(es_speaker):
        print("Error: Benchmark speaker files missing in samples/!")
        return

    for lang in LANGUAGES:
        texts = EN_TEXTS if lang == "en" else ES_TEXTS
        speaker = en_speaker if lang == "en" else es_speaker

        for i, text in enumerate(texts):
            idx = i + 1
            print(f"Generating {lang} sample {idx}/5...")

            # 1. Base (24kHz)
            req_base = TTSRequest(
                text=text, speaker_files=[speaker], language=lang, apply_novasr=False
            )
            start = time.perf_counter()
            out_base = await tts.generate_speech_async(req_base)
            end = time.perf_counter()

            base_file = f"sample_{lang}_{idx}_base.wav"
            out_base.save(os.path.join(SAMPLES_DIR, base_file))

            # 2. Enhanced (48kHz)
            req_enh = TTSRequest(
                text=text, speaker_files=[speaker], language=lang, apply_novasr=True
            )
            start_enh = time.perf_counter()
            out_enh = await tts.generate_speech_async(req_enh)
            end_enh = time.perf_counter()

            enh_file = f"sample_{lang}_{idx}_enhanced.wav"
            out_enh.save(os.path.join(SAMPLES_DIR, enh_file))

            # Metrics
            rtf_base = (end - start) / (len(out_base.array) / out_base.sample_rate)
            rtf_enh = (end_enh - start_enh) / (len(out_enh.array) / out_enh.sample_rate)

            metrics.append(
                {
                    "language": lang,
                    "id": idx,
                    "text": text,
                    "base": {
                        "file": base_file,
                        "rtf": rtf_base,
                        "duration": len(out_base.array) / out_base.sample_rate,
                    },
                    "enhanced": {
                        "file": enh_file,
                        "rtf": rtf_enh,
                        "duration": len(out_enh.array) / out_enh.sample_rate,
                    },
                }
            )

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"✅ Samples and metrics generated in {SAMPLES_DIR}")


if __name__ == "__main__":
    print("Initializing TTS Engine...")
    tts = TTS().from_pretrained("AstraMindAI/xttsv2", gpt_model="AstraMindAI/xtts2-gpt")
    asyncio.run(generate_samples(tts))
