#!/usr/bin/env python3
"""
End-to-end WER benchmark for Auralis-Enhanced.
Generates 50 EN + 50 ES audios via Auralis TTS, transcribes with Parakeet ASR,
then computes WER to verify audio quality before and after bottleneck fixes.
"""

import argparse
import json
import os
import sys
import time
import wave
import struct
import requests
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import re

# ─── Test corpus ─────────────────────────────────────────────────────────────

EN_SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Artificial intelligence is transforming how we interact with technology every day.",
    "She carefully placed the delicate crystal vase on the mahogany shelf.",
    "The astronauts completed their spacewalk and returned safely to the station.",
    "Climate change poses significant challenges to ecosystems around the world.",
    "He practiced the violin for three hours before the evening performance began.",
    "The ancient library contained thousands of manuscripts written by hand.",
    "Scientists discovered a new species of deep-sea fish in the Pacific Ocean.",
    "The chef prepared a magnificent seven-course dinner for the visiting dignitaries.",
    "Modern cryptography relies on mathematical problems that are hard to reverse.",
    "She read the entire novel in one sitting, unable to put it down.",
    "The thunderstorm knocked out power to the entire neighborhood for six hours.",
    "Children learn language naturally through immersion and constant exposure.",
    "The bridge was constructed using advanced composite materials for durability.",
    "He solved the complex differential equation using Laplace transforms.",
    "The concert hall filled with applause as the orchestra took its final bow.",
    "Machine learning models can now generate realistic images from text descriptions.",
    "The expedition team reached the summit just before the afternoon storm arrived.",
    "She built a small greenhouse in her backyard to grow tomatoes and herbs.",
    "The parliament voted unanimously to pass the new environmental protection bill.",
    "Neural networks process information in ways inspired by the human brain.",
    "The vintage car needed new brake pads and a complete engine overhaul.",
    "They spent the afternoon kayaking along the winding river through the forest.",
    "The professor explained the principles of quantum entanglement to the students.",
    "Fresh bread from the local bakery always sells out before noon on weekends.",
    "The documentary explored the cultural traditions of indigenous communities worldwide.",
    "He installed solar panels on his roof to reduce his electricity bills.",
    "The hospital staff worked tirelessly during the unexpected surge in patients.",
    "Photosynthesis converts sunlight into chemical energy stored in glucose molecules.",
    "The hikers followed the marked trail through dense pine forest for eight miles.",
    "She designed an app that helps farmers track weather patterns and crop yields.",
    "The puppy chased its tail around the kitchen until it became dizzy.",
    "Renaissance paintings are celebrated for their extraordinary attention to detail.",
    "The satellite transmitted high-resolution images of the hurricane to the ground station.",
    "He learned to play chess at age seven and became a grandmaster by thirty.",
    "The city council approved plans for a new waterfront park and amphitheater.",
    "Bacteria can evolve antibiotic resistance through natural selection over time.",
    "She organized a fundraiser that raised over twenty thousand dollars for the shelter.",
    "The submarine descended to the ocean floor to study hydrothermal vents.",
    "Long-distance running requires both physical endurance and mental discipline.",
    "The architect designed the museum to maximize natural light in every gallery.",
    "They celebrated their anniversary with a candlelight dinner and a walk by the sea.",
    "The new software update introduced several improvements to the user interface.",
    "Volcanoes release enormous amounts of energy stored in the Earth's mantle.",
    "The detective pieced together the clues and identified the suspect by morning.",
    "She translated the ancient manuscript from Latin into modern English.",
    "The company launched its flagship product after three years of research and development.",
    "Migrating birds navigate thousands of miles using the Earth's magnetic field.",
    "The students collaborated on a project about renewable energy solutions.",
    "He composed a symphony in honor of the city's two-hundredth anniversary.",
]

ES_SENTENCES = [
    "El zorro marrón rápido salta sobre el perro perezoso cerca del río.",
    "La inteligencia artificial está transformando la forma en que interactuamos con la tecnología.",
    "Ella colocó cuidadosamente el delicado jarrón de cristal en el estante.",
    "Los astronautas completaron su caminata espacial y regresaron sanos y salvos.",
    "El cambio climático plantea desafíos significativos para los ecosistemas del mundo.",
    "Él practicó el violín durante tres horas antes de la actuación de la tarde.",
    "La antigua biblioteca contenía miles de manuscritos escritos a mano.",
    "Los científicos descubrieron una nueva especie de pez en las profundidades del océano.",
    "El chef preparó una magnifica cena de siete platos para los dignatarios visitantes.",
    "La criptografía moderna se basa en problemas matemáticos difíciles de revertir.",
    "Ella leyó toda la novela en una sola sesión, sin poder dejarla.",
    "La tormenta eléctrica dejó sin electricidad a todo el vecindario durante seis horas.",
    "Los niños aprenden el lenguaje de forma natural mediante la inmersión constante.",
    "El puente se construyó utilizando materiales compuestos avanzados para mayor durabilidad.",
    "Él resolvió la compleja ecuación diferencial usando transformadas de Laplace.",
    "La sala de conciertos se llenó de aplausos cuando la orquesta hizo su reverencia final.",
    "Los modelos de aprendizaje automático ahora pueden generar imágenes realistas a partir de texto.",
    "El equipo de expedición llegó a la cima justo antes de la tormenta de la tarde.",
    "Ella construyó un pequeño invernadero en su jardín trasero para cultivar tomates y hierbas.",
    "El parlamento votó unánimemente para aprobar el nuevo proyecto de ley de protección ambiental.",
    "Las redes neuronales procesan información de formas inspiradas en el cerebro humano.",
    "El automóvil clásico necesitaba nuevas pastillas de freno y una revisión completa del motor.",
    "Pasaron la tarde remando en kayak por el sinuoso río a través del bosque.",
    "El profesor explicó los principios del entrelazamiento cuántico a los estudiantes.",
    "El pan fresco de la panadería local siempre se agota antes del mediodía los fines de semana.",
    "El documental exploró las tradiciones culturales de las comunidades indígenas de todo el mundo.",
    "Él instaló paneles solares en su techo para reducir sus facturas de electricidad.",
    "El personal del hospital trabajó incansablemente durante el inesperado aumento de pacientes.",
    "La fotosíntesis convierte la luz solar en energía química almacenada en moléculas de glucosa.",
    "Los excursionistas siguieron el sendero marcado a través del denso bosque de pinos durante ocho kilómetros.",
    "Ella diseñó una aplicación que ayuda a los agricultores a rastrear patrones climáticos.",
    "El cachorro persiguió su cola alrededor de la cocina hasta que se mareó.",
    "Las pinturas renacentistas son célebres por su extraordinaria atención al detalle.",
    "El satélite transmitió imágenes de alta resolución del huracán a la estación terrestre.",
    "Aprendió a jugar ajedrez a los siete años y se convirtió en gran maestro a los treinta.",
    "El concejo municipal aprobó los planes para un nuevo parque y anfiteatro frente al mar.",
    "Las bacterias pueden desarrollar resistencia a los antibióticos mediante la selección natural.",
    "Ella organizó una recaudación de fondos que recaudó más de veinte mil dólares para el refugio.",
    "El submarino descendió al fondo del océano para estudiar las chimeneas hidrotermales.",
    "La carrera de larga distancia requiere tanto resistencia física como disciplina mental.",
    "El arquitecto diseñó el museo para maximizar la luz natural en cada galería.",
    "Celebraron su aniversario con una cena a la luz de las velas y un paseo junto al mar.",
    "La nueva actualización de software introdujo varias mejoras en la interfaz de usuario.",
    "Los volcanes liberan enormes cantidades de energía almacenada en el manto terrestre.",
    "El detective unió las pistas e identificó al sospechoso antes del amanecer.",
    "Ella tradujo el antiguo manuscrito del latín al español moderno.",
    "La empresa lanzó su producto principal después de tres años de investigación y desarrollo.",
    "Las aves migratorias navegan miles de kilómetros usando el campo magnético terrestre.",
    "Los estudiantes colaboraron en un proyecto sobre soluciones de energía renovable.",
    "Él compuso una sinfonía en honor al bicentenario de la ciudad.",
]

assert len(EN_SENTENCES) == 50, f"Need 50 EN sentences, got {len(EN_SENTENCES)}"
assert len(ES_SENTENCES) == 50, f"Need 50 ES sentences, got {len(ES_SENTENCES)}"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Lower-case and strip punctuation for WER computation."""
    text = text.lower()
    text = re.sub(r"[^a-záéíóúüñàèìòùâêîôûäëïöü\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate."""
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    # Dynamic programming
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=int)
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def generate_audio(
    text: str,
    language: str,
    voice_name: str,
    auralis_url: str,
    output_path: str,
    timeout: int = 180,
) -> Tuple[bool, float]:
    """Call Auralis TTS endpoint and save the WAV file. Returns (success, latency_s)."""
    payload = {
        "model": "tts-1",
        "input": text,
        "voice": voice_name,           # OpenAI-style voice name mapped in server
        "response_format": "wav",
        "language": language,
        "temperature": 0.75,
        "top_p": 0.85,
        "top_k": 50,
        "repetition_penalty": 10.0,
        "max_ref_length": 30,
        "gpt_cond_len": 12,
        "gpt_cond_chunk_len": 6,
        "apply_novasr": False,
    }

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{auralis_url}/v1/audio/speech",
            json=payload,
            timeout=timeout,
        )
        latency = time.perf_counter() - t0
        if resp.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(resp.content)
            return True, latency
        else:
            print(f"  TTS error {resp.status_code}: {resp.text[:200]}")
            return False, latency
    except Exception as e:
        latency = time.perf_counter() - t0
        print(f"  TTS exception: {e}")
        return False, latency


def transcribe_audio(
    audio_path: str,
    parakeet_url: str,
    timeout: int = 60,
) -> Optional[str]:
    """Transcribe audio file using Parakeet ASR. Returns transcript or None."""
    try:
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
            data = {"model": "parakeet-tdt-0.6b-v3", "response_format": "json"}
            resp = requests.post(
                f"{parakeet_url}/v1/audio/transcriptions",
                files=files,
                data=data,
                timeout=timeout,
            )
        if resp.status_code == 200:
            result = resp.json()
            return result.get("text", "").strip()
        else:
            print(f"  ASR error {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"  ASR exception: {e}")
        return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_benchmark(
    tag: str,
    auralis_url: str,
    parakeet_url: str,
    en_speaker: str = None,  # kept for compat, not used
    es_speaker: str = None,  # kept for compat, not used
    output_dir: str = "/home/op/Auralis-Enhanced/benchmark_outputs",
    resume: bool = True,
):
    """Run the full 50+50 benchmark and return results dict."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results_file = os.path.join(output_dir, f"results_{tag}.json")

    # Load previous results if resuming
    results = {}
    if resume and os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} existing results")

    tasks = (
        [(f"en_{i:02d}", EN_SENTENCES[i], "en", "nova") for i in range(50)]
        + [(f"es_{i:02d}", ES_SENTENCES[i], "es", "alloy") for i in range(50)]
    )

    latencies = []
    wer_scores = []

    for task_id, text, lang, voice in tasks:
        if task_id in results and results[task_id].get("transcript") is not None:
            print(f"  [SKIP] {task_id} already done")
            latencies.append(results[task_id]["latency_s"])
            wer_scores.append(results[task_id]["wer"])
            continue

        audio_path = os.path.join(output_dir, f"{tag}_{task_id}.wav")
        print(f"\n[{tag}] Generating {task_id} ({lang}): {text[:60]}...")

        # Generate
        ok, latency = generate_audio(
            text, lang, voice, auralis_url, audio_path
        )
        if not ok:
            print(f"  FAILED to generate {task_id}")
            results[task_id] = {"text": text, "lang": lang, "latency_s": latency,
                                 "transcript": None, "wer": None, "success": False}
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            continue

        print(f"  Generated in {latency:.2f}s")

        # Transcribe
        transcript = transcribe_audio(audio_path, parakeet_url)
        if transcript is None:
            print(f"  FAILED to transcribe {task_id}")
            results[task_id] = {"text": text, "lang": lang, "latency_s": latency,
                                 "transcript": None, "wer": None, "success": False}
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            continue

        score = wer(text, transcript)
        print(f"  Transcript: {transcript[:80]}")
        print(f"  WER: {score:.3f} | Latency: {latency:.2f}s")

        results[task_id] = {
            "text": text,
            "lang": lang,
            "latency_s": latency,
            "transcript": transcript,
            "wer": score,
            "success": True,
        }
        latencies.append(latency)
        wer_scores.append(score)

        # Save after every sample
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    valid_wer = [w for w in wer_scores if w is not None]
    valid_lat = [l for l in latencies if l is not None]
    summary = {
        "tag": tag,
        "n_total": len(tasks),
        "n_success": sum(1 for r in results.values() if r.get("success")),
        "mean_wer": float(np.mean(valid_wer)) if valid_wer else None,
        "median_wer": float(np.median(valid_wer)) if valid_wer else None,
        "en_wer": float(np.mean([results[f"en_{i:02d}"]["wer"] for i in range(50)
                                   if results.get(f"en_{i:02d}", {}).get("wer") is not None])) if valid_wer else None,
        "es_wer": float(np.mean([results[f"es_{i:02d}"]["wer"] for i in range(50)
                                   if results.get(f"es_{i:02d}", {}).get("wer") is not None])) if valid_wer else None,
        "mean_latency_s": float(np.mean(valid_lat)) if valid_lat else None,
        "p95_latency_s": float(np.percentile(valid_lat, 95)) if valid_lat else None,
    }
    print(f"\n{'='*60}")
    print(f"SUMMARY [{tag}]")
    print(f"  Success:      {summary['n_success']}/{summary['n_total']}")
    print(f"  Mean WER:     {summary['mean_wer']:.3f}" if summary['mean_wer'] else "  Mean WER: N/A")
    print(f"  EN WER:       {summary['en_wer']:.3f}" if summary['en_wer'] else "  EN WER: N/A")
    print(f"  ES WER:       {summary['es_wer']:.3f}" if summary['es_wer'] else "  ES WER: N/A")
    print(f"  Mean latency: {summary['mean_latency_s']:.2f}s" if summary['mean_latency_s'] else "  Mean latency: N/A")
    print(f"  P95 latency:  {summary['p95_latency_s']:.2f}s" if summary['p95_latency_s'] else "  P95 latency: N/A")

    summary_file = os.path.join(output_dir, f"summary_{tag}.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary, results


def compare_results(before: dict, after: dict):
    """Print comparison of before vs after WER."""
    print("\n" + "=" * 60)
    print("BEFORE vs AFTER COMPARISON")
    print("=" * 60)
    metrics = ["mean_wer", "en_wer", "es_wer", "mean_latency_s", "p95_latency_s"]
    for m in metrics:
        b = before.get(m)
        a = after.get(m)
        if b is not None and a is not None:
            delta = a - b
            pct = (delta / b * 100) if b != 0 else 0
            arrow = "↓" if delta < 0 else "↑"
            print(f"  {m:20s}: {b:.3f} → {a:.3f}  ({arrow}{abs(pct):.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, choices=["baseline", "optimized"])
    parser.add_argument("--auralis-url", default="http://localhost:9951")
    parser.add_argument("--parakeet-url", default="http://localhost:5092")
    parser.add_argument("--en-speaker",
                        default="/home/op/Auralis-Enhanced/tests/resources/audio_samples/female.wav",
                        help="Not used directly (voice names used instead), kept for compat")
    parser.add_argument("--es-speaker",
                        default="/home/op/reference_audio/google_argentinian_5679/speaker_5679_reference_48s.wav",
                        help="Not used directly (voice names used instead), kept for compat")
    parser.add_argument("--output-dir", default="/home/op/Auralis-Enhanced/benchmark_outputs")
    parser.add_argument("--compare", action="store_true",
                        help="Compare baseline vs optimized after both runs")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.auralis_url.endswith("9951"):
        # Wait for warm-up on first request
        print(f"Checking Auralis server at {args.auralis_url}...")
        for attempt in range(30):
            try:
                r = requests.get(f"{args.auralis_url}/health", timeout=5)
                if r.status_code == 200:
                    print(f"  Server healthy: {r.json()}")
                    break
            except Exception:
                pass
            print(f"  Waiting... ({attempt+1}/30)")
            time.sleep(10)

    summary, results = run_benchmark(
        tag=args.tag,
        auralis_url=args.auralis_url,
        parakeet_url=args.parakeet_url,
        en_speaker=args.en_speaker,
        es_speaker=args.es_speaker,
        output_dir=args.output_dir,
        resume=not args.no_resume,
    )

    if args.compare:
        baseline_file = os.path.join(args.output_dir, "summary_baseline.json")
        optimized_file = os.path.join(args.output_dir, "summary_optimized.json")
        if os.path.exists(baseline_file) and os.path.exists(optimized_file):
            with open(baseline_file) as f:
                b = json.load(f)
            with open(optimized_file) as f:
                a = json.load(f)
            compare_results(b, a)
        else:
            print("Both baseline and optimized summaries needed for comparison")
