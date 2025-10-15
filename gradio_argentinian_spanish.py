#!/usr/bin/env python3
"""
Gradio interface for Argentinian Spanish XTTS-v2 TTS Server
"""

import asyncio
import base64
import time
import uuid
import shutil
from pathlib import Path
from typing import List, Optional

import gradio as gr
import torch
import torchaudio

from auralis import TTS, TTSRequest, TTSOutput, AudioPreprocessingConfig, setup_logger

logger = setup_logger(__file__)

# Initialize TTS with Argentinian Spanish model
print("🚀 Loading Argentinian Spanish XTTS-v2 model...")
tts = TTS(scheduler_max_concurrency=4)
model_path = "/home/op/Auralis/converted_models/argentinian_spanish/core_xttsv2"
gpt_model = "/home/op/Auralis/converted_models/argentinian_spanish/gpt"

try:
    tts = tts.from_pretrained(model_path, gpt_model=gpt_model)
    logger.info(f"✅ Successfully loaded Argentinian Spanish model")
    print("✅ Model loaded successfully!")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    print(f"❌ Error: {e}")
    raise

# Create a temporary directory to store files
temp_dir = Path("/tmp/auralis_argentinian")
temp_dir.mkdir(exist_ok=True)

def shorten_filename(original_path: str) -> str:
    """Copies the given file to a temporary directory with a shorter, random filename."""
    ext = Path(original_path).suffix
    short_name = "file_" + uuid.uuid4().hex[:8] + ext
    short_path = temp_dir / short_name
    shutil.copyfile(original_path, short_path)
    return str(short_path)

def process_text_and_generate(input_text, ref_audio_files, speed, enhance_speech, temperature, top_p, top_k, repetition_penalty, language):
    """Process text and generate audio."""
    log_messages = ""
    
    if not input_text or len(input_text.strip()) == 0:
        log_messages += "⚠️ Please provide some text to convert!\n"
        return None, log_messages
    
    if not ref_audio_files:
        log_messages += "⚠️ Please provide at least one reference audio!\n"
        return None, log_messages

    # Use the uploaded audio files
    audio_files = ref_audio_files[:5]  # Limit to 5 files

    request = TTSRequest(
        text=input_text,
        speaker_files=audio_files,
        stream=False,
        enhance_speech=enhance_speech,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        language=language,
    )

    try:
        log_messages += f"🎤 Generating speech...\n"
        log_messages += f"📝 Text: {input_text[:100]}{'...' if len(input_text) > 100 else ''}\n"
        log_messages += f"🌍 Language: {language}\n"
        
        with torch.no_grad():
            output = tts.generate_speech(request)
            if output:
                if speed != 1:
                    output = output.change_speed(speed)
                log_messages += f"✅ Successfully generated audio!\n"
                log_messages += f"📊 Sample rate: {output.sample_rate} Hz\n"
                log_messages += f"⏱️ Duration: {len(output.array) / output.sample_rate:.2f} seconds\n"
                return (output.sample_rate, output.array), log_messages
            else:
                log_messages += "❌ No output was generated. Check that the model was correctly loaded\n"
                return None, log_messages
    except Exception as e:
        logger.error(f"Error: {e}")
        log_messages += f"❌ An error occurred: {e}\n"
        return None, log_messages

def build_gradio_ui():
    """Builds and launches the Gradio UI for Argentinian Spanish TTS."""
    
    # Example texts in Argentinian Spanish
    example_texts = [
        "¡Che, boludo! ¿Cómo andás? Todo bien por acá.",
        "Este modelo de texto a voz está re copado, funciona bárbaro.",
        "¿Viste lo que pasó ayer? Fue una locura total, te lo juro.",
        "Mirá, la verdad es que no tengo ni idea de qué hacer con esto.",
        "Dale, vamos a tomar unos mates y charlamos un rato.",
    ]
    
    with gr.Blocks(title="Argentinian Spanish TTS", theme="soft") as ui:

        gr.Markdown(
            """
            # 🇦🇷 Argentinian Spanish XTTS-v2 TTS Demo
            
            Convert text to speech with authentic Argentinian Spanish accent!
            
            **Features:**
            - 🎭 Voice cloning from reference audio
            - 🗣️ Natural Rioplatense accent (Buenos Aires region)
            - 🎵 Voseo conjugations and Argentinian expressions
            - ⚡ High-quality neural TTS
            
            **Tips:**
            - Upload 5-30 seconds of clear reference audio
            - Use Argentinian Spanish expressions for best results
            - Try phrases with "che", "boludo", "vos", etc.
            """
        )

        with gr.Tab("🎤 Text to Speech"):
            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(
                        label="Enter Text (Spanish)",
                        placeholder="Escribí el texto que querés convertir...",
                        lines=5,
                        value=example_texts[0]
                    )
                    
                    ref_audio_files = gr.Files(
                        label="Reference Audio Files (Upload voice samples)",
                        file_types=["audio"]
                    )
                    
                    with gr.Accordion("⚙️ Advanced Settings", open=False):
                        speed = gr.Slider(
                            label="Playback Speed",
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.1
                        )
                        enhance_speech = gr.Checkbox(
                            label="Enhance Reference Speech",
                            value=False,
                            info="Apply audio enhancement to reference"
                        )
                        temperature = gr.Slider(
                            label="Temperature (Creativity)",
                            minimum=0.5,
                            maximum=1.0,
                            value=0.75,
                            step=0.05,
                            info="Higher = more creative, lower = more consistent"
                        )
                        top_p = gr.Slider(
                            label="Top P (Nucleus Sampling)",
                            minimum=0.5,
                            maximum=1.0,
                            value=0.85,
                            step=0.05
                        )
                        top_k = gr.Slider(
                            label="Top K",
                            minimum=0,
                            maximum=100,
                            value=50,
                            step=10
                        )
                        repetition_penalty = gr.Slider(
                            label="Repetition Penalty",
                            minimum=1.0,
                            maximum=10.0,
                            value=5.0,
                            step=0.5
                        )
                        language = gr.Dropdown(
                            label="Target Language",
                            choices=["es", "auto"],
                            value="es",
                            info="Use 'es' for Spanish"
                        )
                    
                    generate_button = gr.Button("🎙️ Generate Speech", variant="primary", size="lg")
                    
                with gr.Column():
                    audio_output = gr.Audio(label="🔊 Generated Audio", type="numpy")
                    log_output = gr.Textbox(label="📋 Log Output", lines=10)

            generate_button.click(
                process_text_and_generate,
                inputs=[input_text, ref_audio_files, speed, enhance_speech, temperature, top_p, top_k, repetition_penalty, language],
                outputs=[audio_output, log_output],
            )
            
            # Example buttons
            gr.Markdown("### 📝 Example Texts (Click to use)")
            with gr.Row():
                for i, example in enumerate(example_texts[:3]):
                    btn = gr.Button(f"Example {i+1}", size="sm")
                    btn.click(lambda x=example: x, outputs=input_text)

        with gr.Tab("🎙️ Record & Clone"):
            with gr.Row():
                with gr.Column():
                    input_text_mic = gr.Textbox(
                        label="Enter Text (Spanish)",
                        placeholder="Escribí el texto que querés convertir...",
                        lines=5,
                        value="¡Che, boludo! ¿Cómo andás?"
                    )
                    
                    mic_ref_audio = gr.Audio(
                        label="🎤 Record Your Voice (5-30 seconds)",
                        sources=["microphone"],
                        type="numpy"
                    )

                    with gr.Accordion("⚙️ Advanced Settings", open=False):
                        speed_mic = gr.Slider(label="Playback Speed", minimum=0.5, maximum=2.0, value=1.0, step=0.1)
                        enhance_speech_mic = gr.Checkbox(label="Enhance Reference Speech", value=True)
                        temperature_mic = gr.Slider(label="Temperature", minimum=0.5, maximum=1.0, value=0.75, step=0.05)
                        top_p_mic = gr.Slider(label="Top P", minimum=0.5, maximum=1.0, value=0.85, step=0.05)
                        top_k_mic = gr.Slider(label="Top K", minimum=0, maximum=100, value=50, step=10)
                        repetition_penalty_mic = gr.Slider(label="Repetition Penalty", minimum=1.0, maximum=10.0, value=5.0, step=0.5)
                        language_mic = gr.Dropdown(label="Target Language", choices=["es", "auto"], value="es")
                    
                    generate_button_mic = gr.Button("🎙️ Generate Speech", variant="primary", size="lg")
                    
                with gr.Column():
                    audio_output_mic = gr.Audio(label="🔊 Generated Audio", type="numpy")
                    log_output_mic = gr.Textbox(label="📋 Log Output", lines=10)

            def process_mic_and_generate(input_text_mic, mic_ref_audio, speed_mic, enhance_speech_mic, temperature_mic, top_p_mic, top_k_mic, repetition_penalty_mic, language_mic):
                if mic_ref_audio:
                    import hashlib
                    data = str(time.time()).encode("utf-8")
                    hash_val = hashlib.sha1(data).hexdigest()[:10]
                    output_path = temp_dir / f"mic_{hash_val}.wav"

                    torch_audio = torch.from_numpy(mic_ref_audio[1].astype(float))
                    try:
                        torchaudio.save(str(output_path), torch_audio.unsqueeze(0), mic_ref_audio[0])
                        return process_text_and_generate(
                            input_text_mic, [str(output_path)], speed_mic, enhance_speech_mic,
                            temperature_mic, top_p_mic, top_k_mic, repetition_penalty_mic, language_mic
                        )
                    except Exception as e:
                        logger.error(f"Error saving audio file: {e}")
                        return None, f"❌ Error saving audio file: {e}"
                else:
                    return None, "⚠️ Please record an audio!"

            generate_button_mic.click(
                process_mic_and_generate,
                inputs=[input_text_mic, mic_ref_audio, speed_mic, enhance_speech_mic, temperature_mic, top_p_mic, top_k_mic, repetition_penalty_mic, language_mic],
                outputs=[audio_output_mic, log_output_mic],
            )

        gr.Markdown(
            """
            ---
            ### 🇦🇷 About Argentinian Spanish
            
            This model is trained on Argentinian Spanish (Rioplatense dialect) and includes:
            - **Voseo**: Uses "vos" instead of "tú" (e.g., "¿Cómo andás?" instead of "¿Cómo andas?")
            - **Argentinian vocabulary**: "che", "boludo", "copado", "bárbaro", etc.
            - **Rioplatense accent**: Characteristic pronunciation from Buenos Aires region
            - **Natural intonation**: Authentic speech patterns and rhythm
            
            **Model**: marianbasti/XTTS-v2-argentinian-spanish  
            **Architecture**: XTTS-v2 (Coqui TTS)  
            **Voice Cloning**: Supported with 5-30 seconds of reference audio
            """
        )

    return ui

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🇦🇷 Argentinian Spanish XTTS-v2 Gradio Interface")
    print("="*50 + "\n")
    
    ui = build_gradio_ui()
    ui.launch(
        debug=True,
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True
    )
