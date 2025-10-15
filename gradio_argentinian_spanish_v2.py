#!/usr/bin/env python3
"""
Gradio interface for Argentinian Spanish XTTS-v2 TTS Server
Version 2: With voice library and improved state management
"""

import asyncio
import base64
import time
import uuid
import shutil
import json
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime

import gradio as gr
import torch
import torchaudio
import numpy as np

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

# Create directories
temp_dir = Path("/tmp/auralis_argentinian")
temp_dir.mkdir(exist_ok=True)

voice_library_dir = Path("/home/op/Auralis/voice_library")
voice_library_dir.mkdir(exist_ok=True)

output_dir = Path("/home/op/Auralis/generated_audio")
output_dir.mkdir(exist_ok=True)

# Voice library metadata
voice_library_file = voice_library_dir / "voices.json"

def load_voice_library() -> Dict:
    """Load voice library metadata"""
    if voice_library_file.exists():
        with open(voice_library_file, 'r') as f:
            return json.load(f)
    return {}

def save_voice_library(library: Dict):
    """Save voice library metadata"""
    with open(voice_library_file, 'w') as f:
        json.dump(library, f, indent=2)

def get_voice_choices() -> List[str]:
    """Get list of available voices"""
    library = load_voice_library()
    choices = ["Upload new voice..."] + list(library.keys())
    return choices

def save_voice_to_library(audio_files: List, voice_name: str) -> str:
    """Save uploaded audio to voice library"""
    if not audio_files or not voice_name:
        return "❌ Please provide audio files and a voice name"
    
    voice_name = voice_name.strip()
    if not voice_name:
        return "❌ Voice name cannot be empty"
    
    library = load_voice_library()
    
    # Create voice directory
    voice_dir = voice_library_dir / voice_name
    voice_dir.mkdir(exist_ok=True)
    
    # Copy audio files
    saved_files = []
    for i, audio_file in enumerate(audio_files[:5]):  # Limit to 5 files
        if isinstance(audio_file, str):
            src_path = audio_file
        else:
            src_path = audio_file.name
        
        ext = Path(src_path).suffix
        dest_path = voice_dir / f"sample_{i}{ext}"
        shutil.copyfile(src_path, dest_path)
        saved_files.append(str(dest_path))
    
    # Update library metadata
    library[voice_name] = {
        "files": saved_files,
        "created": datetime.now().isoformat(),
        "num_samples": len(saved_files)
    }
    save_voice_library(library)
    
    return f"✅ Voice '{voice_name}' saved with {len(saved_files)} samples!"

def get_voice_files(voice_name: str) -> List[str]:
    """Get audio files for a voice from library"""
    if voice_name == "Upload new voice...":
        return []
    
    library = load_voice_library()
    if voice_name in library:
        return library[voice_name]["files"]
    return []

def process_text_and_generate(input_text, voice_selection, ref_audio_files, speed, enhance_speech, temperature, top_p, top_k, repetition_penalty, language):
    """Process text and generate audio with proper state management."""
    log_messages = ""
    
    # Validate input text
    if not input_text or len(input_text.strip()) == 0:
        log_messages += "⚠️ Please provide some text to convert!\n"
        return None, log_messages, gr.update()
    
    # Get audio files based on selection
    audio_files = []
    if voice_selection and voice_selection != "Upload new voice...":
        audio_files = get_voice_files(voice_selection)
        log_messages += f"🎤 Using saved voice: {voice_selection}\n"
    elif ref_audio_files:
        audio_files = [f.name if hasattr(f, 'name') else f for f in ref_audio_files[:5]]
        log_messages += f"🎤 Using uploaded audio files\n"
    
    if not audio_files:
        log_messages += "⚠️ Please select a voice or upload reference audio!\n"
        return None, log_messages, gr.update()

    # Create unique request ID to prevent caching
    request_id = uuid.uuid4().hex
    
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
        request_id=request_id  # Force unique request
    )

    try:
        log_messages += f"🎤 Generating speech...\n"
        log_messages += f"📝 Text: {input_text[:100]}{'...' if len(input_text) > 100 else ''}\n"
        log_messages += f"🌍 Language: {language}\n"
        log_messages += f"🔑 Request ID: {request_id[:8]}...\n"
        
        with torch.no_grad():
            output = tts.generate_speech(request)
            if output:
                if speed != 1:
                    output = output.change_speed(speed)
                
                # Convert to proper format (int16)
                audio_array = output.array
                if audio_array.dtype == np.float16 or audio_array.dtype == np.float32:
                    # Normalize to [-1, 1] if needed
                    if audio_array.max() > 1.0 or audio_array.min() < -1.0:
                        audio_array = audio_array / np.abs(audio_array).max()
                    # Convert to int16
                    audio_array = (audio_array * 32767).astype(np.int16)
                
                # Save to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"argentinian_tts_{timestamp}_{request_id[:8]}.wav"
                output_path = output_dir / output_filename
                
                # Save using torchaudio
                audio_tensor = torch.from_numpy(audio_array).float() / 32767.0
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                torchaudio.save(str(output_path), audio_tensor, output.sample_rate)
                
                log_messages += f"✅ Successfully generated audio!\n"
                log_messages += f"📊 Sample rate: {output.sample_rate} Hz\n"
                log_messages += f"⏱️ Duration: {len(audio_array) / output.sample_rate:.2f} seconds\n"
                log_messages += f"💾 Saved to: {output_filename}\n"
                
                # Return audio and update file display
                return (output.sample_rate, audio_array), log_messages, gr.update(value=str(output_path), visible=True)
            else:
                log_messages += "❌ No output was generated. Check that the model was correctly loaded\n"
                return None, log_messages, gr.update()
    except Exception as e:
        import traceback
        logger.error(f"Error: {e}\n{traceback.format_exc()}")
        log_messages += f"❌ An error occurred: {e}\n"
        return None, log_messages, gr.update()

def build_gradio_ui():
    """Builds and launches the Gradio UI for Argentinian Spanish TTS."""
    
    # Example texts in Argentinian Spanish
    example_texts = [
        "¡Che, boludo! ¿Cómo andás? Todo bien por acá.",
        "Este modelo de texto a voz está re copado, funciona bárbaro.",
        "¿Viste lo que pasó ayer? Fue una locura total, te lo juro.",
        "Mirá, la verdad es que no tengo ni idea de qué hacer con esto.",
        "Dale, vamos a tomar unos mates y charlamos un rato.",
        "La verdad que no entiendo nada, pero bueno, vamos a ver qué onda.",
        "Che, ¿me pasás la sal? Gracias, sos un genio.",
        "Esto está re piola, me encanta cómo suena el acento argentino.",
    ]
    
    with gr.Blocks(title="Argentinian Spanish TTS", theme="soft") as ui:

        gr.Markdown(
            """
            # 🇦🇷 Argentinian Spanish XTTS-v2 TTS Demo
            
            Convert text to speech with authentic Argentinian Spanish accent!
            
            **Features:**
            - 🎭 Voice cloning from reference audio
            - 📚 Voice library - save and reuse voices
            - 💾 Automatic audio saving
            - 🗣️ Natural Rioplatense accent
            - ⚡ High-quality neural TTS
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
                    
                    with gr.Row():
                        voice_selection = gr.Dropdown(
                            label="🎭 Select Voice from Library",
                            choices=get_voice_choices(),
                            value="Upload new voice...",
                            interactive=True
                        )
                        refresh_voices_btn = gr.Button("🔄 Refresh", size="sm")
                    
                    ref_audio_files = gr.Files(
                        label="📁 Or Upload New Reference Audio (5-30 seconds)",
                        file_types=["audio"],
                        visible=True
                    )
                    
                    with gr.Accordion("💾 Save Voice to Library", open=False):
                        voice_name_input = gr.Textbox(
                            label="Voice Name",
                            placeholder="e.g., 'Maria', 'Juan', 'My Voice'..."
                        )
                        save_voice_btn = gr.Button("💾 Save Voice", variant="secondary")
                        save_voice_status = gr.Textbox(label="Status", interactive=False)
                    
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
                            info="Higher = more creative"
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
                            value="es"
                        )
                    
                    generate_button = gr.Button("🎙️ Generate Speech", variant="primary", size="lg")
                    
                with gr.Column():
                    audio_output = gr.Audio(label="🔊 Generated Audio", type="numpy")
                    output_file_path = gr.Textbox(
                        label="💾 Saved File Path",
                        interactive=False,
                        visible=False
                    )
                    log_output = gr.Textbox(label="📋 Log Output", lines=12)

            # Event handlers
            generate_button.click(
                process_text_and_generate,
                inputs=[input_text, voice_selection, ref_audio_files, speed, enhance_speech, temperature, top_p, top_k, repetition_penalty, language],
                outputs=[audio_output, log_output, output_file_path],
            )
            
            save_voice_btn.click(
                save_voice_to_library,
                inputs=[ref_audio_files, voice_name_input],
                outputs=[save_voice_status]
            )
            
            refresh_voices_btn.click(
                lambda: gr.update(choices=get_voice_choices()),
                outputs=[voice_selection]
            )
            
            # Example buttons
            gr.Markdown("### 📝 Example Texts (Click to use)")
            with gr.Row():
                for i in range(min(4, len(example_texts))):
                    btn = gr.Button(f"Example {i+1}", size="sm")
                    btn.click(lambda x=example_texts[i]: x, outputs=input_text)

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
                    
                    save_recorded_voice = gr.Textbox(
                        label="💾 Save as Voice Name (optional)",
                        placeholder="e.g., 'My Voice'"
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
                    output_file_path_mic = gr.Textbox(
                        label="💾 Saved File Path",
                        interactive=False,
                        visible=False
                    )
                    log_output_mic = gr.Textbox(label="📋 Log Output", lines=12)

            def process_mic_and_generate(input_text_mic, mic_ref_audio, save_recorded_voice, speed_mic, enhance_speech_mic, temperature_mic, top_p_mic, top_k_mic, repetition_penalty_mic, language_mic):
                if not mic_ref_audio:
                    return None, "⚠️ Please record an audio!", gr.update(), gr.update()
                
                import hashlib
                data = str(time.time()).encode("utf-8")
                hash_val = hashlib.sha1(data).hexdigest()[:10]
                output_path = temp_dir / f"mic_{hash_val}.wav"

                torch_audio = torch.from_numpy(mic_ref_audio[1].astype(float))
                try:
                    torchaudio.save(str(output_path), torch_audio.unsqueeze(0), mic_ref_audio[0])
                    
                    # Save to library if name provided
                    save_status = ""
                    if save_recorded_voice and save_recorded_voice.strip():
                        save_status = save_voice_to_library([str(output_path)], save_recorded_voice.strip())
                    
                    audio, log, file_path = process_text_and_generate(
                        input_text_mic, "Upload new voice...", [str(output_path)], speed_mic, enhance_speech_mic,
                        temperature_mic, top_p_mic, top_k_mic, repetition_penalty_mic, language_mic
                    )
                    
                    if save_status:
                        log = save_status + "\n" + log
                    
                    return audio, log, file_path, gr.update(choices=get_voice_choices())
                    
                except Exception as e:
                    logger.error(f"Error saving audio file: {e}")
                    return None, f"❌ Error saving audio file: {e}", gr.update(), gr.update()

            generate_button_mic.click(
                process_mic_and_generate,
                inputs=[input_text_mic, mic_ref_audio, save_recorded_voice, speed_mic, enhance_speech_mic, temperature_mic, top_p_mic, top_k_mic, repetition_penalty_mic, language_mic],
                outputs=[audio_output_mic, log_output_mic, output_file_path_mic, voice_selection],
            )

        with gr.Tab("📚 Voice Library"):
            gr.Markdown("## Manage Your Saved Voices")
            
            with gr.Row():
                library_display = gr.JSON(label="Saved Voices", value=load_voice_library())
                
            with gr.Row():
                refresh_library_btn = gr.Button("🔄 Refresh Library", variant="secondary")
                
            refresh_library_btn.click(
                lambda: load_voice_library(),
                outputs=[library_display]
            )
            
            gr.Markdown(
                """
                ### 💡 Tips for Voice Library:
                - Upload 5-30 seconds of clear speech
                - Use descriptive names for your voices
                - Saved voices persist across sessions
                - Voice files are stored in: `/home/op/Auralis/voice_library/`
                """
            )

        gr.Markdown(
            """
            ---
            ### 🇦🇷 About Argentinian Spanish
            
            This model includes authentic Argentinian characteristics:
            - **Voseo**: "vos" instead of "tú" 
            - **Rioplatense accent**: Buenos Aires pronunciation
            - **Local vocabulary**: "che", "boludo", "copado", "bárbaro"
            
            ### 💾 Generated Audio
            All generated audio is automatically saved to: `/home/op/Auralis/generated_audio/`
            """
        )

    return ui

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🇦🇷 Argentinian Spanish XTTS-v2 Gradio Interface v2")
    print("="*50 + "\n")
    
    ui = build_gradio_ui()
    ui.launch(
        debug=True,
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True
    )
