"""Generate speech with Auralis' optional MLX backend on Apple Silicon."""

from auralis import TTS, TTSRequest

MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"


tts = TTS(backend="mlx").from_pretrained(
    MODEL,
    voice="Chelsie",
)

request = TTSRequest(
    text="Auralis now runs natively on Apple Silicon through MLX.",
    speaker_files=None,
    language="en",
)

output = tts.generate_speech(request)
output.save("auralis_mlx.wav")
print(f"Saved {output.get_info()[2]:.2f}s of audio to auralis_mlx.wav")
