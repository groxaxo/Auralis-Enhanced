"""Example: NovaSR Audio Super-Resolution

This example demonstrates the NovaSR audio super-resolution feature in Auralis Enhanced.
NovaSR automatically upscales TTS output from 24kHz to 48kHz for professional-quality audio.
"""

from auralis import TTS, TTSRequest


def main():
    print("🎵 Auralis Enhanced - NovaSR Audio Super-Resolution Example")
    print("=" * 70)

    print("\n📦 Loading TTS model...")
    tts = TTS().from_pretrained("AstraMindAI/xttsv2", gpt_model="AstraMindAI/xtts2-gpt")

    print("\n✅ Example 1: NovaSR enabled (48kHz output)")
    print("-" * 70)

    request = TTSRequest(
        text="Hello! This is Auralis Enhanced with NovaSR audio super-resolution.",
        speaker_files=["examples/sample_speaker.wav"],
        apply_novasr=True,
    )

    output = tts.generate_speech(request)
    print(f"   Output sample rate: {output.sample_rate} Hz")
    print(f"   NovaSR applied: {output._novasr_applied}")
    output.save("output_with_novasr.wav")
    print("   ✓ Saved to: output_with_novasr.wav (48kHz)")

    print("\n⚡ Example 2: NovaSR disabled for speed (24kHz output)")
    print("-" * 70)

    request_no_novasr = TTSRequest(
        text="This output skips NovaSR for faster processing.",
        speaker_files=["examples/sample_speaker.wav"],
        apply_novasr=False,
    )

    output_no_novasr = tts.generate_speech(request_no_novasr)
    print(f"   Output sample rate: {output_no_novasr.sample_rate} Hz")
    print(f"   NovaSR applied: {output_no_novasr._novasr_applied}")
    output_no_novasr.save("output_without_novasr.wav")
    print("   ✓ Saved to: output_without_novasr.wav (24kHz)")

    print("\n🔧 Example 3: Manual NovaSR application")
    print("-" * 70)

    request_manual = TTSRequest(
        text="Applying NovaSR manually after generation.",
        speaker_files=["examples/sample_speaker.wav"],
        apply_novasr=False,
    )

    output_manual = tts.generate_speech(request_manual)
    print(f"   Initial sample rate: {output_manual.sample_rate} Hz")

    output_enhanced = output_manual.apply_super_resolution()
    print(f"   Enhanced sample rate: {output_enhanced.sample_rate} Hz")
    print(f"   NovaSR applied: {output_enhanced._novasr_applied}")
    output_enhanced.save("output_manual_novasr.wav")
    print("   ✓ Saved to: output_manual_novasr.wav (48kHz)")

    print("\n📊 Example 4: File size comparison")
    print("-" * 70)

    import os

    files = [
        ("output_without_novasr.wav", "24kHz (no NovaSR)"),
        ("output_with_novasr.wav", "48kHz (with NovaSR)"),
    ]

    for filename, description in files:
        if os.path.exists(filename):
            size_mb = os.path.getsize(filename) / (1024 * 1024)
            print(f"   {filename}: {size_mb:.2f} MB ({description})")

    print("\n" + "=" * 70)
    print("✅ NovaSR demonstration complete!")
    print("\n💡 Key takeaways:")
    print("   • NovaSR provides professional 48kHz output")
    print("   • Extremely fast (3600x real-time processing)")
    print("   • Tiny model size (~52KB)")
    print("   • Can be disabled with apply_novasr=False for speed")
    print("   • Perfect for audiobooks, podcasts, and professional production")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure you have:")
        print("   1. Installed all dependencies: pip install -r requirements.txt")
        print("   2. A reference speaker file at examples/sample_speaker.wav")
        print("   3. Internet connection to download NovaSR model (first run)")
