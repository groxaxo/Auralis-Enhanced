"""Example: FlashSR Audio Super-Resolution

This example demonstrates the FlashSR audio super-resolution feature in Auralis Enhanced.
FlashSR automatically upscales TTS output from 24kHz to 48kHz for professional-quality audio.
"""

from auralis import TTS, TTSRequest

def main():
    print("🎵 Auralis Enhanced - FlashSR Audio Super-Resolution Example")
    print("=" * 70)
    
    # Initialize TTS
    print("\n📦 Loading TTS model...")
    tts = TTS().from_pretrained(
        "AstraMindAI/xttsv2",
        gpt_model='AstraMindAI/xtts2-gpt'
    )
    
    # Example 1: Standard usage with FlashSR enabled by default
    print("\n✅ Example 1: FlashSR enabled by default (48kHz output)")
    print("-" * 70)
    
    request = TTSRequest(
        text="Hello! This is Auralis Enhanced with FlashSR audio super-resolution.",
        speaker_files=['examples/sample_speaker.wav']
        # apply_flashsr=True is the default
    )
    
    output = tts.generate_speech(request)
    print(f"   Output sample rate: {output.sample_rate} Hz")
    print(f"   FlashSR applied: {output._flashsr_applied}")
    output.save('output_with_flashsr.wav')
    print("   ✓ Saved to: output_with_flashsr.wav (48kHz)")
    
    # Example 2: Disable FlashSR for faster processing (24kHz output)
    print("\n⚡ Example 2: FlashSR disabled for speed (24kHz output)")
    print("-" * 70)
    
    request_no_flashsr = TTSRequest(
        text="This output skips FlashSR for faster processing.",
        speaker_files=['examples/sample_speaker.wav'],
        apply_flashsr=False  # Explicitly disable
    )
    
    output_no_flashsr = tts.generate_speech(request_no_flashsr)
    print(f"   Output sample rate: {output_no_flashsr.sample_rate} Hz")
    print(f"   FlashSR applied: {output_no_flashsr._flashsr_applied}")
    output_no_flashsr.save('output_without_flashsr.wav')
    print("   ✓ Saved to: output_without_flashsr.wav (24kHz)")
    
    # Example 3: Manual FlashSR application
    print("\n🔧 Example 3: Manual FlashSR application")
    print("-" * 70)
    
    # Generate without FlashSR first
    request_manual = TTSRequest(
        text="Applying FlashSR manually after generation.",
        speaker_files=['examples/sample_speaker.wav'],
        apply_flashsr=False
    )
    
    output_manual = tts.generate_speech(request_manual)
    print(f"   Initial sample rate: {output_manual.sample_rate} Hz")
    
    # Apply FlashSR manually
    output_enhanced = output_manual.apply_super_resolution()
    print(f"   Enhanced sample rate: {output_enhanced.sample_rate} Hz")
    print(f"   FlashSR applied: {output_enhanced._flashsr_applied}")
    output_enhanced.save('output_manual_flashsr.wav')
    print("   ✓ Saved to: output_manual_flashsr.wav (48kHz)")
    
    # Example 4: Quality comparison
    print("\n📊 Example 4: File size comparison")
    print("-" * 70)
    
    import os
    
    files = [
        ('output_without_flashsr.wav', '24kHz (no FlashSR)'),
        ('output_with_flashsr.wav', '48kHz (with FlashSR)'),
    ]
    
    for filename, description in files:
        if os.path.exists(filename):
            size_mb = os.path.getsize(filename) / (1024 * 1024)
            print(f"   {filename}: {size_mb:.2f} MB ({description})")
    
    print("\n" + "=" * 70)
    print("✅ FlashSR demonstration complete!")
    print("\n💡 Key takeaways:")
    print("   • FlashSR is enabled by default for professional 48kHz output")
    print("   • Minimal performance impact (200-400x real-time processing)")
    print("   • Can be disabled with apply_flashsr=False for speed")
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
        print("   3. Internet connection to download FlashSR model (first run)")
