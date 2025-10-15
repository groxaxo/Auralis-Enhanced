#!/usr/bin/env python3
"""Quick test script for the Argentinian Spanish XTTS server"""

import base64
import requests
import sys

PORT = 5000

print("🧪 Testing Argentinian Spanish XTTS-v2 Server...")
print(f"   Endpoint: http://localhost:{PORT}")
print()

# Read the sample audio file
try:
    with open('/home/op/Auralis/examples/speech.mp3', 'rb') as f:
        audio_data = f.read()
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    print("✓ Reference audio loaded")
except Exception as e:
    print(f"✗ Failed to load reference audio: {e}")
    sys.exit(1)

# Test text in Spanish with Argentinian expressions
test_text = "¡Che, boludo! ¿Cómo andás? Este modelo de texto a voz está re copado. ¡Funciona bárbaro!"

# Create the request
request_data = {
    "input": test_text,
    "model": "tts-1",
    "voice": [audio_base64],
    "response_format": "wav",
    "language": "es",
    "temperature": 0.75
}

print(f"✓ Request prepared")
print(f"   Text: {test_text}")
print()

try:
    print("⏳ Sending request to server...")
    response = requests.post(
        f'http://localhost:{PORT}/v1/audio/speech',
        json=request_data,
        timeout=60
    )
    
    if response.status_code == 200:
        # Save the audio output
        output_file = '/home/op/Auralis/test_argentinian_output.wav'
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        print()
        print("=" * 50)
        print("✅ SUCCESS! Server is working correctly!")
        print("=" * 50)
        print()
        print(f"📁 Audio saved to: {output_file}")
        print(f"📊 Audio size: {len(response.content):,} bytes")
        print()
        print("🎵 To play the audio:")
        print(f"   ffplay {output_file}")
        print(f"   or")
        print(f"   aplay {output_file}")
        print()
        
    else:
        print()
        print(f"❌ Error: HTTP {response.status_code}")
        print(f"   Response: {response.text[:500]}")
        sys.exit(1)
        
except requests.exceptions.ConnectionError:
    print()
    print("❌ Connection Error: Server is not running")
    print()
    print("To start the server:")
    print("   ./launch_argentinian_spanish_server.sh")
    print()
    sys.exit(1)
    
except requests.exceptions.Timeout:
    print()
    print("❌ Timeout: Server took too long to respond")
    print("   This might indicate the server is overloaded")
    sys.exit(1)
    
except Exception as e:
    print()
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
