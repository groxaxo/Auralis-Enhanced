#!/usr/bin/env python3
"""Test script for the Argentinian Spanish XTTS server"""

import base64
import requests
import json

# Read the sample audio file
with open('/home/op/Auralis/examples/speech.mp3', 'rb') as f:
    audio_data = f.read()
    audio_base64 = base64.b64encode(audio_data).decode('utf-8')

# Create the request
request_data = {
    "input": "Hola, ¿cómo estás? Este es un modelo de texto a voz entrenado con acento argentino. ¡Che, qué bueno que funciona!",
    "model": "tts-1",
    "voice": [audio_base64],
    "response_format": "wav",
    "language": "es"
}

print("Sending request to server...")
response = requests.post(
    'http://localhost:5000/v1/audio/speech',
    json=request_data,
    timeout=60
)

if response.status_code == 200:
    # Save the audio output
    output_file = '/home/op/Auralis/test_output_argentinian.wav'
    with open(output_file, 'wb') as f:
        f.write(response.content)
    print(f"✓ Success! Audio saved to: {output_file}")
    print(f"  Audio size: {len(response.content)} bytes")
else:
    print(f"✗ Error: {response.status_code}")
    print(f"  Response: {response.text}")
    try:
        import json
        error_data = json.loads(response.text)
        if 'traceback' in error_data:
            print(f"\nTraceback:\n{error_data['traceback']}")
    except:
        pass
