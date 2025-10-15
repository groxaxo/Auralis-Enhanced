# Argentinian Spanish XTTS-v2 TTS Server

This directory contains scripts to manage the Argentinian Spanish XTTS-v2 text-to-speech server.

## Model Information

- **Model**: marianbasti/XTTS-v2-argentinian-spanish
- **Source**: Hugging Face
- **Architecture**: XTTS-v2 (Coqui TTS)
- **Language**: Spanish (Argentinian accent)
- **Converted Location**: `/home/op/Auralis/converted_models/argentinian_spanish/`

## Quick Start

### Start the Server
```bash
./launch_argentinian_spanish_server.sh
```

### Check Server Status
```bash
./status_argentinian_spanish_server.sh
```

### Stop the Server
```bash
./stop_argentinian_spanish_server.sh
```

## Server Configuration

- **Host**: 0.0.0.0 (accessible from all interfaces)
- **Port**: 5000
- **Max Concurrency**: 8
- **Log File**: `argentinian_spanish_server.log`

## API Endpoints

### Generate Speech
```bash
POST http://localhost:5000/v1/audio/speech
```

**Request Body** (JSON):
```json
{
  "input": "Hola, ¿cómo estás? Este es un modelo argentino.",
  "model": "tts-1",
  "voice": ["<base64-encoded-audio>"],
  "response_format": "wav",
  "language": "es"
}
```

### API Documentation
Interactive API documentation available at:
```
http://localhost:5000/docs
```

## Example Usage

### Python Example
```python
import base64
import requests

# Read reference audio
with open('reference_voice.mp3', 'rb') as f:
    audio_base64 = base64.b64encode(f.read()).decode('utf-8')

# Make request
response = requests.post(
    'http://localhost:5000/v1/audio/speech',
    json={
        "input": "¡Che, qué bueno que funciona este modelo argentino!",
        "model": "tts-1",
        "voice": [audio_base64],
        "response_format": "wav",
        "language": "es"
    }
)

# Save output
with open('output.wav', 'wb') as f:
    f.write(response.content)
```

### cURL Example
```bash
# Prepare base64 audio
AUDIO_BASE64=$(base64 -w 0 reference_voice.mp3)

# Make request
curl -X POST http://localhost:5000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": \"Hola desde Argentina\",
    \"model\": \"tts-1\",
    \"voice\": [\"$AUDIO_BASE64\"],
    \"response_format\": \"wav\",
    \"language\": \"es\"
  }" \
  --output output.wav
```

## Monitoring

### View Live Logs
```bash
tail -f argentinian_spanish_server.log
```

### Check GPU Usage
```bash
nvidia-smi
```

### Check Memory Usage
```bash
ps aux | grep oai_server
```

## Troubleshooting

### Server Won't Start
1. Check if conda environment exists:
   ```bash
   conda env list | grep auralis_env
   ```

2. Check if port 5000 is already in use:
   ```bash
   netstat -tuln | grep 5000
   ```

3. Check logs for errors:
   ```bash
   tail -50 argentinian_spanish_server.log
   ```

### Server Crashes
- Check available GPU memory: `nvidia-smi`
- Reduce `MAX_CONCURRENCY` in the launch script
- Check system resources: `htop`

### Poor Audio Quality
- Ensure reference audio is high quality (16kHz+, clear speech)
- Try adjusting temperature (0.5-1.0) in the request
- Use longer reference audio (5-10 seconds recommended)

## Model Conversion

The model was converted from the original PyTorch checkpoint using:
```bash
python src/auralis/models/xttsv2/utils/checkpoint_converter.py \
  <checkpoint.pth> \
  --output_dir converted_models/argentinian_spanish
```

This creates two components:
- `core_xttsv2/` - Main XTTS model
- `gpt/` - GPT-2 based autoregressive model

## Performance

- **GPU Memory**: ~9 GB
- **Concurrency**: Up to 3x for 1047 token sequences
- **Latency**: ~2-5 seconds for typical sentences
- **GPU Blocks**: 196 (GPU) + 2184 (CPU)

## Notes

- The server uses vLLM for efficient inference
- Flash Attention is enabled for faster processing
- The model supports voice cloning with reference audio
- Optimal reference audio length: 5-30 seconds
- Supports streaming output for real-time applications

## Support

For issues or questions:
- Check the main Auralis documentation
- Review the conversion logs
- Inspect server logs for detailed error messages
