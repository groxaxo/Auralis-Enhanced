# 🎉 Gradio Interface is Ready!

## 🇦🇷 Argentinian Spanish XTTS-v2 Gradio UI

The Gradio web interface for the Argentinian Spanish TTS model is now running!

---

## 🌐 Access the Interface

**URL**: http://localhost:7861

Simply open this URL in your web browser to access the interactive TTS interface.

---

## ✨ Features

### 🎤 Text to Speech Tab
- **Voice Cloning**: Upload reference audio files (5-30 seconds)
- **Argentinian Examples**: Pre-loaded example texts with authentic expressions
- **Advanced Controls**: Temperature, Top-P, Top-K, repetition penalty
- **Speed Control**: Adjust playback speed (0.5x - 2.0x)
- **Audio Enhancement**: Optional speech enhancement for reference audio

### 🎙️ Record & Clone Tab
- **Microphone Recording**: Record your voice directly in the browser
- **Instant Cloning**: Use your recorded voice as reference
- **Real-time Generation**: Generate speech with your cloned voice
- **Same Advanced Controls**: All the same parameters as Text to Speech

---

## 🚀 Quick Start

1. **Open the interface**: http://localhost:7861
2. **Upload reference audio** (or record your voice)
3. **Enter Spanish text** (try the example buttons!)
4. **Click "Generate Speech"**
5. **Listen to the result!**

---

## 📝 Example Texts (Argentinian Spanish)

The interface includes authentic Argentinian expressions:

1. "¡Che, boludo! ¿Cómo andás? Todo bien por acá."
2. "Este modelo de texto a voz está re copado, funciona bárbaro."
3. "¿Viste lo que pasó ayer? Fue una locura total, te lo juro."
4. "Mirá, la verdad es que no tengo ni idea de qué hacer con esto."
5. "Dale, vamos a tomar unos mates y charlamos un rato."

---

## 🎯 Tips for Best Results

### Reference Audio
- **Duration**: 5-30 seconds of clear speech
- **Quality**: High quality audio (16kHz or higher)
- **Content**: Natural speech, not singing or shouting
- **Background**: Minimal background noise

### Text Input
- **Language**: Spanish (Argentinian expressions work best)
- **Length**: Works with any length, but shorter is faster
- **Punctuation**: Use proper punctuation for better intonation

### Parameters
- **Temperature** (0.5-1.0): Higher = more creative/varied
- **Top P** (0.5-1.0): Controls diversity of output
- **Top K** (0-100): Limits vocabulary choices
- **Repetition Penalty** (1.0-10.0): Prevents repetitive speech

---

## 🛠️ Management

### Start Gradio
```bash
./launch_gradio_argentinian.sh
```

### Stop Gradio
```bash
pkill -f 'gradio_argentinian_spanish.py'
```

### View Logs
```bash
tail -f /home/op/Auralis/gradio_argentinian.log
```

### Check Status
```bash
netstat -tuln | grep 7861
```

---

## 📊 Current Status

**Gradio Interface**: ✅ RUNNING

- **Port**: 7861
- **URL**: http://localhost:7861
- **Model**: Argentinian Spanish XTTS-v2
- **Log File**: `/home/op/Auralis/gradio_argentinian.log`

---

## 🔧 Technical Details

### Model Information
- **Base Model**: marianbasti/XTTS-v2-argentinian-spanish
- **Architecture**: XTTS-v2 (Coqui TTS)
- **Dialect**: Rioplatense (Buenos Aires region)
- **Features**: Voice cloning, voseo, Argentinian vocabulary

### System Resources
- **GPU Memory**: ~3 GB
- **Concurrency**: 4 parallel requests
- **GPU Blocks**: 196 (GPU) + 2184 (CPU)
- **Max Sequence Length**: 1047 tokens

---

## 📱 Interface Tabs

### 1. Text to Speech
- Upload audio files
- Enter text manually
- Use example texts
- Adjust all parameters
- Generate and download audio

### 2. Record & Clone
- Record voice with microphone
- Automatic voice cloning
- Instant speech generation
- Same parameter controls

---

## 🎨 Argentinian Spanish Features

The model includes authentic Argentinian characteristics:

- ✅ **Voseo**: "vos" instead of "tú"
- ✅ **Rioplatense Accent**: Buenos Aires pronunciation
- ✅ **Local Vocabulary**: "che", "boludo", "copado", "bárbaro"
- ✅ **Natural Intonation**: Authentic speech patterns
- ✅ **Conjugations**: Proper voseo verb forms

---

## 🐛 Troubleshooting

### Interface won't load?
1. Check if Gradio is running: `netstat -tuln | grep 7861`
2. Check logs: `tail -50 /home/op/Auralis/gradio_argentinian.log`
3. Restart: `./launch_gradio_argentinian.sh`

### Generation fails?
1. Ensure reference audio is provided
2. Check audio format (WAV, MP3, etc.)
3. Try shorter text first
4. Check GPU memory: `nvidia-smi`

### Slow generation?
1. Reduce text length
2. Lower concurrency if needed
3. Check system resources
4. Ensure no other heavy processes running

---

## 📚 Files

### Scripts
- `gradio_argentinian_spanish.py` - Main Gradio application
- `launch_gradio_argentinian.sh` - Launcher script

### Logs
- `gradio_argentinian.log` - Application logs

### Model
- `/home/op/Auralis/converted_models/argentinian_spanish/`

---

## 🎊 Enjoy Testing!

The Gradio interface provides an easy, visual way to test the Argentinian Spanish TTS model. 

Try different voices, experiment with parameters, and have fun creating authentic Argentinian Spanish speech!

**Access now**: http://localhost:7861

---

**Note**: The interface runs on port 7861 to avoid conflicts with other services.
