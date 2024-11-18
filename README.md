# Auralis (_/auˈralis/_): Blazing-Fast Text-to-Speech Inference Engine

Hey there! Welcome to **Auralis**, a super-speedy text-to-speech inference engine that'll blow your socks off. 🚀

Imagine turning the entire first Harry Potter book into speech in just **10 minutes** using less than **10GB of VRAM** on a single NVIDIA 3090. Sounds crazy, right? Well, that's exactly what Auralis does!

## What's This All About?

Auralis is a high-performance TTS (Text-to-Speech) engine that leverages a two-step scheduler and the power of **VLLM** to serve TTS models at lightning speed. Whether you're dealing with short sentences or entire novels, this bad boy handles it all with ease.

## Why Should You Care?

- **Speed Demon**: Processes long texts in a fraction of the time.
- **Resource-Friendly**: Runs on consumer-grade GPUs without hogging all the VRAM.
- **Asynchronous Magic**: Handles multiple requests without breaking a sweat.
- **Easy to Use**: Simple API that won't make you pull your hair out.
- **Extensible**: Add your own models and tweak to your heart's content.

## How Does It Work?

Auralis uses a nifty two-phase process:

1. **Token Generation**: Converts your input text into tokens using VLLM's async engine.
2. **Speech Generation**: Transforms those tokens into silky-smooth audio using a TTS Engine.

By splitting the workload, we keep things efficient and lightning-fast. Think of it like a well-oiled assembly line for speech synthesis.

## Getting Started

### Installation

```python
pip install auralis
```

### Usage

Here's how you can get started with generating speech:

```python
from auralis import TTS, TTSRequest

# Initialize the TTS engine
tts = TTS() # you can set a max concurrency level here to speed everything up, i.e. TTS(scheduler_max_concurrency=36).from_pretrained('AstraMindAI/xtts2-gpt')
tts.from_pretrained('AstraMindAI/xtts2-gpt')

# Create a TTS request
request = TTSRequest(
    text="Hello, world! This is Auralis speaking.",
    language='en',
    speaker_files=['path/to/your/speaker.wav'],  # Path to a reference audio file
    stream=False  # Set to True if you want streaming output
)

# Generate speech
output = tts.generate_speech(request)

# Save the audio to a file
output.save('output.wav')
```

### Streaming Long Texts

Got a long text? No worries! Auralis can be run as a generator so you don't have to wait forever.

```python
from auralis import TTS, TTSRequest

# Initialize the TTS engine
tts = TTS()
tts.from_pretrained('AstraMindAI/xtts2-gpt')

# Create a TTS request
request = TTSRequest(
    text="Hello, world! This is a very long text that will take a while to generate.",
    language='en',
    speaker_files=['path/to/your/speaker.wav'],  # Path to a reference audio file
    stream=True 
)

# Generate speech
audio_generator = tts.generate_speech(request)

# Process the audio chunks as they come in
for audio_chunk in audio_generator:
    # Do something with each chunk (like saving or playing it)
    #audio_chunk.play()
    #audio_chunk.save(f'output_{i}.wav')
    pass
```

###  `TTSOutput` Output Class
We were tired of complex handling of model output, so we created a unified output object with some of the most important features when working with audio generation tools.
Some of them are:

1. **Format Conversion Utilities:**
```python
- to_tensor(): Converts numpy array to torch tensor
- to_bytes(): Converts audio to bytes format (supports 'wav' or 'raw')
- from_tensor(): Creates TTSOutput from torch tensor
- from_file(): Creates TTSOutput from audio file
```

2. **Audio Processing Utilities:**
```python
- combine_outputs(): Combines multiple TTSOutput instances into one
- resample(): Creates new TTSOutput with resampled audio
- get_info(): Returns (number of samples, sample rate, duration)
```

3. **File Handling:**
```python
- save(): Saves audio to file with optional resampling and format specification
```

4. **Playback Utilities:**
```python
- play(): Plays audio through default sound device
- display(): Shows audio player in Jupyter notebook
- preview(): Smart play method that tries display() first, then falls back to play()
```

### Adding Your Own Models

Feeling adventurous? You can add your own models to Auralis. Check out [ADDING_MODELS.md](docs/ADDING_MODELS.md) for a step-by-step guide.

## Under the Hood

Want to know how the magic happens? Dive into the technical details on our [blog post](https://www.astramind.ai/blog/Auralis).

## Contributing

We're a small dev team working on this project, and we'd love your feedback or contributions! Feel free to open issues or submit pull requests.

## License

This project is licensed under the Apache-2 License.
