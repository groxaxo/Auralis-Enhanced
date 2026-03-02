# Auralis Enhanced - Futuristic Web UI

A modern, futuristic web interface for Auralis Enhanced TTS with voice cloning and NovaSR audio enhancement.

## Features

- 🎨 **Futuristic Cyberpunk Design** - Glass morphism, neon accents, and smooth animations
- 🎙️ **Voice Cloning** - Record or upload audio samples for voice cloning
- 🎵 **Audio Visualization** - Interactive waveform player with WaveSurfer.js
- ⚡ **Real-time Generation** - Progress tracking with stage indicators
- 🎚️ **Advanced Parameters** - Fine-tune temperature, top-p, top-k, and more
- 🔊 **NovaSR Enhancement** - Toggle 48kHz professional audio quality
- 📚 **Voice Library** - Save and manage your voice profiles
- 🌙 **Dark Mode** - Optimized for dark environments

## Tech Stack

- **Next.js 15** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **shadcn/ui** - Beautiful, accessible components
- **Framer Motion** - Smooth animations
- **WaveSurfer.js** - Audio waveform visualization
- **Zustand** - Lightweight state management

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn
- Auralis Enhanced backend running on port 8000

### Installation

```bash
# Install dependencies
npm install

# Start development server (auto port allocation)
npm run dev
```

The app will start on an available port (displayed in terminal).

### Production Build

```bash
npm run build
npm start
```

## Project Structure

```
frontend/
├── src/
│   ├── app/                    # Next.js App Router pages
│   │   ├── page.tsx           # Main generation page
│   │   ├── voice-cloning/     # Voice cloning interface
│   │   ├── voice-library/     # Saved voice profiles
│   │   └── settings/          # Configuration
│   ├── components/
│   │   ├── ui/                # shadcn/ui components
│   │   ├── audio/             # Audio-specific components
│   │   └── layout/            # Layout components
│   ├── lib/                   # Utilities and API client
│   ├── stores/                # Zustand state management
│   ├── types/                 # TypeScript definitions
│   └── hooks/                 # Custom React hooks
├── tailwind.config.ts         # Tailwind configuration
└── package.json
```

## Configuration

The API base URL can be configured in Settings or by modifying `src/stores/index.ts`:

```typescript
apiBaseUrl: 'http://localhost:8000'
```

## Features in Detail

### Audio Generation
- Text input with character/word count
- Multiple reference audio support
- Language selection (18+ languages)
- Speed control (0.5x - 2.0x)

### Voice Cloning
- Record directly from browser (up to 30 seconds)
- Upload audio files (WAV, MP3, OGG, WebM, M4A, FLAC)
- Preview before saving
- Voice profile naming

### Advanced Settings
- **Temperature** - Controls randomness (0.5 - 1.0)
- **Top P** - Nucleus sampling threshold (0.5 - 1.0)
- **Top K** - Top-k sampling (0 - 100)
- **Repetition Penalty** - Prevents loops (1.0 - 10.0)

### NovaSR Enhancement
Toggle to upscale audio from 24kHz to professional 48kHz quality.

## API Integration

The frontend communicates with the Auralis backend via REST API:

- `POST /v1/audio/speech` - Generate speech
- `GET /health` - Health check
- `POST /v1/voices/upload` - Upload voice files
- `GET /v1/voices` - List available voices

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## License

Apache 2.0 - Same as Auralis Enhanced
