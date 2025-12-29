# Welcome to Auralis Enhanced

**Auralis Enhanced** is a production-ready, high-performance text-to-speech (TTS) library that improves upon x-tts with optimized inference, FlashSR audio super-resolution for professional 48kHz output, enhanced features, and comprehensive deployment documentation.

!!! success "Auralis Enhanced"
    This is the enhanced fork with FlashSR audio super-resolution, network deployment capabilities, production-ready configurations, and complete server deployment guides.

## Features

- **🎵 FlashSR Audio Super-Resolution**: Automatic 48kHz professional-quality output (200-400x real-time, 2MB model)
- **High Performance**: Optimized inference with VLLM integration
- **Multi-Language Support**: Support for 17+ languages out of the box
- **Easy Integration**: Simple API for both basic and advanced use cases
- **Flexible Architecture**: Easy to extend with new models and components
- **Production Ready**: Built-in logging, metrics, and error handling
- **Network Deployment**: Pre-configured for `0.0.0.0` binding - accessible from any network interface
- **Comprehensive Guides**: Complete production deployment documentation (systemd, Docker, Nginx)

## Recent Updates

!!! info "Latest Changes"
    **December 2024**: FlashSR integration - Automatic audio super-resolution from 24kHz to 48kHz for professional broadcast-quality output using the ultra-fast FlashSR model (200-400x real-time processing, negligible overhead)
    
    **October 2024**: Repository cleanup - removed test audio files and updated `.gitignore` to exclude audio/voice files (`.mp3`, `.wav`, `.opus`) from version control while preserving documentation assets. This keeps the repository lean and focused on code.

## Quick Installation

```bash
pip install auralis
```

## Basic Usage

```python
from auralis import TTS

# Initialize TTS
tts = TTS()

# Generate speech
audio = tts.generate("Hello, world!")

# Save to file
audio.save("output.wav")
```

## Why Auralis Enhanced?

!!! tip "Key Benefits"
    - 🚀 **Faster Inference**: Optimized for speed with VLLM
    - 🌍 **Language Support**: 17+ languages supported
    - 🎯 **Easy to Use**: Simple, intuitive API
    - 📊 **Monitoring**: Built-in logging and metrics
    - 🔧 **Extensible**: Easy to add new models
    - 🌐 **Network Ready**: Pre-configured for production server deployment
    - 📚 **Complete Documentation**: Comprehensive deployment and configuration guides

## Project Structure

```
auralis/
├── core/           # Core TTS functionality
├── models/         # Model implementations
│   └── xttsv2/    # XTTSv2 model and components
├── common/         # Shared utilities
└── docs/          # Documentation
```

## Getting Started

Check out our [Quick Start Guide](getting-started/quickstart.md) to begin using Auralis in your projects.

## Documentation

- [Performance Tuning](advanced/performance-tuning.md): Optimize for production
- [Deployment Guide](advanced/deployment.md): Deploy in production
- [Adding Models](advanced/adding-models.md): Extend with your models
- [Using OAI Server](advanced/using-oai-server.md): OpenAI API compatibility
- [FlashSR Integration Analysis](analysis/flashsr-integration-analysis.md): Audio super-resolution compatibility analysis
- [Logging Reference](api/common/logging.md): Logging system
- [Documentation Guide](contributing/documentation.md): Contribute to docs

## Need Help?

- Check the documentation sections above
- Open an issue on GitHub
- Join our community discussions 