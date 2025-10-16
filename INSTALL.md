# Installation Guide

## System Dependencies

Before installing Python dependencies, you need to install system-level packages:

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-dev build-essential
```

### Fedora/RHEL/CentOS
```bash
sudo dnf install -y portaudio-devel python3-devel gcc gcc-c++
```

### macOS
```bash
brew install portaudio
```

## Python Environment Setup

1. **Create a new Conda environment:**
   ```bash
   conda create -n auralis_env python=3.10 -y
   ```

2. **Activate the environment:**
   ```bash
   conda activate auralis_env
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

## Troubleshooting

### PortAudio library not found
If you get `OSError: PortAudio library not found`, install the system package:
- **Ubuntu/Debian**: `sudo apt-get install portaudio19-dev`
- **Fedora/RHEL**: `sudo dnf install portaudio-devel`
- **macOS**: `brew install portaudio`

### Missing gradio module
If you get `ModuleNotFoundError: No module named 'gradio'`, run:
```bash
pip install gradio>=4.0.0
```

### Missing openai module
If you get `ModuleNotFoundError: No module named 'openai'`, run:
```bash
pip install openai>=1.0.0
```

## Verifying Installation

Test the installation:
```bash
python -c "from auralis import TTS; print('Auralis installed successfully!')"
```

## Quick Start

After successful installation:

```bash
# Start Backend API Server
python -m auralis.entrypoints.oai_server --host 0.0.0.0 --port 8000

# In another terminal, start Frontend UI
cd examples
python gradio_example.py
```
