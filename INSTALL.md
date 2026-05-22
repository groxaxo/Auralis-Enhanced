# Installation Guide

This fork targets **Blackwell (RTX 50-series)** GPUs in addition to the
older Ampere / Ada cards the upstream Auralis supports. The instructions
below cover both.

## 1. System Dependencies

Before installing Python packages, install the system-level audio and
build dependencies for your distribution.

### Ubuntu / Debian
```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-dev build-essential ffmpeg
```

### Fedora / RHEL / CentOS
```bash
sudo dnf install -y portaudio-devel python3-devel gcc gcc-c++ ffmpeg
```

### Arch / Manjaro
```bash
sudo pacman -S --needed portaudio base-devel ffmpeg
```

### macOS
```bash
brew install portaudio ffmpeg
```

## 2. CUDA Toolkit (GPU only)

| GPU family | CUDA toolkit needed |
| :--- | :--- |
| RTX 30xx / 40xx, A100, H100 | 12.4 or newer |
| **RTX 50xx (Blackwell, SM 12.0)** | **12.8 or newer** |

The PyTorch wheels installed in step 4 bundle their own CUDA runtime, so
you do not strictly need a host CUDA install; you only need a NVIDIA
driver new enough to support the wheels (driver ≥ 565 for cu128).

## 3. Python Environment

```bash
conda create -n auralis python=3.10 -y
conda activate auralis
```

Python 3.10 is recommended; 3.11 and 3.12 also work.

## 4. GPU Stack

### Blackwell (RTX 50xx) — recommended

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0
```

### Ampere / Ada / Hopper (RTX 30xx, 40xx, A100, H100)

Either the cu128 wheels above also work, or stick with cu124:

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1
```

### CPU only

```bash
pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0
```

## 5. Auralis Enhanced + vLLM

```bash
pip install -r requirements.txt
pip install -e .
```

This installs vLLM 0.9.2 (the earliest release with prebuilt Blackwell
cubins), `transformers` <4.55 (vLLM's ovis config registration breaks
against newer transformers), and the rest of the runtime dependencies.

## 6. NovaSR (optional, for 48 kHz output)

```bash
pip install git+https://github.com/ysharma3501/NovaSR.git
```

NovaSR is fetched lazily from the Hugging Face Hub on first use and is
only required when `apply_novasr=True` is passed on a request.

## 7. Verify

```bash
python -c "
from auralis import TTS
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
print('Compute capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'n/a')
print('Auralis import OK')
"
```

For a quick end-to-end check:

```bash
python scripts/benchmark.py --conc 4 --n 4 --no-serial
```

The run should complete in under a minute on a Blackwell laptop GPU and
print a `RESULT` line with the measured RTF.

## Troubleshooting

### `CUDA error: no kernel image is available for execution on the device`
The installed PyTorch wheel does not have cubins for your GPU's SM
version. For RTX 50-series, make sure you installed the **cu128** wheels
in step 4, not the default cu124 ones.

### `AssertionError: GPU memory was not properly cleaned up before initializing the vLLM instance`
This is vLLM's optimistic memory-profiler check; on tiny models like the
XTTS GPT it can fire even when nothing is wrong. The XTTS engine patches
the assertion out automatically (see
`src/auralis/models/xttsv2/XTTSv2.py:_patch_vllm_memory_profile_assert`),
so this should not bite users — file an issue if you see it.

### `Failed to load NovaSR model: No module named 'NovaSR'`
NovaSR is an optional dependency; install it with step 6 above. If you
do not need 48 kHz output, leave it off — TTS will keep emitting at 24 kHz.

### `OSError: PortAudio library not found`
Install the system package from step 1 (`portaudio19-dev` on Debian /
Ubuntu, `portaudio-devel` on Fedora / RHEL).

### `ModuleNotFoundError: No module named 'gradio'` / `'openai'`
These are not in the runtime requirements; install them only if you
plan to use the Gradio demo or the OpenAI-compatible server:

```bash
pip install 'gradio>=4.0.0' 'openai>=1.0.0'
```

## Quick Start

After successful installation:

```bash
# Start the OpenAI-compatible backend API server
python -m auralis.entrypoints.oai_server --host 0.0.0.0 --port 8000

# In another terminal, run the Gradio demo UI
cd examples
python gradio_example.py
```

See [`README.md`](README.md) for Python API examples and the throughput
benchmark.
