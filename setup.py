from pathlib import Path

from setuptools import find_packages, setup


setup(
    name="auralis",
    version="0.2.9",
    description="High-performance TTS with optional vLLM/CUDA and MLX backends",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Marco Lironi",
    author_email="marcolironi@astramind.ai",
    url="https://github.com/groxaxo/Auralis-Enhanced",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    entry_points={
        "console_scripts": [
            "auralis.openai=auralis.entrypoints.oai_server:main",
        ],
    },
    install_requires=[
        "aiofiles",
        "beautifulsoup4",
        "cachetools",
        "colorama",
        "cutlet",
        "EbookLib",
        "einops",
        "ffmpeg",
        "fsspec",
        "hangul_romanize",
        "huggingface_hub",
        "ipython",
        "langid",
        "librosa",
        "networkx",
        "num2words",
        "numpy>=1.26",
        "opencc",
        "packaging",
        "pyloudnorm",
        "pypinyin",
        "safetensors",
        "setuptools",
        "sounddevice",
        "soundfile",
        "spacy==3.7.5",
        "tokenizers",
        "torch",
        "torchaudio",
        "transformers",
    ],
    extras_require={
        "cuda": [
            "vllm==0.6.4.post1; platform_system == 'Linux'",
            "nvidia-ml-py; platform_system == 'Linux'",
        ],
        "mlx": [
            "mlx-audio[tts]>=0.4.5,<0.5; platform_system == 'Darwin' and platform_machine == 'arm64'",
        ],
        "server": [
            "aiohttp",
            "fastapi>=0.95.0",
            "openai>=1.0.0",
            "python-multipart",
            "uvicorn[standard]>=0.22.0",
        ],
        "examples": [
            "gradio>=4.0.0",
            "requests",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.21",
        ],
        "docs": [
            "mkdocs>=1.4.0",
            "mkdocs-material>=9.0.0",
            "mkdocstrings>=0.20.0",
            "mkdocstrings-python>=1.0.0",
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
    ],
)
