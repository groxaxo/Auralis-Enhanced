from pathlib import Path

from setuptools import setup, find_packages
import sys
import platform

def check_platform():
    if sys.platform != 'linux' and sys.platform != 'linux2':
        raise RuntimeError(
            f"""
            Following vllm requirements are not met:
            Current platform: {platform.system()} but only linux platforms are supported.
            """
        )

check_platform()
setup(
    name='auralis',
    version='0.2.8.post2',
    description='This is a faster implementation for TTS models, to be used in highly async environment',
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    author='Marco Lironi',
    author_email='marcolironi@astramind.ai',
    url='https://github.com/astramind.ai/auralis',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    entry_points={
            'console_scripts': [
                'auralis.openai=auralis.entrypoints.oai_server:main',
            ],
        },
    # Keep in sync with requirements.txt. `torch` / `torchaudio` /
    # `torchvision` are pulled in transitively (and pinned exactly) by
    # vllm 0.9.2, so we do not list them here; users on Blackwell should
    # install the cu128 wheels manually before `pip install -e .` per
    # INSTALL.md, otherwise pip resolves a CPU-only wheel from PyPI.
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
        "numpy>=2.0",
        "nvidia-ml-py",
        "opencc",
        "packaging",
        "pyloudnorm",
        "pypinyin",
        "pytest",
        "safetensors",
        "setuptools",
        "sounddevice",
        "soundfile",
        "spacy>=3.8",
        "tokenizers",
        "transformers>=4.51,<4.55",
        "vllm>=0.9.2",
    ],
    extras_require={
        'docs': [
            'mkdocs>=1.4.0',
            'mkdocs-material>=9.0.0',
            'mkdocstrings>=0.20.0',
            'mkdocstrings-python>=1.0.0',
        ],
    },
    python_requires='>=3.10',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
