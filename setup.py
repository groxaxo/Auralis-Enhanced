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
    version='0.2.3',
    description='This is a faster implementation for TTS models, to be used in highly async environment',
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    author='Marco Lironi',
    author_email='marcolironi@astramind.ai',
    url='https://github.com/astramind.ai/auralis',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        "aiofiles==24.1.0",
        "beautifulsoup4==4.12.3",
        "cachetools==5.3.3",
        "colorama==0.4.6",
        "cutlet==0.4.0",
        "EbookLib==0.18",
        "einops==0.8.0",
        "fastapi==0.115.5",
        "ffmpeg==1.4",
        "fsspec==2024.10.0",
        "hangul_romanize==0.1.0",
        "huggingface_hub==0.26.1",
        "ipython==8.12.3",
        "networkx==3.4.2",
        "num2words==0.5.13",
        "opencc==1.1.9",
        "packaging==24.2",
        "pypinyin==0.53.0",
        "pytest==8.3.3",
        "safetensors==0.4.5",
        "setuptools==75.1.0",
        "vllm==0.6.4.post1",
        "sounddevice==0.5.1",
        "soundfile==0.12.1",
        "spacy==3.7.5",
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "triton==3.1.0",
        "langid",
        "librosa",
        "numpy",
        "pyloudnorm",
        "tokenizers",
        "transformers"
    ],
    python_requires='>=3.10',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
