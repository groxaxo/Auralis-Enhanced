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
    version='0.1.0',
    description='This is a faster implementation for TTS models, to be used in highly async environment',
    author='Marco Lironi',
    author_email='marcolironi@astramind.ai',
    url='https://github.com/astramind.ai/auralis',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        'asyncio',
        'torch',
        'torchaudio',
        'librosa',
        'vllm',
        'safetensors',
    ],
    python_requires='>=3.10',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: Linux',
    ],
)
