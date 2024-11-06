from setuptools import setup, find_packages

setup(
    name='fasterTTS',
    version='0.1.0',
    description='This is a faster implementation for TTS models, to be used in highly async environment',
    author='Marco Lironi',
    author_email='marcolironi@astramind.ai',
    url='https://github.com/astramind.ai/fasterTTS',
    packages=find_packages(exclude=['tests', 'docs']),
    install_requires=[
        'asyncio',
        'torch>=1.8.0',
        'torchaudio',
        'librosa',
        'vllm',
        'safetensors',
    ],
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: Linux',
    ],
)
