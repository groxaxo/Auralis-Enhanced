from setuptools import setup, find_packages

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
