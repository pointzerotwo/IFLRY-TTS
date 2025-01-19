from setuptools import setup, find_packages

setup(
    name="iflry-tts",
    version="0.1.0",
    description="IFLRY-branded live translation and TTS system.",
    author="IFLRY",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "soundfile",
        "numpy",
        "pynput",
        "sounddevice",
        "librosa",
        "art",
    ],
    extras_require={
        "cpu": [
            "torch>=1.10,<2.0",
            "torchaudio>=0.10,<2.0",
            "torchvision>=0.10,<2.0",
        ],
        "gpu": [
            "torch>=1.10,<2.0",
            "torchaudio>=0.10,<2.0",
            "torchvision>=0.10,<2.0",
            "nvidia-pyindex",
        ],
    },
    entry_points={
        "console_scripts": [
            "iflry-tts=iflry_tts.main:main",
        ],
    },
    python_requires=">=3.8,<3.12",  # Limit Python versions for compatibility
)
