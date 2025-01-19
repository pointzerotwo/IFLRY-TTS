from setuptools import setup, find_packages

setup(
    name="iflry-tts",
    version="0.1.0",
    description="IFLRY-branded live translation and TTS system.",
    author="IFLRY",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "soundfile",
        "numpy",
        "pynput",
        "sounddevice",
        "librosa",
        "art",
    ],
    entry_points={
        "console_scripts": [
            "iflry-tts=iflry_tts.main:main",
        ],
    },
)
