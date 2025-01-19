# IFLRY TTS System

IFLRY live translation and text-to-speech system powered by Whisper, MarianMT, and OuteTTS.

## Installation

1. Clone the repository.
2. Create and activate a conda environment: 
   ```conda create -n iflry_tts python=3.10 -y
   conda activate iflry_tts
3. Install PyTorch:
   ```# CPU-Only
   conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

   # GPU (CUDA 11.8)
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
4. Install the package:
   ```bash
   pip install .
