import sys
import time
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import torch
from pynput import keyboard
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    MarianMTModel,
    MarianTokenizer,
)
from iflry_tts.utils import (
    add_silence_padding,
    audio_callback,
    run_pipeline,
    choose_lang_direction,
    load_marian_model,
    source_language,
    pick_whisper_model,
    choose_speaker,
)
from iflry_tts.branding import show_logo, show_splash
from outetts import HFModelConfig_v2, InterfaceHF, GenerationConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

SAMPLE_RATE = 16000
recording_in_progress = False
audio_q = queue.Queue()

GLOBAL_ASR_PROCESSOR = None
GLOBAL_ASR_MODEL = None
GLOBAL_TRANS_TOKENIZER = None
GLOBAL_TRANS_MODEL = None
GLOBAL_TRANS_DESC = None
GLOBAL_SPEAKER = None
GLOBAL_INTERFACE = None
GLOBAL_INPUT_STREAM = None
GLOBAL_WHISPER_MODEL_NAME = None


def main():
    show_logo()
    show_splash()

    # 1) Pick translation direction
    lang_key = choose_lang_direction()
    trans_tokenizer, trans_model, trans_desc = load_marian_model(lang_key)

    # 2) Decide source language and pick Whisper model
    src_lang = source_language(lang_key)
    whisper_model_name = pick_whisper_model(src_lang)
    print(f"Loading Whisper model: {whisper_model_name}")
    asr_processor = WhisperProcessor.from_pretrained(whisper_model_name)
    asr_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name).to(device)

    # 3) Setup OuteTTS
    model_cfg = HFModelConfig_v2(
        model_path="OuteAI/OuteTTS-0.3-1B",
        tokenizer_path="OuteAI/OuteTTS-0.3-1B",
    )
    interface = InterfaceHF(model_version="0.3", cfg=model_cfg)

    # 4) Choose OuteTTS speaker
    speaker = choose_speaker(interface)

    # Assign to global references
    global GLOBAL_TRANS_TOKENIZER, GLOBAL_TRANS_MODEL, GLOBAL_TRANS_DESC
    GLOBAL_TRANS_TOKENIZER = trans_tokenizer
    GLOBAL_TRANS_MODEL = trans_model
    GLOBAL_TRANS_DESC = trans_desc

    global GLOBAL_ASR_PROCESSOR, GLOBAL_ASR_MODEL, GLOBAL_SPEAKER
    GLOBAL_ASR_PROCESSOR = asr_processor
    GLOBAL_ASR_MODEL = asr_model
    GLOBAL_SPEAKER = speaker
    global GLOBAL_WHISPER_MODEL_NAME
    GLOBAL_WHISPER_MODEL_NAME = whisper_model_name

    # 5) Start the InputStream
    global GLOBAL_INPUT_STREAM
    GLOBAL_INPUT_STREAM = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
    )
    GLOBAL_INPUT_STREAM.start()

    print("\nConfiguration done! Press SPACE to talk, release to finalize, ESC to quit.\n")

    # 6) Key listener
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    # Cleanup
    GLOBAL_INPUT_STREAM.stop()
    GLOBAL_INPUT_STREAM.close()
    print("Exiting...")


if __name__ == "__main__":
    main()
