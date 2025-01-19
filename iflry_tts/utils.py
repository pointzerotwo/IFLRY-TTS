import numpy as np
import sounddevice as sd
import torch
from transformers import MarianTokenizer, MarianMTModel, WhisperProcessor, WhisperForConditionalGeneration

recording_in_progress = False
audio_q = None  # This should be initialized in the main script


def add_silence_padding(wave, sr, pad_sec=0.25):
    """
    Add silence padding at the beginning and end of the waveform.

    Args:
        wave (np.ndarray): The audio waveform.
        sr (int): The sampling rate.
        pad_sec (float): Duration of silence padding in seconds.

    Returns:
        np.ndarray: Padded waveform.
    """
    pad_samples = int(sr * pad_sec)
    silence = np.zeros(pad_samples, dtype=wave.dtype)
    return np.concatenate([silence, wave, silence], axis=0)


def audio_callback(indata, frames, time_info, status):
    """
    Callback for audio input stream. Adds recorded audio to a global queue.

    Args:
        indata (np.ndarray): Recorded audio data.
        frames (int): Number of frames.
        time_info (dict): Timing information.
        status (CallbackFlags): Status of the callback.
    """
    if recording_in_progress:
        audio_q.put(indata.copy())


def run_pipeline(final_array, asr_processor, asr_model, trans_tokenizer, trans_model, trans_desc, speaker, whisper_model_name):
    """
    Process the audio input through ASR, translation, and TTS.

    Args:
        final_array (np.ndarray): Final recorded audio data.
        asr_processor (WhisperProcessor): Whisper processor.
        asr_model (WhisperForConditionalGeneration): Whisper ASR model.
        trans_tokenizer (MarianTokenizer): Translation tokenizer.
        trans_model (MarianMTModel): Translation model.
        trans_desc (str): Translation description.
        speaker (object): OuteTTS speaker object.
        whisper_model_name (str): Name of the Whisper model.
    """
    # Check minimum length of input
    if len(final_array) < 100:
        print("[Info] Not enough audio. Possibly no speech.")
        return

    # ASR: Speech-to-Text
    print(f"Running Whisper ASR on final audio ({whisper_model_name})...")
    input_features = asr_processor(
        final_array, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    with torch.no_grad():
        pred_ids = asr_model.generate(input_features)
    transcription = asr_processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
    print(f"[Whisper recognized]: '{transcription}'")

    if not transcription:
        print("[Warn] Transcription empty.")
        return

    # Translation: Text-to-Text
    print(f"Translating using {trans_desc}...")
    translate_inputs = trans_tokenizer(
        transcription, return_tensors="pt", padding=True, truncation=True
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    with torch.no_grad():
        translated_tokens = trans_model.generate(
            translate_inputs["input_ids"], attention_mask=translate_inputs["attention_mask"]
        )
    translated_text = trans_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    print(f"[Translation result]: '{translated_text}'")

    # TTS: Text-to-Speech
    print("Synthesizing with OuteTTS...")
    from outetts import GenerationConfig

    gen_cfg = GenerationConfig(
        text=translated_text,
        temperature=0.4,
        repetition_penalty=1.1,
        max_length=4096,
        speaker=speaker,
    )
    out = speaker.interface.generate(config=gen_cfg)

    out_filename = "outetts_temp.wav"
    out.save(out_filename)
    print(f"OuteTTS saved TTS to {out_filename}")

    # Add silence padding
    wave, sr_out = sf.read(out_filename)
    wave_padded = add_silence_padding(wave, sr_out, pad_sec=0.25)

    print("Playing TTS with silence padding...\n")
    sd.play(wave_padded, samplerate=sr_out)
    sd.wait()


def choose_lang_direction():
    """
    Display menu for selecting a translation direction.

    Returns:
        str: Selected translation direction (e.g., "EN->DE").
    """
    lang_map = {
        "EN->DE": "English -> German",
        "DE->EN": "German -> English",
        "EN->FR": "English -> French",
        "FR->EN": "French -> English",
        "DE->FR": "German -> French",
        "FR->DE": "French -> German",
    }

    print("\nChoose a translation direction:")
    for i, (key, desc) in enumerate(lang_map.items(), start=1):
        print(f"  {i}) {key} ({desc})")

    while True:
        choice = input("Enter number: ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(lang_map):
                return list(lang_map.keys())[idx - 1]
        print("Invalid choice, try again.")


def load_marian_model(lang_key):
    """
    Load MarianMT model and tokenizer for the specified language direction.

    Args:
        lang_key (str): Language direction (e.g., "EN->DE").

    Returns:
        MarianTokenizer, MarianMTModel, str: Tokenizer, model, and description.
    """
    lang_map = {
        "EN->DE": ("Helsinki-NLP/opus-mt-en-de", "English -> German"),
        "DE->EN": ("Helsinki-NLP/opus-mt-de-en", "German -> English"),
        "EN->FR": ("Helsinki-NLP/opus-mt-en-fr", "English -> French"),
        "FR->EN": ("Helsinki-NLP/opus-mt-fr-en", "French -> English"),
        "DE->FR": ("Helsinki-NLP/opus-mt-de-fr", "German -> French"),
        "FR->DE": ("Helsinki-NLP/opus-mt-fr-de", "French -> German"),
    }
    model_name, desc = lang_map[lang_key]
    print(f"Loading Marian model: {model_name} for {desc}...")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Fix padding token warnings
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model, desc


def source_language(lang_key):
    """
    Extract the source language from a language key.

    Args:
        lang_key (str): Language key (e.g., "EN->DE").

    Returns:
        str: Source language (e.g., "EN").
    """
    return lang_key.split("->")[0]


def pick_whisper_model(source_lang):
    """
    Select the Whisper model for the given source language.

    Args:
        source_lang (str): Source language (e.g., "EN").

    Returns:
        str: Whisper model name.
    """
    if source_lang == "EN":
        return "openai/whisper-small.en"
    else:
        return "openai/whisper-small"


def choose_speaker(interface):
    """
    Prompt user to select an OuteTTS speaker.

    Args:
        interface (InterfaceHF): OuteTTS interface.

    Returns:
        object: Selected speaker.
    """
    interface.print_default_speakers()
    print("\nEnter one of the default speaker names exactly as above:")
    name = input("Speaker name: ").strip()
    speaker = interface.load_default_speaker(name=name)
    print("Speaker loaded:", speaker)
    return speaker
