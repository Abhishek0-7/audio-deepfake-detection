import torchaudio
import torch

def load_audio(file_path):
    waveform, sr = torchaudio.load(file_path)
    return waveform.squeeze(), sr

def preprocess_label(file_path):
    # Example: If filename contains 'bonafide' or 'spoof'
    if "bonafide" in file_path:
        return 0
    else:
        return 1
