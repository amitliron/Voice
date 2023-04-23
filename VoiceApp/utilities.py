from pydub.utils                       import mediainfo
from huggingface_hub.hf_api            import HfFolder

import soundfile                       as sf
import librosa

def get_sample_rate(file):
    info      = mediainfo(file)
    return int(info['sample_rate'])

def get_wav_duration(wav_file):
    f = sf.SoundFile(wav_file)
    return f.frames / f.samplerate


def convert_to_16sr_file(source_path, dest_path):
    speech, sr = librosa.load(source_path, sr=16000)
    sf.write(dest_path, speech, sr)
    return speech

def save_huggingface_token():
    print("Save hugging face token")
    MY_TOKEN = "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"
    HfFolder.save_token(MY_TOKEN)