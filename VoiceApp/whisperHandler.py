import whisper
import numpy as np
import datetime
import requests
import torch
from scipy.io.wavfile import write
import librosa


def Get_Whisper_From_Server(audio):
    audio_data = {'wav': [str(i) for i in audio.tolist()], 'languages': [None]}
    live_url   = 'http://10.53.140.33:86/gradio_demo_live/'
    res        = requests.get(live_url, json=audio_data)
    if type(res) == requests.Response:
        if 200 != res.status_code:
            print(f"Error, Resonse: {res.status_code}")
            #write("/home/amitli/Downloads/bla3.wav", 16000, audio)
            return "Error", "heb", 0.5

    res        =  res.json()[0]
    language      = res['language']
    text          = res["text"]
    no_speech_prb = res["no_speech_prob"]
    return text, language, no_speech_prb

def Get_Whisper_Text(whisper_model, audio):

    # load audio and pad/trim it to fit 30 seconds
    #print(f"Start Whisper: {datetime.datetime.now()}")
    audio = audio.astype(np.float32)
    audio = whisper.pad_or_trim(audio)
    mel   = whisper.log_mel_spectrogram(audio).to(whisper_model.device)

    # decode the audio
    options         = whisper.DecodingOptions(beam_size=5, fp16=False, language="en")
    result          = whisper.decode(whisper_model, mel, options)
    lang            = result.language
    text            = result.text
    no_speech_prob  = result.no_speech_prob

    #return text, lang, no_speech_prob
    return f"test_{len(audio)}", "en", 0.7


if __name__ == "__main__":

    y, sr = librosa.load("/home/amitli/Downloads/bla33.wav", sr=16000)
    text, language, no_speech_prb = Get_Whisper_From_Server(y)
    print(text)





