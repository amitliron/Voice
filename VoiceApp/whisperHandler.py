import whisper
import numpy as np
import datetime
import requests

def Get_Whisper_From_Server(audio):
    # audio_data = {'wav': [str(i) for i in audio.tolist()]}
    # live_url   = 'http://10.53.140.33:86/gradio_demo_live/'
    # res        = requests.get(live_url, json=audio_data)
    # return res.json()[0]
    return "TODO", "heb", 0.5

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

    return text, lang, no_speech_prob