import whisper
import numpy as np




def Get_Whisper_Text(whisper_model, audio):

    # load audio and pad/trim it to fit 30 seconds
    audio = audio.astype(np.float32)
    audio = whisper.pad_or_trim(audio)
    print(f"Whisper model: {whisper_model.device}")
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)

    # decode the audio
    # options = whisper.DecodingOptions(language = 'he', beam_size=8, fp16 = False)
    #options = whisper.DecodingOptions(language='en', beam_size=5, fp16=False)
    options = whisper.DecodingOptions(beam_size=5, fp16=False)
    result = whisper.decode(whisper_model, mel, options)
    lang = result.language
    text = result.text

    return text, lang