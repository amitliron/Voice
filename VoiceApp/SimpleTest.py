import os
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import whisper
from huggingface_hub.hf_api            import HfFolder
from pyannote.audio                    import Pipeline
from datetime                          import datetime
from tqdm                              import tqdm

import torch


SAMPLE_RATE = 16000
INIT_WAV = f"{os.getcwd()}/init.wav"


def Get_Whisper_Text(whisper_model, file_name):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(file_name)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)

    # decode the audio
    options = whisper.DecodingOptions(beam_size=5, fp16=False)
    result = whisper.decode(whisper_model, mel, options)
    result = result.text

    return result

if __name__ == "__main__":
    print(f"Start test ({datetime.now()})")
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device = {DEVICE}")

    print(f"Save HF Token ({datetime.now()})")
    MY_TOKEN = "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"
    HfFolder.save_token(MY_TOKEN)

    print (f"Load Whisper model ({datetime.now()})")
    whisper_model = whisper.load_model("large", device=DEVICE)

    print(f"Init WHisper ({datetime.now()})")
    res = Get_Whisper_Text(whisper_model, INIT_WAV)
    print(f"FInsih Init WHisper ({datetime.now()})")

    print(f"Run Whisper ({datetime.now()})")
    res = Get_Whisper_Text(whisper_model, INIT_WAV)
    print(f"Whisper Finished ({datetime.now()})")

    print(f"Run Whisper ({datetime.now()})")
    res = Get_Whisper_Text(whisper_model, INIT_WAV)
    print(f"Whisper Finished ({datetime.now()})")
    print(res)

    print(f"Load pyyanote model ({datetime.now()})")
    sd_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

    print(f"Init PyAnnote ({datetime.now()})")
    diarization_res = sd_pipeline(INIT_WAV)
    print(f"FInsih Init PyAnnote ({datetime.now()})")

    input_file = "./TestFiles/shiri_and_amit.wav"
    print(f"Run Diarization pipeline ({datetime.now()})")
    diarization_res = sd_pipeline(input_file)
    print(f"Diarization pipeline Finished ({datetime.now()})")

    print(f"Run Diarization pipeline ({datetime.now()})")
    diarization_res = sd_pipeline(input_file)
    print(f"Diarization pipeline Finished ({datetime.now()})")



    print(f"Finish test ({datetime.now()})")
