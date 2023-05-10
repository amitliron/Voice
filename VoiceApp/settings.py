import torch
import os
import whisper
from pyannote.audio import Pipeline

#
#   General
#
DEVICE             = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE        = 16000
SAVE_RESULTS_PATH  = f"{os.getcwd()}/TmpFiles"



#
#   whisper server
#
RAMBO_IP     = "10.53.140.33:86"
SERVER_IP    = RAMBO_IP
WHISPER_URL  = f'http://{SERVER_IP}/gradio_demo_live/'
LANGUAGES    = whisper.tokenizer.LANGUAGES

#
#   whisper thresholds
#
compression_ratio_threshold = 2.4
logprob_threshold           = -1.0
no_speech_threshold         = 0.95

#
#   local whisper
#
RUN_LOCAL_WHISPER = False
whisper_model     = None

#
#   offline diarization
#
sd_pipeline          = None #Pipeline.from_pretrained("pyannote/speaker-diarization")
run_online_pyyannote = True