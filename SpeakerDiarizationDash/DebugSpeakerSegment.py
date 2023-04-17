from pyannote.audio              import Model
from pyannote.audio              import Inference
from huggingface_hub.hf_api      import HfFolder
from pydub.utils                 import mediainfo
from pyannote.audio.utils.signal import Binarize
from pyannote.audio.utils.signal import Peak

import numpy                as     np
import soundfile            as sf
import plotly.express       as px

import librosa

BATCH_AXIS   = 0
TIME_AXIS    = 1
SPEAKER_AXIS = 2

def myBinary(prob, onset):
    for i in range(len(prob)):
        if prob[i] < onset:
            prob[i] = None
        else:
            prob[i] = 0.5
    return prob

if __name__ == "__main__":
    print("Start")
    SAMPLE_WAV = "/home/amitli/Datasets/test_speaker_segmentation.wav"
    MY_TOKEN = "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"
    HfFolder.save_token(MY_TOKEN)

    segment_model = Model.from_pretrained("pyannote/segmentation")
    inference = Inference(segment_model, duration=5.0, step=2.5)

    to_vad = lambda o: np.max(o, axis=SPEAKER_AXIS, keepdims=True)
    vad = Inference("pyannote/segmentation", pre_aggregation_hook=to_vad)
    vad_prob = vad(SAMPLE_WAV)
    binarize = Binarize(onset=0.5)
    speech = binarize(vad_prob)
    my_vad = myBinary(vad_prob.data.reshape(-1), 0.5)


    # ----------------------------------------------------------------------------------
    #
    # change detection
    #
    # ----------------------------------------------------------------------------------

    to_scd = lambda probability: np.max(np.abs(np.diff(probability, n=1, axis=TIME_AXIS)),
                                        axis=SPEAKER_AXIS, keepdims=True)
    scd = Inference("pyannote/segmentation", pre_aggregation_hook=to_scd)
    scd_prob = scd(SAMPLE_WAV)

    peak = Peak(alpha=0.05)
    scd_peak = peak(scd_prob).crop(speech.get_timeline())

    print("End")