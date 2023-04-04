from diart import OnlineSpeakerDiarization
from diart.sources import MicrophoneAudioSource
from diart.inference import RealTimeInference
from diart.sinks import RTTMWriter
import os

from WhisperObserver import WhisperObserver


if __name__ == "__main__":

    pipeline = OnlineSpeakerDiarization()
    mic = MicrophoneAudioSource(pipeline.config.sample_rate)
    inference = RealTimeInference(pipeline, mic, do_plot=True)
    inference.attach_observers(RTTMWriter(mic.uri, f"{os.getcwd()}/file.rttm"))
    inference.attach_observers(WhisperObserver())
    print("Start...\n")
    prediction = inference()
    print("end")
