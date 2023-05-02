import torch
import librosa

if __name__ == "__main__":
    input_file = "/home/amitli/Downloads/amit.wav"
    speech, sr = librosa.load(input_file, sr=16000)

    print(f"sample rate = {sr}, type: {type(speech)}")

    vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                               model='silero_vad',
                                               force_reload=False)

    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = vad_utils

    speech = torch.from_numpy(speech)

    speech_timestamps = get_speech_timestamps(speech, vad_model, sampling_rate=16000)
    for val in speech_timestamps:
        start_sec = round(val['start'] / 16000, 3)
        end_sec   = round(val['end'] /   16000, 3)
        print(f"Start: {start_sec} end: {end_sec}")
    print(f"File End At: {round(len(speech)/16000, 3)}")
    audio = collect_chunks(speech_timestamps, speech)

    from scipy.io.wavfile import write
    write("/home/amitli/Downloads/amit_vad.wav", 16000, audio.numpy())

    audio = collect_chunks([speech_timestamps[3]], speech)
    write("/home/amitli/Downloads/amit_vad_2.wav", 16000, audio.numpy())
