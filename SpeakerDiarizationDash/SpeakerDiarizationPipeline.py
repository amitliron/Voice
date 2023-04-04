import os

import librosa
import torch
from torch.utils.data  import Dataset, DataLoader
import torchaudio
import whisper
from pyannote.audio import Pipeline
import re
import soundfile         as sf

class SpeakerDiarizationPipeline():

    def __init__(self, device=None):
        self.sd_pipeline    = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                                    use_auth_token="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")

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._whisper_model = whisper.load_model("base", device=self.device)
        self._tmpWavFile = f"{os.getcwd()}/tmp.wav"
        self._tmpWhisperFile = f"{os.getcwd()}/whisper.wav"
        self._SAMPLE_RATE = 16000

    def HebrewNormalizer(self, hebrew_text):

        # --- step 2: replace signs
        hebrew_text = hebrew_text.replace('$', " דולר")
        hebrew_text = hebrew_text.replace('₪', " שח")
        hebrew_text = hebrew_text.replace('€', " יורו")
        # hebrew_text = hebrew_text.replace('.', " נקודה")
        hebrew_text = hebrew_text.replace('ת"א', "תל אביב")
        hebrew_text = hebrew_text.replace('ב"ש', "באר שבע")
        hebrew_text = hebrew_text.replace('ע"י', "על ידי")
        hebrew_text = hebrew_text.replace('אח"כ', "אחר כך")
        hebrew_text = hebrew_text.replace('\"', "")

        valid_tokens = "פ ם ן ו ט א ר ק ף ך ל ח י ע כ ג ד ש ץ ת צ מ נ ה ב ס ז 1 2 3 4 5 6 7 8 9 0"
        valid_tokens = set([x.lower() for x in valid_tokens])
        # The caret in the character class ([^) means match anything but
        invalid_chars_regex = f"[^\s{re.escape(''.join(set(valid_tokens)))}]"

        """ DO ADAPT FOR YOUR USE CASE. this function normalizes the target text. """
        hebrew_text = re.sub(invalid_chars_regex, " ", hebrew_text)
        hebrew_text = re.sub("\s+", " ", hebrew_text).strip()
        # --- return result
        return hebrew_text


    def GetWhisperText(self, file_name):
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(file_name)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self._whisper_model.device)

        # decode the audio
        options = whisper.DecodingOptions(beam_size=8, patience=2, fp16=False)  # , temperature=0.2
        result = whisper.decode(self._whisper_model, mel, options)
        if result.language == 'he':
            result = result.text
            result = self.HebrewNormalizer(result)
        else:
            result = "No Hebrew"

        return result

    def SeperateSpeakers(self, pyannote_diarization_res,  speech_raw_data):
        for turn, _, speaker in pyannote_diarization_res.itertracks(yield_label=True):
            start_sec = turn.start
            end_sec = turn.end
            speech_len = round(end_sec-start_sec, 2)
            speaker = speaker

            start_sample = int(start_sec * self._SAMPLE_RATE)
            end_sample = int(end_sec * self._SAMPLE_RATE)

            if speech_len >= 0.5:
                self.SaveAsWavFile(speech_raw_data[start_sample : end_sample], self._tmpWhisperFile)
                whisper_text = self.GetWhisperText(self._tmpWhisperFile)
                print(f"Speaker: {speaker}\n\tTimes: {round(start_sec, 2)} - {round(end_sec, 2)}\n\tLength: {speech_len} sec\n\tText: {whisper_text}")

    def SaveAsWavFile(self, speech, fileName):
        sf.write(fileName, speech, self._SAMPLE_RATE)


    def RunSpeakerDiarization(self, speech_raw_data):
        self.SaveAsWavFile(speech_raw_data, self._tmpWavFile)
        diarization = self.sd_pipeline(self._tmpWavFile)
        self.SeperateSpeakers(diarization,  speech_raw_data)





if __name__ == "__main__":
    speakerDiarizationPipeline = SpeakerDiarizationPipeline()
    test_file = "/home/amitli/Datasets/speaker-diarization/Youtube/Yom_Kippur/Yom_Kippur_1.wav"

    speech, sr = librosa.load(test_file, sr=16000)
    speakerDiarizationPipeline.RunSpeakerDiarization(speech)
