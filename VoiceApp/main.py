import os

import pandas               as pd
import plotly.graph_objects as go

import utilities
import whisperHandler
DEBUG = True

#if DEBUG is True:
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gradio as gr
import numpy as np
import time
import whisper
import torch
import librosa
import pickle
import queue

from fastapi                           import FastAPI
from tqdm                              import tqdm
from huggingface_hub.hf_api            import HfFolder
from pyannote.audio                    import Pipeline
from pyannote.core                     import notebook
from datetime                          import datetime
from recording_util                    import RecordingUtil


import matplotlib.pyplot               as plt
import plotly.express                  as px
import datetime                        as dt




class CExtractVoiceProperties:

    def __init__(self):
        self.file_path     = None
        self.DEVICE        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Device = {self.DEVICE}")
        if DEBUG is False:
            self.whisper_model = whisper.load_model("large", device=self.DEVICE)
            self.sd_pipeline   = Pipeline.from_pretrained("pyannote/speaker-diarization")
        else:
            self.whisper_model = whisper.load_model("tiny", device=self.DEVICE)
            self.sd_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

        self.vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                        model='silero_vad',
                                                        force_reload=False)


        (self.vad_get_speech_timestamps,
         self.vad_save_audio,
         self.vad_read_audio,
         self.VADIterator,
         self.vad_collect_chunks) = vad_utils

        self.vad_speech_model, vad_utils2 = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                   model='silero_vad',
                                                   force_reload=False)


        (self.speech_get_speech_timestamps, _,
         _,
         _,
         _) = vad_utils2


        self.vad_iterator = self.VADIterator(self.vad_model)
        self.MICROPHONE_SAMPLE_RATE       = None
        self.VAD_JOB_RATE                 = 0.5

        self.settings_record_wav = False
        self.settings_decoding_lang = []
        self.settings_use_prompt = False

        self.recordingUtil = RecordingUtil()
        self.stream_iter = 0
        self.diarization_text_result = ""
        self.run_online = True
        self.stream_start_time = 0
        self.speech_queue = queue.Queue()
        self.processed_queue = queue.Queue()
        #self.debug_queue = queue.Queue()
        self.vad_queue = queue.Queue()
        self.previous_speech = None
        self.last_30_sec_vad = None
        self.total_num_of_vad_elm = 0
        self.stream_results = ""
        self.last_speech_silence = False
        self.last_lang = ""
        self.last_text = []
        self.last_vad_plot = None
        self.debug_param   = 0
        if DEBUG is False:
            self.init()

    def clear(self):
        self.total_num_of_vad_elm = 0
        self.MICROPHONE_SAMPLE_RATE = None
        self.speech_queue = queue.Queue()
        self.vad_queue = queue.Queue()
        self.vad_iterator.reset_states()
        self.processed_queue = queue.Queue()
        #self.debug_queue = queue.Queue()
        self.previous_speech = None
        self.last_30_sec_vad = None
        self.last_speech_silence = False
        self.file_path = None
        self.diarization_text_result = ""
        self.stream_iter = 0
        self.stream_start_time = 0
        self.stream_results = ""
        self.last_lang = ""
        self.last_text = []
        self.last_vad_plot = None
        self.debug_param = 0


    def init(self):

        INIT_WAV = f"{os.getcwd()}/init.wav"

        # if DEBUG is False:
        #     print(f"Start init pyyanote ({datetime.now()})")
        #     diarization_res = self.sd_pipeline(INIT_WAV)
        #     print(f"Finish init pyyanote ({datetime.now()})")

        # print(f"Start init Whisper ({datetime.now()})")
        # res = self.Get_Whisper_Text(INIT_WAV)
        # print(f"Finish init Whisper ({datetime.now()})")

    # def Get_Whisper_Text(self,  file_name):
    #
    #     audio = whisper.load_audio(file_name)
    #     audio = whisper.pad_or_trim(audio)
    #     mel = whisper.log_mel_spectrogram(audio).to(self.whisper_model.device)
    #
    #     # decode the audio
    #     options = whisper.DecodingOptions(beam_size=5, fp16=False)
    #     result = whisper.decode(self.whisper_model, mel, options)
    #     result = result.text
    #
    #     return result


SAMPLE_RATE                  = 16000
extractVoiceProperties       = CExtractVoiceProperties()
SAVE_RESULTS_PATH            = f"{os.getcwd()}/TmpFiles"
NO_SPEECH_PROBABILITY        = 0.6



def reformat_freq(sr, y):
    '''
    Took it from: https://gradio.app/real-time-speech-recognition/
    :param sr:
    :param y:
    :return:
    '''
    if sr not in (
        48000,
        16000,
    ):  #  only supports 16k, (we convert 48k -> 16k)
        raise ValueError("Unsupported rate", sr)
    if sr == 48000:
        y = (
            ((y / max(np.max(y), 1)) * 32767)
            .reshape((-1, 3))
            .mean(axis=1)
            .astype("int16")
        )
        sr = 16000
    return sr, y

def schedule_preprocess_speech_job():

    #
    #   Step 1: collect last (5) seconds speech
    #
    q_len = extractVoiceProperties.speech_queue.qsize()
    if q_len == 0:
        return
    speech = np.array([])
    for i in range(q_len):
        speech = np.concatenate((speech, extractVoiceProperties.speech_queue.get()))
    speech = np.int16(speech / np.max(np.abs(speech)) * 32767)
    new_sample_rate , speech = reformat_freq(extractVoiceProperties.MICROPHONE_SAMPLE_RATE, speech)

    #
    #   Step 2: check if we have older speech which we didn't finished to preocess
    #
    if extractVoiceProperties.previous_speech is not None:
        speech = np.concatenate((extractVoiceProperties.previous_speech, speech))

    #
    #   Step 3: run vad
    #
    speech                 = torch.from_numpy(speech)
    speech_timestamps      = extractVoiceProperties.speech_get_speech_timestamps(speech.float(),
                                                                              extractVoiceProperties.vad_speech_model,
                                                                              sampling_rate=16000)

    #
    #   Step 4: check if we have speech
    #
    if len(speech_timestamps) == 0:
        return

    #
    #   Step 5: handle sub speech (no at the end edge)
    #
    for val in speech_timestamps:
        start_sample = max(val['start']-1600,0)
        end_sample   = min(val['end'] + 1600, len(speech))
        if end_sample >= len(speech):
            break
        tmp_speech = speech[start_sample:end_sample].numpy()
        extractVoiceProperties.processed_queue.put(tmp_speech)
        print(
            f"Push speech to whisper Q, audio length: {round(len(tmp_speech) / 16000, 2)} seconds, |Q| = {extractVoiceProperties.processed_queue.qsize()}")


    #
    #   Step 6: save last unprcoess voice (when VAD end at the edge)
    #
    end_with_threshold = min(val['end'] + 1600, len(speech))
    if end_with_threshold>= len(speech):
        extractVoiceProperties.previous_speech = speech[speech_timestamps[-1]["start"] :].numpy()






def schedule_vad_job():

    global extractVoiceProperties

    #
    #   if no new mic samples
    #
    vad_q_len = extractVoiceProperties.vad_queue.qsize()
    if vad_q_len == 0:
        return extractVoiceProperties.last_vad_plot

    #
    #   start first time vad with at least 5 seconds
    #
    # diff_time = time.time() - extractVoiceProperties.stream_start_time
    # if diff_time < 5:
    #     return extractVoiceProperties.last_vad_plot

    #
    #   collect samples (probably ~10 items for first time, and ~2 items for the others)
    #
    speech = np.array([])
    for i in range(vad_q_len):
        speech                                      = np.concatenate((speech, extractVoiceProperties.vad_queue.get()))
        extractVoiceProperties.total_num_of_vad_elm = extractVoiceProperties.total_num_of_vad_elm + 1
    speech = np.int16(speech / np.max(np.abs(speech)) * 32767)

    #
    # run VAD (for each half second)
    #
    arr_vad                          = []
    VAD_WINDOW                       = int(extractVoiceProperties.MICROPHONE_SAMPLE_RATE * extractVoiceProperties.VAD_JOB_RATE)

    for i in range(0, len(speech), VAD_WINDOW):
        chunk = speech[i: i + VAD_WINDOW]
        chunk = torch.from_numpy(chunk)
        chunk = chunk.float()
        if len(chunk) < VAD_WINDOW:
            break
        speech_dict = extractVoiceProperties.vad_iterator(chunk, return_seconds=True)
        if speech_dict:
            None
        speech_prob = extractVoiceProperties.vad_model(chunk, 16000).item()
        arr_vad.append(speech_prob)

    #
    # add results to last results (and save only last 30 seconds -> last 60 VAD probabilties
    #
    if extractVoiceProperties.last_30_sec_vad is None:
        extractVoiceProperties.last_30_sec_vad = arr_vad
    else:
        extractVoiceProperties.last_30_sec_vad   = extractVoiceProperties.last_30_sec_vad + arr_vad
        NUMBER_OF_VAD_PROBABILTIES_IN_30_SECONDS = int(30/extractVoiceProperties.VAD_JOB_RATE)
        if len(extractVoiceProperties.last_30_sec_vad) > NUMBER_OF_VAD_PROBABILTIES_IN_30_SECONDS:
            start_offset = len(extractVoiceProperties.last_30_sec_vad) - NUMBER_OF_VAD_PROBABILTIES_IN_30_SECONDS
            extractVoiceProperties.last_30_sec_vad = extractVoiceProperties.last_30_sec_vad[start_offset:]

    #
    #   create plot
    #
    if len(extractVoiceProperties.last_30_sec_vad) < 2:
        return extractVoiceProperties.last_vad_plot

    end_time   = extractVoiceProperties.total_num_of_vad_elm
    start_time = max(end_time - len(extractVoiceProperties.last_30_sec_vad), 0)
    x_time = np.arange(start=start_time, stop=end_time, step=1)
    x_time = x_time / 2

    vad_speech = (np.array(extractVoiceProperties.last_30_sec_vad) > 0.5)
    vad_speech = vad_speech.astype(int)

    df         = pd.DataFrame()
    df['time'] = x_time
    df['vad']  = extractVoiceProperties.last_30_sec_vad
    df['speech'] = vad_speech

    # arr_time   = df["time"].values
    # vad_values = df["vad"].values
    # condition_speech = (vad_values < 0.5).astype(int)
    # condition_other = (vad_values >= 0.5).astype(int)
    # arr_speech  = vad_values * condition_speech
    # arr_speech[arr_speech == 0] = None
    # arr_no_speech = vad_values * condition_other
    # arr_no_speech[arr_no_speech == 0] = None
    #
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=arr_time, y=arr_speech))
    # fig.add_trace(go.Scatter(x=arr_time, y=arr_no_speech))

    fig        = px.line(df, x = "time", y="vad", title='silero-vad')
    extractVoiceProperties.last_vad_plot = fig


    #
    #
    #
    # if extractVoiceProperties.total_num_of_vad_elm > 60:
    #     speech = np.array([])
    #     l = extractVoiceProperties.debug_queue.qsize()
    #     for i in range(l):
    #         speech = np.concatenate((speech, extractVoiceProperties.debug_queue.get()))
    #     speech = np.int16(speech / np.max(np.abs(speech)) * 32767)
    #     full_fill_name = f"/home/amitli/Downloads/amit.wav"
    #     from scipy.io.wavfile import write
    #     write(full_fill_name, extractVoiceProperties.MICROPHONE_SAMPLE_RATE, speech)
    #     print("ok")


    #
    #   return vad (last 30 seconds) figure
    #
    return fig






def convert_text_to_html(full_html, current_text, current_lang):
    '''
        style text results in html format
    '''

    if current_lang == "he" or current_lang == "ar":
        current_line = f"<p style='text-align:right;'> {current_text} </p>" + "\n"
    else:
        current_line = f"<p style='text-align:left;'> {current_text} </p>"   + "\n"

    full_html.append(current_line)
    return full_html


def schedule_whisper_job():
    global extractVoiceProperties

    #
    #   Step 1: get current Q len (number of speech to decode with whisper)
    #
    q_len = extractVoiceProperties.processed_queue.qsize()
    if q_len == 0:
        return extractVoiceProperties.last_lang, extractVoiceProperties.last_text

    #
    #   Step 2: for each speech -> decode with whisper
    #
    for i in range(q_len):
        speech = extractVoiceProperties.processed_queue.get()

        print(f"Send Rqeust to whisper, audio length: {round(len(speech)/16000, 2)} seconds, [|Q| = {extractVoiceProperties.processed_queue.qsize()}")
        text, lang, no_speech_prob       = whisperHandler.Get_Whisper_Text(extractVoiceProperties.whisper_model, speech)
        extractVoiceProperties.last_lang = lang
        #extractVoiceProperties.last_text = extractVoiceProperties.last_text + "\n" + text
        extractVoiceProperties.last_text = convert_text_to_html(extractVoiceProperties.last_text, text, lang)

        print(f"Got Results from Whisper:\n\tText: {text}\n\tLanguage: {lang}\n\tno_speech_prob = {no_speech_prob}")

        if extractVoiceProperties.settings_record_wav is True:
            extractVoiceProperties.recordingUtil.record_wav(speech, sample_rate=16000)

    #
    #   Step 3: return results
    #
    return  extractVoiceProperties.last_lang , extractVoiceProperties.last_text




def show_saved_wav_file(wav_file_path):
    global extractVoiceProperties
    extractVoiceProperties.file_path = wav_file_path
    return wav_file_path


def process_diarizartion_results(diarization, speech, whisper_model):
    l_speakers_samples = []
    l_text = []
    l_speaker = []
    language = None

    for turn, _, speaker in tqdm(diarization.itertracks(yield_label=True)):
        start_time = turn.start
        end_time = turn.end
        duration = end_time - start_time
        if duration < 0.1:
            continue

        start_sample                   = int(start_time * SAMPLE_RATE)
        end_sample                     = int(end_time * SAMPLE_RATE)
        speaker_samples                = speech[start_sample:end_sample]
        text, language, no_speech_prob = whisperHandler.Get_Whisper_Text(whisper_model, speaker_samples)
        l_speakers_samples.append(speaker_samples)
        l_speaker.append(speaker)
        l_text.append(text)

    return l_speakers_samples, l_speaker, l_text, language


def prepare_text(l_text, l_speaker, language):
    '''

    :param l_text:         list of whisper text
    :param l_speaker:      list of speakers
    :param language:       language (he/en/...)
    :return:               HTML page with language alignment.
                           each speaker with different color
    '''

    if language == "he":
        align = "right"
    else:
        align = "left"

    text_results = ""
    speaker_dict = {}
    colors = ["red",  "blue", "green"]
    for i, sp in enumerate(set(l_speaker)):
        speaker_dict[sp] = colors[i]

    for i in range(len(l_speaker)):
        current_text = f"<p style='color:{speaker_dict[l_speaker[i]]}; text-align:{align};'> {l_speaker[i]} {l_text[i]} </p>" + "\n"
        text_results = text_results + current_text
    return text_results


def save_results_to_file(html_res, fig_res):
    with open(f"{SAVE_RESULTS_PATH}/html_res.txt", 'w') as file:
        file.write(html_res)
    fig_res.savefig(f"{SAVE_RESULTS_PATH}/fig_res.png")


def myBinary(prob, onset):
    '''
    Note: I implement this method and didn't used  pyannote Binarize, for simple plot used
    :param prob:        pyannote diarizartion results
    :param onset:       threshold
    :return:
    '''
    for i in range(len(prob)):
        if prob[i] < onset:
            prob[i] = None
        else:
            prob[i] = 0.5
    return prob

def get_vad_plot_from_diarization(wav_file):


    return None

def handle_wav_file(audioRecord, audioUpload):
    '''
    :param input_file:  wav file (Sample rate doesn't matter)
    :return:            speaker diarization plot
                                  +
                        whisper results for each speaker
    '''

    # audioRecord, audioUpload
    if (audioRecord is None) and (audioUpload is None):
        res = "<p style='color:red; text-align:left;'> Input is missing </p>"
        return res

    if (audioRecord is not None) and (audioUpload is not None):
        res = "<p style='color:red; text-align:left;'> Two inputs are selected, choose one of them </p>"
        return res

    if audioRecord is not None:
        input_file = audioRecord
    else:
        input_file = audioUpload

    global extractVoiceProperties

    if extractVoiceProperties.run_online is False:
        with open(f"{SAVE_RESULTS_PATH}/html_res.txt", 'r') as file:
            html_whisper_text = file.read()

        diarization_figure, ax = plt.subplots()
        import matplotlib.image as mpimg
        img = mpimg.imread(f"{SAVE_RESULTS_PATH}/fig_res.png")
        ax.imshow(img)
        return html_whisper_text

    if (input_file is None):
        print("Missing WAV file")
        return None

    file_duration = utilities.get_wav_duration(input_file)
    sample_rate   = utilities.get_sample_rate(input_file)
    print(f"file_duration = {file_duration},  sample_rate = {sample_rate}")

    if sample_rate != 16000:
        print("Change sample rate to 16000")
        utilities.convert_to_16sr_file(input_file, input_file)

    print(f"Run Diarization pipeline ({datetime.now()})")
    diarization_res = extractVoiceProperties.sd_pipeline(input_file)
    print(f"Diarization pipeline Finished ({datetime.now()})")
    print(f"process diarzation results ({datetime.now()})")
    speech, sr = librosa.load(input_file, sr=SAMPLE_RATE)


    res = process_diarizartion_results(diarization_res, speech, extractVoiceProperties.whisper_model)
    print(f"process diarzation finished ({datetime.now()})")
    l_speakers_samples = res[0]
    l_speaker = res[1]
    l_text = res[2]
    language = res[3]

    html_whisper_text      = prepare_text(l_text, l_speaker, language)
    diarization_figure, ax = plt.subplots()
    res                    = notebook.plot_annotation(diarization_res, ax=ax, time=True, legend=True)
    save_results_to_file(html_whisper_text, diarization_figure)
    vad_plot = get_vad_plot_from_diarization(diarization_res)


    return html_whisper_text

def handle_streaming(audio):

    global extractVoiceProperties

    rate  = audio[0]
    voice = audio[1]

    if extractVoiceProperties.MICROPHONE_SAMPLE_RATE is None:
        print(f"MICROPHONE_SAMPLE_RATE = {rate}")
        extractVoiceProperties.MICROPHONE_SAMPLE_RATE = rate

    if rate != extractVoiceProperties.MICROPHONE_SAMPLE_RATE:
        print(f"-sample rate changed: {extractVoiceProperties.MICROPHONE_SAMPLE_RATE} ->  {rate} - \n")
        extractVoiceProperties.MICROPHONE_SAMPLE_RATE = rate

    extractVoiceProperties.vad_queue.put(voice)
    extractVoiceProperties.speech_queue.put(voice)

    if extractVoiceProperties.stream_start_time == 0:
        print("First time start recording")
        extractVoiceProperties.stream_start_time = time.time()




def set_running_option(value):
    global extractVoiceProperties
    if value == "Run Diariation Model":
        extractVoiceProperties.run_online = True
    else:
        extractVoiceProperties.run_online = False

def change_settings(settings_record_wav, settings_decoding_lang, settings_use_prompt):
    print(f"Settings changed to: Reocrd Wav: {settings_record_wav}, Decoding Lang: {settings_decoding_lang}, Decoding Prompt:{settings_use_prompt}")
    global extractVoiceProperties
    extractVoiceProperties.settings_record_wav = settings_record_wav
    extractVoiceProperties.settings_use_prompt = settings_use_prompt
    extractVoiceProperties.settings_decoding_lang = []
    if settings_decoding_lang == "Hebrew":
        extractVoiceProperties.settings_decoding_lang = ["he"]
    elif settings_decoding_lang == "English":
        extractVoiceProperties.settings_decoding_lang = ["en"]

def create_new_gui():
    with gr.Blocks(theme=gr.themes.Glass()) as demo:


        with gr.Tab("Real Time"):
            stream_input       = gr.Audio(source="microphone")
            output_stream_lang = gr.Label(label = "Detect Lanugage: ")
            #output_stream_text = gr.Textbox(label = "Whisper Results:")
            output_stream_text = gr.outputs.HTML(label="Whisper Results:")
            output_stream_plt  = gr.Plot(labal = "Voice Activity Detection:")

        with gr.Tab("Offline"):
            with gr.Row():
                audioUpload = gr.Audio(source="upload", type="filepath")
                audioRecord = gr.Audio(source="microphone", type="filepath")

            audioProcessRecButton = gr.Button("Process")
            output_diarization_text = gr.outputs.HTML(label="")
            #output_diarization_img = gr.Plot(label = "Diarization")
            audioProcessRecButton.click(fn=handle_wav_file, inputs=[audioRecord, audioUpload],
                                        outputs=[output_diarization_text])

        with gr.Tab("Settings"):
            settings_record_wav = gr.Checkbox(label="Record WAV", info="Record WAV files for debug")
            settings_decoding_lang = gr.Dropdown(["None", "Hebrew", "English"], label="DecodingLanguage",
                                                 info="Run Whisper with language decoding")
            settings_use_prompt = gr.Checkbox(label="Use Whisper prompt", info="Run Whisper with prompt decoding")

            settings_record_wav.change(change_settings, inputs=[settings_record_wav,
                                                                settings_decoding_lang,
                                                                settings_use_prompt], outputs=[])

            settings_decoding_lang.change(change_settings, inputs=[settings_record_wav,
                                                                   settings_decoding_lang,
                                                                   settings_use_prompt], outputs=[])

            settings_use_prompt.change(change_settings, inputs=[settings_record_wav,
                                                                settings_decoding_lang,
                                                                settings_use_prompt], outputs=[])

        with gr.Tab("About"):
            gr.Label("Version 1")

        stream_input.stream(fn      = handle_streaming,
                            inputs  = [stream_input],
                            outputs = [])

        demo.load(schedule_vad_job, None, [output_stream_plt], every=extractVoiceProperties.VAD_JOB_RATE)
        demo.load(schedule_preprocess_speech_job, None, None, every=5)
        demo.load(schedule_whisper_job, None, [output_stream_lang, output_stream_text], every=2)

        #set_running_option(False)
    return demo



def debug_only(num, input_sr=None):

    print(f"\n {num}:")
    full_path = f"/home/amitli/Downloads/1/{num}.wav"
    if input_sr is None:
        speech, sr = librosa.load(full_path)
    else:
        speech, sr = librosa.load(full_path, sr=input_sr)
    text, lang, no_speech_prob = whisperHandler.Get_Whisper_Text(extractVoiceProperties.whisper_model, speech)
    print(f"sr = {sr}")
    print(f"no_speech_prob = {no_speech_prob}")
    print(f"lang = {lang}")
    print(f"text = {text}")


if __name__ == "__main__":

    utilities.save_huggingface_token()
    demo = create_new_gui()
    #demo.queue().launch(share=False, debug=False)

    #  openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 365 -nodes
    #  https://10.53.140.33:8432/

    demo.queue().launch(share=False,
                        debug=False,
                        server_name="0.0.0.0",
                        server_port=8432,
                        ssl_verify=False,
                        ssl_certfile="cert.pem",
                        ssl_keyfile="key.pem")
