import os

import pandas as pd

import utilities
import whisperHandler

#os.environ["CUDA_VISIBLE_DEVICES"] = ""


import gradio as gr
import numpy as np
import time
import whisper
import torch
import librosa
import pickle

from tqdm                              import tqdm
from huggingface_hub.hf_api            import HfFolder
from pyannote.audio                    import Pipeline
from pyannote.core                     import notebook

import matplotlib.pyplot               as plt
import plotly.express                  as px



class CExtractVoiceProperties:

    def __init__(self):
        self.file_path     = None
        self.DEVICE        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Device = {self.DEVICE}")
        self.whisper_model = whisper.load_model("base", device=self.DEVICE)
        self.sd_pipeline   = Pipeline.from_pretrained("pyannote/speaker-diarization")

        self.vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                        model='silero_vad',
                                                        force_reload=False)

        (self.vad_get_speech_timestamps,
         _, self.vad_read_audio,
         *_) = vad_utils

        self.stream_iter = 0
        self.diarization_text_result = ""
        self.run_online = False
        self.stream_start_time = 0
        self.debug       = None
        self.debug2      = 0
        self.speech_queue = []
        self.stream_results = ""

    def clear(self):
        self.speech_queue = []
        self.file_path = None
        self.diarization_text_result = ""
        self.stream_iter = 0
        self.stream_start_time = 0
        self.debug = None
        self.debug2 = 0
        self.stream_results = ""


STREAM_SLEEP_TIME_IN_SECONDS = 10
SAMPLE_RATE = 16000
extractVoiceProperties = CExtractVoiceProperties()
SAVE_RESULTS_PATH = "/home/amitli/Downloads"


def schedule_job():
    global extractVoiceProperties

    if  len(extractVoiceProperties.speech_queue) < 10:
        return None, None

    print(f"queue len = {len(extractVoiceProperties.speech_queue)}")
    speech = np.array([])
    for i in range(len(extractVoiceProperties.speech_queue)):
        speech = np.concatenate((speech, extractVoiceProperties.speech_queue[i]))

    speech = np.int16(speech / np.max(np.abs(speech)) * 32767)
    print(f"speech = {len(speech)}")
    extractVoiceProperties.speech_queue = []

    full_path = f"/home/amitli/Downloads/test.wav"
    from scipy.io.wavfile import write
    write(full_path, 48000, speech)

    speech =  utilities.convert_to_16sr_file(full_path, full_path)

    text, lang = whisperHandler.Get_Whisper_Text(extractVoiceProperties.whisper_model, speech)
    #print (f"lang = {lang}")
    output_stream_img = None

    extractVoiceProperties.stream_results = extractVoiceProperties.stream_results + "\n" + text
    output_stream_text = extractVoiceProperties.stream_results
    return output_stream_text, output_stream_img




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

        start_sample = int(start_time * SAMPLE_RATE)
        end_sample = int(end_time * SAMPLE_RATE)
        speaker_samples = speech[start_sample:end_sample]
        text, language = whisperHandler.Get_Whisper_Text(whisper_model, speaker_samples)
        l_speakers_samples.append(speaker_samples)
        l_speaker.append(speaker)
        l_text.append(text)

    return l_speakers_samples, l_speaker, l_text, language


def prepare_text(l_text, l_speaker, language):

    if language == "he":
        align = "right"
    else:
        align = "left"

    text_results = ""
    speaker_dict = {}
    colors = ["red", "green", "blue"]
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

def handle_wav_file(input_file):

    global extractVoiceProperties

    if extractVoiceProperties.run_online is False:
        with open(f"{SAVE_RESULTS_PATH}/html_res.txt", 'r') as file:
            html_whisper_text = file.read()

        diarization_figure, ax = plt.subplots()
        import matplotlib.image as mpimg
        img = mpimg.imread(f"{SAVE_RESULTS_PATH}/fig_res.png")
        ax.imshow(img)
        return html_whisper_text, diarization_figure

    if (input_file is None):
        print("Missing WAV file")
        return None, None

    file_duration = utilities.get_wav_duration(input_file)
    sample_rate   = utilities.get_sample_rate(input_file)
    print(f"file_duration = {file_duration},  sample_rate = {sample_rate}")

    if sample_rate != 16000:
        print("Change sample rate to 16000")
        utilities.convert_to_16sr_file(input_file, input_file)

    print("Run Diarization pipeline")
    diarization_res = extractVoiceProperties.sd_pipeline(input_file)

    print("Get speech from wav file")
    speech, sr = librosa.load(input_file, sr=SAMPLE_RATE)

    print("process diarzation results")
    res = process_diarizartion_results(diarization_res, speech, extractVoiceProperties.whisper_model)
    l_speakers_samples = res[0]
    l_speaker = res[1]
    l_text = res[2]
    language = res[3]

    print("prepare html whisper restlts")
    html_whisper_text = prepare_text(l_text, l_speaker, language)

    diarization_figure, ax = plt.subplots()
    res = notebook.plot_annotation(diarization_res, ax=ax, time=True, legend=True)

    print("Save results to files")
    save_results_to_file(html_whisper_text, diarization_figure)

    print(html_whisper_text)

    return html_whisper_text, diarization_figure


def create_vad_plot(vad_results, time_length, current_iter, last_results):

    current_time = current_iter*STREAM_SLEEP_TIME_IN_SECONDS
    time = np.arange(start=current_time, stop=current_time + STREAM_SLEEP_TIME_IN_SECONDS * 1000, step=1)
    values = np.zeros(current_time + STREAM_SLEEP_TIME_IN_SECONDS * 1000)
    for i in range(len(vad_results)):
        start_time_ms = vad_results[i]['start']
        end_time_ms   = vad_results[i]['end']
        values[start_time_ms:end_time_ms] = 1

    df = pd.DataFrame()
    df['time'] = time
    df['vad'] = values
    fig = px.line(df, x="time", y="vad", title='silero-vad')
    return fig


def handle_streaming(audio, state=""):

    #time.sleep(STREAM_SLEEP_TIME_IN_SECONDS)

    global extractVoiceProperties
    extractVoiceProperties.stream_iter = extractVoiceProperties.stream_iter + 1

    rate  = audio[0]
    voice = audio[1]
    #voice = np.int16(voice / np.max(np.abs(voice)) * 32767)
    extractVoiceProperties.speech_queue.append(voice)
    # if (extractVoiceProperties.stream_iter%20) == 0: # 10 seconds
    #     print("POP")
    #     extractVoiceProperties.speech_queue.pop(0)

    # from scipy.io.wavfile import write
    # full_path = f"/home/amitli/Downloads/1/{extractVoiceProperties.stream_iter}.wav"
    # write(full_path, rate, voice)


    diff_time = 0
    if extractVoiceProperties.debug is None:
        extractVoiceProperties.debug = time.time()
    else:
        current_time = time.time()
        diff_time    = current_time - extractVoiceProperties.debug
        extractVoiceProperties.debug = time.time()

    #print(f"len = {len(voice)}, sample rate = {rate}, diff time: {round(diff_time, 2)}")
    #state += ""
    #
    #
    # print("Run VAD")
    # # tensor_speech = extractVoiceProperties.vad_read_audio(full_path, sampling_rate=SAMPLE_RATE)
    # # speech_timestamps = extractVoiceProperties.vad_get_speech_timestamps(tensor_speech,
    # #                                                                      extractVoiceProperties.vad_model,
    # #                                                                      sampling_rate=SAMPLE_RATE)
    # vad_plot = None
    # # vad_plot = create_vad_plot(speech_timestamps,
    # #                            time_length=STREAM_SLEEP_TIME_IN_SECONDS,
    # #                            current_iter=extractVoiceProperties.stream_iter,
    # #                            last_results=None)
    #
    # print("Run Whisper")
    # text, lang = None, None #whisperHandler.Get_Whisper_Text(extractVoiceProperties.whisper_model, voice)
    # # if lang == "he":
    # #     align = "right"
    # # else:
    # #     align = "left"
    # # align = "left"
    # # text = "amit liron ok shira liron it ok too"
    # # #html_text = f"<p style='color:green; text-align:{align};'> {text} </p>" + "\n"
    # # html_text = text
    # # print(html_text)

    #return text, vad_plot, state



def set_running_option(value):
    global extractVoiceProperties
    if value == "Online calculation":
        extractVoiceProperties.run_online = True
    else:
        extractVoiceProperties.run_online = False


def create_gui():

    with gr.Blocks(theme=gr.themes.Glass()) as demo:

        radio_run_type = gr.Radio(["Online calculation", "Offline calculation"], value="Offline calculation", label="Choose how to run")
        with gr.Tab("Input"):
            with gr.Tab("Load File"):
                audioUpload        = gr.Audio(source="upload", type="filepath")
                audioProcessButton = gr.Button("Process")

            with gr.Tab("Record New File"):
                audioRecord = gr.Audio(source="microphone", type="filepath")
                audioShowFileButton = gr.Button("Show Save File")
                audioShowFileText = gr.Textbox()
                audioProcessRecButton = gr.Button("Process")
                audioShowFileButton.click(fn=show_saved_wav_file, inputs=audioRecord, outputs=[audioShowFileText])

            with gr.Tab("Live Streaming"):
                state         = gr.State(value="")
                stream_input  = gr.Audio(source="microphone")

        with gr.Tab("Diarization Output"):
            with gr.Row():
                output_text = gr.outputs.HTML(label="")
            with gr.Row():
                output_img = gr.Plot()

        with gr.Tab("Streaming Output"):
            with gr.Row():
                output_stream_text = gr.Text() #gr.outputs.HTML(label="")
            with gr.Row():
                output_stream_img = gr.Plot()

        with gr.Tab("About"):
            gr.Label("Version 1")

        radio_run_type.change(set_running_option, radio_run_type, [])
        audioProcessButton.click(fn=handle_wav_file, inputs=audioUpload, outputs=[output_text, output_img])
        audioProcessRecButton.click(fn=handle_wav_file, inputs=audioRecord, outputs=[output_text, output_img])
        stream_input.stream(fn=handle_streaming,
                            inputs=[stream_input, state],
                            outputs=[]) # output_stream_text, output_stream_img, state

        demo.load(schedule_job, None, [output_stream_text, output_stream_img], every=5)

    return demo

if __name__ == "__main__":


    utilities.save_huggingface_token()
    demo = create_gui()
    demo.queue().launch()
    #demo.queue().launch(share=True, debug=False)
