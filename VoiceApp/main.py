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
from datetime                          import datetime

import matplotlib.pyplot               as plt
import plotly.express                  as px



class CExtractVoiceProperties:

    def __init__(self):
        self.file_path     = None
        self.DEVICE        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Device = {self.DEVICE}")
        self.whisper_model = whisper.load_model("large", device=self.DEVICE)
        self.sd_pipeline   = Pipeline.from_pretrained("pyannote/speaker-diarization")

        self.vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                        model='silero_vad',
                                                        force_reload=False)

        (self.vad_get_speech_timestamps,
         self.vad_save_audio,
         self.vad_read_audio,
         self.VADIterator,
         self.vad_collect_chunks) = vad_utils

        self.stream_iter = 0
        self.diarization_text_result = ""
        self.run_online = True
        self.stream_start_time = 0
        self.speech_queue = []
        self.vad_queue = []
        self.stream_results = ""
        self.last_lang = ""
        self.last_text = ""
        self.last_vad_plot = None
        self.init()

    def clear(self):
        self.speech_queue = []
        self.vad_queue = []
        self.file_path = None
        self.diarization_text_result = ""
        self.stream_iter = 0
        self.stream_start_time = 0
        self.stream_results = ""
        self.last_lang = ""
        self.last_text = ""
        self.last_vad_plot = None


    def init(self):

        INIT_WAV = f"{os.getcwd()}/init.wav"
        print(f"Start init pyyanote ({datetime.now()})")
        diarization_res = self.sd_pipeline(INIT_WAV)
        print(f"Finish init pyyanote ({datetime.now()})")

        print(f"Start init Whisper ({datetime.now()})")
        res = self.Get_Whisper_Text(INIT_WAV)
        print(f"Finish init Whisper ({datetime.now()})")

    def Get_Whisper_Text(self,  file_name):

        audio = whisper.load_audio(file_name)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.whisper_model.device)

        # decode the audio
        options = whisper.DecodingOptions(beam_size=5, fp16=False)
        result = whisper.decode(self.whisper_model, mel, options)
        result = result.text

        return result


STREAM_SLEEP_TIME_IN_SECONDS = 10
MICROPHONE_SAMPLE_RATE = 48000
SAMPLE_RATE = 16000
extractVoiceProperties = CExtractVoiceProperties()
SAVE_RESULTS_PATH = f"{os.getcwd()}/TmpFiles"



def schedule_vad_job2():
    global extractVoiceProperties
    print(f"VAD {datetime.now()}, Q = {len(extractVoiceProperties.vad_queue)}")

    speech = np.array([])
    q_len = len(extractVoiceProperties.vad_queue)
    if q_len <= 0:
        return extractVoiceProperties.last_vad_plot

    for i in range(q_len):
        speech = np.concatenate((speech, extractVoiceProperties.vad_queue[i]))
    speech = np.int16(speech / np.max(np.abs(speech)) * 32767)

    if len(speech) > (30 * MICROPHONE_SAMPLE_RATE):
        print("ERROR")

    vad_iterator = extractVoiceProperties.VADIterator(extractVoiceProperties.vad_model)


def schedule_vad_job():
    global extractVoiceProperties

    LAST_30_SEC_IN_Q = 60
    LAST_30_SEC_IN_SAMPLES = 30 * SAMPLE_RATE

    print (f"VAD {datetime.now()}, Q = {len(extractVoiceProperties.vad_queue)}")

    # get last 30 seconds speech
    speech = np.array([])
    q_len  = len(extractVoiceProperties.vad_queue)
    if q_len <= 0:
        return extractVoiceProperties.last_vad_plot

    for i in range(q_len):
        speech = np.concatenate((speech, extractVoiceProperties.vad_queue[i]))
    speech = np.int16(speech / np.max(np.abs(speech)) * 32767)

    if len(speech) > (30 * MICROPHONE_SAMPLE_RATE):
        stream_iter = (len(speech) / MICROPHONE_SAMPLE_RATE) - 30
        extractVoiceProperties.stream_iter = extractVoiceProperties.stream_iter + stream_iter
        speech                           = speech[-MICROPHONE_SAMPLE_RATE * 30: ]
        extractVoiceProperties.vad_queue = extractVoiceProperties.vad_queue[-LAST_30_SEC_IN_Q:]


    full_path = f"{SAVE_RESULTS_PATH}/last_vad.wav"
    from scipy.io.wavfile import write
    write(full_path, MICROPHONE_SAMPLE_RATE, speech)

    utilities.convert_to_16sr_file(full_path, full_path)
    current_length = utilities.get_wav_duration(full_path)

    tensor_speech     = extractVoiceProperties.vad_read_audio(full_path, sampling_rate=SAMPLE_RATE)
    speech_timestamps = extractVoiceProperties.vad_get_speech_timestamps(tensor_speech,
                                                                         extractVoiceProperties.vad_model,
                                                                         sampling_rate=SAMPLE_RATE)

    vad_plot = create_vad_plot(vad_results     = speech_timestamps,
                               start_plot_time = extractVoiceProperties.stream_iter,
                               current_length  = current_length
                               )
    extractVoiceProperties.last_vad_plot = vad_plot

    return vad_plot
#

def schedule_whisper_job():
    '''

    :return: text with whisper prediction
             Every X seconds (see create_gui) we predict the text
    '''
    global extractVoiceProperties

    print (f"whisper Task {datetime.now()}, Q = {len(extractVoiceProperties.speech_queue)}")

    if len(extractVoiceProperties.speech_queue) == 0:
        return extractVoiceProperties.last_lang , extractVoiceProperties.last_text

    speech = np.array([])
    for i in range(len(extractVoiceProperties.speech_queue)):
        speech = np.concatenate((speech, extractVoiceProperties.speech_queue[i]))

    speech = np.int16(speech / np.max(np.abs(speech)) * 32767)
    extractVoiceProperties.speech_queue = []

    full_path = f"{SAVE_RESULTS_PATH}/last_whisper.wav"
    from scipy.io.wavfile import write
    write(full_path, 48000, speech)

    speech =  utilities.convert_to_16sr_file(full_path, full_path)

    print ("Run Whisper")
    text, lang = whisperHandler.Get_Whisper_Text(extractVoiceProperties.whisper_model, speech)
    print("Whisper Results:")
    print(f"\tLang: {lang}")
    print(f"\ttext: {text}")

    extractVoiceProperties.stream_results = extractVoiceProperties.stream_results + "\n" + text
    output_stream_text = extractVoiceProperties.stream_results

    if lang == "he":
        detect_lang = "Detect Language: Hebrew"
    elif lang == "en":
        detect_lang = "Detect Language: English"
    else:
        detect_lang = f"Detect Language: {lang}"

    extractVoiceProperties.last_lang = detect_lang
    extractVoiceProperties.last_text = output_stream_text

    return detect_lang, output_stream_text




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


def create_vad_plot(vad_results,  start_plot_time, current_length):
    '''
    :param vad_results:         list with vad results (start/end time in ms)
    :param start_plot_time:     start the plot display from this start time
    :param current_length:      length of the VAD prediction window
    :return:                    plot (x = time, y = [0/1]) with VAD results in this window
    '''

    time   = np.arange(start=start_plot_time, stop = start_plot_time + current_length * SAMPLE_RATE, step=1)
    time   = time / SAMPLE_RATE
    values = np.zeros(len(time))

    for i in range(len(vad_results)):
        start_time_ms = vad_results[i]['start']
        end_time_ms   = vad_results[i]['end']
        values[start_time_ms:end_time_ms] = 1

    df = pd.DataFrame()
    df['time'] = time
    df['vad'] = values
    fig = px.line(df, x="time", y="vad", title='silero-vad')
    return fig


def handle_streaming(audio):

    global extractVoiceProperties

    rate  = audio[0]
    voice = audio[1]
    extractVoiceProperties.speech_queue.append(voice)
    extractVoiceProperties.vad_queue.append(voice)



def set_running_option(value):
    global extractVoiceProperties
    if value == "Run Diariation Model":
        extractVoiceProperties.run_online = True
    else:
        extractVoiceProperties.run_online = False



def create_new_gui():
    with gr.Blocks(theme=gr.themes.Glass()) as demo:


        with gr.Tab("Real Time"):
            stream_input       = gr.Audio(source="microphone")
            output_stream_lang = gr.Label(label = "Detect Lanugage: ")
            output_stream_text = gr.Text(label = "Whisper Results:")
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

        with gr.Tab("About"):
            gr.Label("Version 1")

        stream_input.stream(fn      = handle_streaming,
                            inputs  = [stream_input],
                            outputs = [])

        demo.load(schedule_whisper_job, None, [output_stream_lang, output_stream_text], every=5)
        demo.load(schedule_vad_job, None, [output_stream_plt], every=1)
    return demo



if __name__ == "__main__":

    utilities.save_huggingface_token()
    demo = create_new_gui()
    #demo.queue().launch()
    demo.queue().launch(share=True, debug=False)
