import sounddevice as sd
import soundfile as sf
import dash
from dash import html
from dash.dependencies import Input, Output


app = dash.Dash(__name__)


app.layout = html.Div([
    html.Button('Start Recording', id='record-button'),
])


@app.callback(Output('record-button', 'children'),
              Input('record-button', 'n_clicks'),
              )
def handle_button_click(n_clicks):

    text = "Start Recording"
    if n_clicks is not None:
        record_from_mic()
        text = "Finished"

    return text

# Run the app

def record_from_mic():
    fs = 44100
    recording_length = 5
    recording = sd.rec(int(recording_length * fs), samplerate=fs, channels=1)
    sd.wait()
    filename = "/home/amitli/Downloads/test.wav"
    sf.write(filename, recording, fs)

    None

if __name__ == '__main__':
    app.run_server(debug=True)
    #app.run_server(debug=False, host="0.0.0.0")
