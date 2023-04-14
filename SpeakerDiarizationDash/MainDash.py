import sounddevice as sd
import soundfile as sf
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State

# Initialize the Dash app
app = dash.Dash(__name__)

# Set up the layout of the app
app.layout = html.Div([
    html.Button('Start Recording', id='record-button'),
])

num_of_clicks = 0


@app.callback(Output('record-button', 'children'),
              Input('record-button', 'n_clicks'),
              )
def handle_button_click(n_clicks):
    global num_of_clicks
    if num_of_clicks == 0:
        button_text = 'Start Recording'
    elif num_of_clicks%2 == 1:
        button_text =  'Start Recording'
        record_from_mic()
    else:
        button_text = "Stop Recording"
    num_of_clicks = num_of_clicks + 1
    return button_text

# Run the app

def record_from_mic():
    fs = 44100
    recording = None
    recording_length = 5
    print("Talk...")
    recording = sd.rec(int(recording_length * fs), samplerate=fs, channels=1)
    sd.wait()
    filename = "/home/amitli/Downloads/test.wav"
    sf.write(filename, recording, fs)
    print("Create wav")

    None

if __name__ == '__main__':
    print("Start...")
    app.run_server(debug=True)
    #record_from_mic()
    print("ENd")