import os
import gradio as gr

def handle_streaming(audio):
    print("---")


def handle_button(a_in):
    print("button...")

def create_new_gui():
    with gr.Blocks(theme=gr.themes.Glass()) as demo:

        with gr.Tab("Real Time"):
            stream_input       = gr.Audio(source="microphone")

        with gr.Tab("Offline"):
            with gr.Row():
                audioUpload = gr.Audio(source="upload", type="filepath")
                audioRecord = gr.Audio(source="microphone", type="filepath")
                button      = gr.Button("Click On Me")

            button.click(fn=handle_button, inputs=[audioRecord],
                                        outputs=[])

        stream_input.stream(fn=handle_streaming,
                            inputs=[stream_input],
                            outputs=[])

    return demo



if __name__ == "__main__":
    demo = create_new_gui()
    demo.queue().launch(share=False, debug=False)