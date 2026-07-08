import gradio as gr
from consuta import respuesta

def chat(message, history):
    return respuesta(message)

gr.ChatInterface(fn=chat).launch(
    server_name="0.0.0.0",
    server_port=7860
)