import gradio as gr
from consuta import respuesta

def chat(message, history):
    return respuesta(message)

gr.ChatInterface(fn=chat).launch()