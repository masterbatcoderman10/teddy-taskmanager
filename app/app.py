from utils.ai import get_answer, get_answer_alt
import gradio as gr

gr.ChatInterface(get_answer_alt).launch()