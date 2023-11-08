'''
example source: https://github.com/roboflow/webcamGPT/blob/main/examples/webcam.py
'''
import os
import cv2
import uuid

import gradio as gr
import numpy as np

import visionapi

HEADER = """
# 📸 VisionAPI. 
"""

inference = visionapi.Inference()

def save_image_to_drive(image: np.ndarray) -> str:
    image_filename = f"{uuid.uuid4()}.jpeg"
    image_directory = "data"
    os.makedirs(image_directory, exist_ok=True)
    image_path = os.path.join(image_directory, image_filename)
    cv2.imwrite(image_path, image)
    return image_path


def respond(image: np.ndarray, prompt: str, chat_history):
    image = np.fliplr(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_path = save_image_to_drive(image)
    response = inference.image(image, prompt)
    chat_history.append(((image_path,), None))
    chat_history.append((prompt, response.message.content))
    return "", chat_history


with gr.Blocks() as demo:
    gr.Markdown(HEADER)
    with gr.Row():
        webcam = gr.Image(source="webcam", streaming=True)
        with gr.Column():
            chatbot = gr.Chatbot(height=500)
            message = gr.Textbox()
            clear_button = gr.ClearButton([message, chatbot])

    message.submit(respond, [webcam, message, chatbot], [message, chatbot])

demo.launch(debug=False, show_error=True)