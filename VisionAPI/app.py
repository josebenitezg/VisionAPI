import os
import cv2
import base64
import numpy as np
from VisionAPI.utils import encode_video
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_MODEL = "gpt-4-vision-preview"

class Inference:
    def __init__(self):
        self.client = OpenAI()
        if OPENAI_API_KEY is None:
            raise ValueError("API_KEY is not set")
        self.api_key = OPENAI_API_KEY
        self.gpt_model = GPT_MODEL
    
    def image_inference(self, image_input, prompt) -> str:
    # Function to encode numpy array or image file to base64
        def encode_to_base64(image):
            if isinstance(image, np.ndarray):
                success, encoded_image = cv2.imencode('.jpg', image)
                if not success:
                    raise ValueError("Could not encode image")
                return base64.b64encode(encoded_image).decode('utf-8')
            elif isinstance(image, str) and os.path.isfile(image):
                with open(image, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
            else:
                raise ValueError("Invalid image input")

        # Prepare the message for the API call
        message_content = [{"type": "text", "text": prompt}]
        if isinstance(image_input, str) and image_input.startswith("http"):
            # Assume image_input is a URL
            message_content.append({
                "type": "image_url",
                "image_url": image_input,
            })
        else:
            # Assume image_input is a numpy array or file path and convert to base64
            base64_image = encode_to_base64(image_input)
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

        # Make the API call
        response = self.client.chat.completions.create(
            model=self.gpt_model,
            messages=[{"role": "user", "content": message_content}],
            max_tokens=300,
        )
        return response.choices[0]


    def video_inference(self, input_video, prompt) -> str:
        '''
        We don't need to send every frame for GPT 
        to understand what's going on
        '''
        base64Frames = encode_video(input_video)
        
        PROMPT_MESSAGES = [
            {
                "role": "user",
                "content": [
                    prompt,
                    *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::10]),
                ],
            },
        ]
        
        params = {
            "model": self.gpt_model,
            "messages": PROMPT_MESSAGES,
            "max_tokens": 200,
        }

        result = self.client.chat.completions.create(**params)
        return result.choices[0].message.content
        
    def webcam_inference(self, prompt, device=0) -> str:
        print('coming soon')