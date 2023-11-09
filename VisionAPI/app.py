import os
import cv2
import base64
import requests
import numpy as np
from pathlib import Path
from visionapi.utils import encode_video
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPTV_MODEL = "gpt-4-vision-preview"
GPT_MODEL = "gpt-4-1106-preview"
TTS_MODEL = "tts-1"  
STT_MODEL = "whisper-1"
VOICE = "alloy"  
IMAGE_GEN_MODEL = "dall-e-3"

class Inference:
    '''
    The Inference class provides methods to perform image and video inference, 
    text-to-speech conversion, and image generation with OpenAI's models.

    Attributes:
    - client (OpenAI): The OpenAI client initialized with the API key.
    - gpt4v_model (str): The identifier for the GPT model used for inferences.
    
    Methods:
    - image_inference: Processes an image with a given prompt and returns the result.
    - video_inference: Processes a video with a given prompt and returns the result.
    - webcam_inference: Placeholder for future webcam processing functionality.
    - text_to_speech: Converts given text to speech and saves it to a file.
    - generate_image: Generates images based on a text prompt and optionally saves them to disk.

    Initialization:
    The constructor initializes the OpenAI client and sets the GPT model. 
    It requires an environment variable 'OPENAI_API_KEY' to be set for the API key.
    
    Example:
    inference = Inference()
    '''
    def __init__(self):
        self.client = OpenAI()
        if OPENAI_API_KEY is None:
            raise ValueError("API_KEY is not set")
        self.api_key = OPENAI_API_KEY
        self.gptv_model = GPTV_MODEL
        self.gpt_model = GPT_MODEL
    
    def image(self, image_input, prompt) -> str:
        '''
        Processes an image or image URL with a given text prompt using the specified GPT model.
        Returns the inference result as a string.
        
        Parameters:
        - image_input (np.ndarray or str): A numpy array of the image or a URL pointing to an image.
        - prompt (str): A text prompt for the GPT model to interpret the image.

        Example:
        response = inference.image(image_array, "What is in this image?")
        '''
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
            model=self.gptv_model,
            messages=[{"role": "user", "content": message_content}],
            max_tokens=300,
        )
        return response.choices[0]


    def video(self, input_video, prompt) -> str:
        '''
        Processes a video file with a given text prompt and returns the result.
        The method selects frames from the video and sends them for processing.

        Parameters:
        - input_video (str): A file path to the input video.
        - prompt (str): A text prompt for the GPT model to interpret the video.

        Example:
        response = inference.video_inference("path/to/video.mp4", "Summarize the actions in the video.")
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
            "model": self.gptv_model,
            "messages": PROMPT_MESSAGES,
            "max_tokens": 200,
        }

        result = self.client.chat.completions.create(**params)
        return result.choices[0].message.content
    
    def TTS(self, text, save_path, stream=True):
        '''
        Converts given text to speech using the TTS model and saves the output as an audio file.

        Parameters:
        - text (str): The text to convert to speech.
        - save_path (str): The file path where the audio will be saved.
        - stream (bool): If True, streams the audio in real-time (default False).

        Example:
        inference.text_to_speech("Hello world!", "path/to/save/speech.mp3", stream=True)
        '''
        # Ensure the save_path is a Path object
        save_path = Path(save_path)
        
        # Prepare the TTS API parameters
        tts_params = {
            "model": TTS_MODEL,
            "voice": VOICE,
            "input": text
        }
        
        # Create the TTS response
        response = self.client.audio.speech.create(**tts_params)
        
        # If streaming is desired, stream to file, otherwise save the file normally
        if stream:
            response.stream_to_file(str(save_path))
        else:
            with save_path.open("wb") as f:
                f.write(response.audio)
        
        return str(save_path)
    
    def generate_image(self, prompt, size="1024x1024", quality="standard", qty=1, save=False):
        '''
        Generates images based on a text prompt using the DALL-E 3 model.
        
        Parameters:
        - prompt (str): The text prompt to generate images from.
        - size (str): The size of the generated images (default "1024x1024").
        - quality (str): The quality of the generated images, e.g., "standard".
        - n (int): The number of images to generate. Note: DALL-E 3 API currently
        allows generating one image at a time, so this method will make `n` separate
        requests to generate `n` images.
        - save (bool): If True, saves the generated images to disk in the current 
        directory. The filenames will be `generated_image_1.png`, `generated_image_2.png`, etc.
        
        Returns:
        - List of URLs of the generated images if `save` is False. If `save` is True,
        returns a list of file paths to the saved images.
        
        Example usage:
        inference = Inference()
        image_urls = inference.generate_image("a white siamese cat", n=10, save=True)
        '''
        image_urls = []
        
        for _ in range(qty):  # Loop over the number of images requested
            # Create the image generation response
            response = self.client.images.generate(
                model=IMAGE_GEN_MODEL,
                prompt=prompt,
                size=size,
                quality=quality,
                n=1  # Each request generates one image
            )
            image_url = response.data[0].url
            image_urls.append(image_url)

            if save:
                # Save the image to disk
                image_response = requests.get(image_url)
                image_response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
                
                # Construct the image filename
                image_number = len(image_urls)
                image_path = Path(f"generated_image_{image_number}.png")
                with open(image_path, "wb") as f:
                    f.write(image_response.content)
                image_urls[-1] = str(image_path)  # Update the URL to local path

        return image_urls
    
    def STT(self, audio_file_path, response_format="text"):
        '''
        Converts speech from an audio file to text using the Whisper model.

        Parameters:
        - audio_file_path (str): The file path to the audio file to transcribe.
        - response_format (str): The format of the response. Can be "json" or "text". Default is "json".

        Returns:
        - If response_format is "json", returns the full JSON response from the API.
        - If response_format is "text", returns just the transcribed text as a string.

        Example usage:
        transcript = inference.speech_to_text("/path/to/audio.mp3")
        print(transcript)
        '''
        with open(audio_file_path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model=STT_MODEL,
                file=audio_file,
                response_format=response_format
            )
        if response_format == "text":
            return transcript
        else:
            return transcript["text"]

    def generate_text(self, messages, stream=False, stop=None, temperature=1, max_tokens=150, top_p=1, frequency_penalty=0, presence_penalty=0):
        '''
        Generates text using OpenAI's text generation models with the latest API.
        This method can stream the responses if needed and can be configured for 
        various parameters such as temperature and max tokens.

        Parameters:
        - system (dict): The system content that includes instructions or examples for the model.
        - user_prompt (str): The user's input, which acts as the prompt for the model.
        - json_mode (bool): If True, outputs in JSON format.
        - stop (list of str): A list of stop sequences where the model's output will stop.
        - temperature (float): Controls randomness. Lowering results in more deterministic outputs.
        - max_tokens (int): The maximum number of tokens to generate.
        - top_p (float): Controls diversity. Lowering results in less random completions.
        - frequency_penalty (float): The penalty for new tokens based on their frequency.
        - presence_penalty (float): The penalty for new tokens based on their presence.

        Returns:
        - A generator that yields the model's outputs as they become available.

        Example usage:
        system = {"prompt": "Translate the following sentences into French:"}
        user_prompt = "How are you today?"
        for chunk in inference.generate_text(system, user_prompt, stream=True):
            print(chunk['choices'][0]['message']['content'])
        '''

        params = {
            "model": self.gpt_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop
        }

        completion = self.client.chat.completions.create(**params, stream=stream)

        return completion  # Return the generator object for JSON responses
