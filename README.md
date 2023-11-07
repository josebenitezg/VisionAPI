## VisionAPI 👀 🚧

#### Hey there

This is a Work In Progress Project.
The goal is to bring GPT-based Models to a simple API

### How to use

##### Installation

```bash
pip install visionapi
```
##### Authentication

```bash
export OPENAI_API_KEY=<your key>
```
##### Image Inference
We can use an image url, local image path or numpy array to make an inference.

```python
import VisionAPI

inference_endpoint = VisionAPI.Inference()

image = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

prompt = "Describe the image"

response = inference_endpoint.image_inference(image, prompt)

print(response)

```
##### Video Inference

```python
import VisionAPI

inference_endpoint = VisionAPI.Inference()

prompt = "These are frames from a video that I want to upload. Generate a compelling description that I can upload along with the video."

video = "video.mp4"

response = inference_endpoint.video_inference(video, prompt)

print(response)

```

Contribute to this project by adding more models and features.