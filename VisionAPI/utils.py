import cv2
import base64
import numpy as np


def encode_video(input_video):
    video = cv2.VideoCapture(input_video)
    base64Frames = []

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()

    print(len(base64Frames), "frames read.")
    return base64Frames