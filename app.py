import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np

# Face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_mcs_mouth.xml")

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

            mouths = mouth_cascade.detectMultiScale(roi_gray, 1.5, 5)
            for (mx, my, mw, mh) in mouths:
                if my > h/2:  # lower half only
                    cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0,0,255), 2)

        return img

st.title("AI Proctoring System")
st.write("Click below to start webcam and monitoring")

rtc_config = {
    "iceServers": [
        {"urls": "stun:stun.l.google.com:19302"},
        {"urls": "turn:openrelay.metered.ca:80", "username": "openrelayproject", "credential": "openrelayproject"},
        {"urls": "turn:openrelay.metered.ca:443", "username": "openrelayproject", "credential": "openrelayproject"}
    ]
}

webrtc_streamer(
    key="proctoring",
    video_transformer_factory=VideoTransformer,
    rtc_configuration=rtc_config,
    media_stream_constraints={"video": True, "audio": False}
)
