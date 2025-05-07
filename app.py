import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np
import math

# Load classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_mcs_mouth.xml")

class ProctoringTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        alert = ""
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(roi_gray)
            mouths = mouth_cascade.detectMultiScale(roi_gray, 1.5, 11)
            
            # Eye detection
            if len(eyes) >= 2:
                eye_centers = []
                for (ex, ey, ew, eh) in eyes[:2]:
                    center = (ex + ew//2, ey + eh//2)
                    eye_centers.append(center)
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
                
                # Head pose estimation: simple angle
                if len(eye_centers) == 2:
                    dx = eye_centers[1][0] - eye_centers[0][0]
                    dy = eye_centers[1][1] - eye_centers[0][1]
                    angle = math.degrees(math.atan2(dy, dx))
                    if abs(angle) > 20:
                        alert = "Looking Away!"
                        cv2.putText(img, alert, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            # Mouth detection
            for (mx, my, mw, mh) in mouths:
                if my > h/2:  # lower half only
                    cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0,0,255), 2)
                    alert = "Mouth Open!"
                    cv2.putText(img, alert, (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        if len(faces) == 0:
            cv2.putText(img, "No face detected!", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        return img

st.title("AI Proctoring System")
st.write("Click below to start webcam monitoring.")

rtc_config = {
    "iceServers": [
        {"urls": "stun:stun.l.google.com:19302"},
        {"urls": "turn:openrelay.metered.ca:80", "username": "openrelayproject", "credential": "openrelayproject"},
        {"urls": "turn:openrelay.metered.ca:443", "username": "openrelayproject", "credential": "openrelayproject"}
    ]
}

webrtc_streamer(
    key="proctoring",
    video_transformer_factory=ProctoringTransformer,
    rtc_configuration=rtc_config,
    media_stream_constraints={"video": True, "audio": False}
)
