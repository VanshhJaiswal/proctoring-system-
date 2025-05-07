import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import math
import mediapipe as mp

st.title("AI Proctoring System")

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_mouth.xml')

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.cheating_detected = False
        self.mouth_open = False

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            # Eye detection
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

            # Mouth detection
            mouth = mouth_cascade.detectMultiScale(roi_gray, 1.7, 11)
            for (mx, my, mw, mh) in mouth:
                if my > h / 2:  # only consider mouth in lower half
                    cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0,0,255), 2)
                    self.mouth_open = True
                else:
                    self.mouth_open = False

        # Head pose detection
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]

                dx = right_eye.x - left_eye.x
                dy = right_eye.y - left_eye.y
                angle = math.degrees(math.atan2(dy, dx))

                if abs(angle) > 15:
                    self.cheating_detected = True
                    cv2.putText(img, f"Head turned ({int(angle)} deg)", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                else:
                    self.cheating_detected = False

        if self.mouth_open:
            cv2.putText(img, "Mouth Open Detected!", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        if len(faces) == 0:
            cv2.putText(img, "No face detected!", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        return img

webrtc_streamer(key="proctoring", video_transformer_factory=VideoTransformer)
