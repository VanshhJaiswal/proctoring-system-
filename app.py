import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import mediapipe as mp
import cv2
import numpy as np

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

class MediapipeTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        self.face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = self.face_detection.process(img_rgb)
        mesh_results = self.face_mesh.process(img_rgb)
        
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                x,y,w,h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        
        if mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                # mouth open detection (landmark 13 = upper lip, 14 = lower lip)
                top_lip = face_landmarks.landmark[13]
                bottom_lip = face_landmarks.landmark[14]
                
                ih, iw, _ = img.shape
                top = np.array([int(top_lip.x * iw), int(top_lip.y * ih)])
                bottom = np.array([int(bottom_lip.x * iw), int(bottom_lip.y * ih)])
                
                distance = np.linalg.norm(top - bottom)
                
                cv2.circle(img, tuple(top), 2, (0,255,0), -1)
                cv2.circle(img, tuple(bottom), 2, (0,0,255), -1)
                cv2.line(img, tuple(top), tuple(bottom), (255,0,0), 2)
                
                if distance > 15:
                    cv2.putText(img, "Mouth Open!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        return img

st.title("AI Proctoring with Mediapipe")
st.write("Click start to begin webcam monitoring.")

rtc_config = {
    "iceServers": [{"urls": "stun:stun.l.google.com:19302"}]
}

webrtc_streamer(
    key="proctor",
    video_transformer_factory=MediapipeTransformer,
    rtc_configuration=rtc_config,
    media_stream_constraints={"video": True, "audio": False}
)
