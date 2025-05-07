import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime

st.title("Online Proctoring System (Streamlit)")

st.markdown("""
This system checks for:
- Face presence
- Multiple faces detection
""")

uploaded_image = st.camera_input("Take a snapshot to verify")

if uploaded_image is not None:
    # Convert to numpy array
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Initialize Mediapipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        # Convert image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(img_rgb)

        # Count faces
        num_faces = 0
        if results.detections:
            num_faces = len(results.detections)
            for detection in results.detections:
                mp_drawing.draw_detection(img, detection)

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", caption=f"Detected {num_faces} face(s)")

        if num_faces == 0:
            st.error("⚠️ No face detected! Please stay in front of the camera.")
        elif num_faces > 1:
            st.warning(f"⚠️ Multiple faces detected! ({num_faces} faces)")
        else:
            st.success("✅ Face detected successfully. You may continue.")

        # Save log
        with open("log.txt", "a") as f:
            f.write(f"{datetime.now()} - Faces detected: {num_faces}\n")
        st.write("Log updated.")

