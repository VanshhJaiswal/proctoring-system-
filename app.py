import streamlit as st
from PIL import Image, ImageDraw
import mediapipe as mp
import numpy as np
from datetime import datetime

st.title("Online Proctoring System (Streamlit - OpenCV-Free)")

st.markdown("""
This system checks for:
- Face presence
- Multiple faces detection
""")

uploaded_image = st.camera_input("Take a snapshot to verify")

if uploaded_image is not None:
    # Load image with PIL
    image = Image.open(uploaded_image)

    # Convert to numpy array
    img_array = np.array(image)

    # Mediapipe face detection
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(img_array)

        num_faces = 0
        draw = ImageDraw.Draw(image)

        if results.detections:
            num_faces = len(results.detections)
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img_array.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                # Draw rectangle
                draw.rectangle([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]], outline="red", width=3)

        st.image(image, caption=f"Detected {num_faces} face(s)")

        if num_faces == 0:
            st.error("⚠️ No face detected! Please stay in front of the camera.")
        elif num_faces > 1:
            st.warning(f"⚠️ Multiple faces detected! ({num_faces} faces)")
        else:
            st.success("✅ Face detected successfully. You may continue.")

        # Log attempt
        with open("log.txt", "a") as f:
            f.write(f"{datetime.now()} - Faces detected: {num_faces}\n")
        st.write("Log updated.")
