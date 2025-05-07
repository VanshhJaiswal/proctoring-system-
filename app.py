import streamlit as st
import time
import os
import base64
import cv2
import numpy as np
from google.cloud import vision
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google Vision API Setup
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
vision_client = vision.ImageAnnotatorClient()

st.set_page_config(page_title="AI Proctoring System", layout="wide")
st.title("🛡️ AI-Based Online Proctoring System")

# Initialize session state
if 'proctoring_logs' not in st.session_state:
    st.session_state.proctoring_logs = []

# Function to analyze image with Google Vision API
def analyze_image(image_bytes):
    image = vision.Image(content=image_bytes)
    response = vision_client.face_detection(image=image)
    faces = response.face_annotations

    mobile_detected = False
    multiple_faces_detected = len(faces) > 1
    face_absent = len(faces) == 0

    return {
        'multiple_faces': multiple_faces_detected,
        'face_absent': face_absent,
        'mobile_detected': mobile_detected,  # Placeholder
        'face_count': len(faces)
    }

# Function to process proctoring data
def process_proctoring_data(result):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alerts = []
    if result['multiple_faces']:
        alerts.append("⚠️ Multiple faces detected!")
    if result['face_absent']:
        alerts.append("⚠️ Candidate not visible!")
    if result['mobile_detected']:
        alerts.append("⚠️ Mobile phone detected!")
    log_entry = {
        "timestamp": timestamp,
        "face_count": result['face_count'],
        "alerts": ", ".join(alerts) if alerts else "✅ No issues"
    }
    return log_entry

# Webcam image capture
st.subheader("📷 Live Webcam Snapshot")
st.write("Take a snapshot to analyze for proctoring checks")
image = st.camera_input("Capture from Webcam")

if image:
    image_bytes = image.getvalue()
    result = analyze_image(image_bytes)
    log = process_proctoring_data(result)
    st.session_state.proctoring_logs.append(log)

    st.markdown("### 📝 Latest Monitoring Result")
    st.write(log)

# Display full logs
if st.session_state.proctoring_logs:
    st.markdown("---")
    st.subheader("📊 Proctoring Log History")
    st.dataframe(st.session_state.proctoring_logs)

# Download log
def convert_logs_to_csv(logs):
    import pandas as pd
    df = pd.DataFrame(logs)
    return df.to_csv(index=False).encode('utf-8')

if st.session_state.proctoring_logs:
    csv = convert_logs_to_csv(st.session_state.proctoring_logs)
    st.download_button(
        label="📥 Download Logs as CSV",
        data=csv,
        file_name='proctoring_logs.csv',
        mime='text/csv'
    )
