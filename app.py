
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import os
import requests
from PIL import Image
from transformers import pipeline

# Load Groq API Key from Streamlit Secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Mediapipe face mesh and drawing utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Pose estimation using Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load Hugging Face QA model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Detection log
activity_log = []

def detect_features(image):
    report = []
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect face landmarks
    face_results = face_mesh.process(img_rgb)
    if face_results.multi_face_landmarks:
        report.append("✅ Face detected")

        for landmarks in face_results.multi_face_landmarks:
            # Detect mouth opening (based on distance between upper and lower lips)
            upper_lip = landmarks.landmark[13]
            lower_lip = landmarks.landmark[14]
            lip_distance = abs(upper_lip.y - lower_lip.y)
            if lip_distance > 0.03:
                report.append("⚠️ Mouth open detected")

    else:
        report.append("❌ No face detected")

    # Pose estimation (to infer head pose roughly)
    pose_results = pose.process(img_rgb)
    if pose_results.pose_landmarks:
        head_y = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y
        left_ear_y = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y
        if abs(head_y - left_ear_y) > 0.05:
            report.append("⚠️ Head pose unusual")

    # Simulated phone detection — crude detection using rectangle assumptions (placeholder)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    phones = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
    if phones is not None:
        report.append("📱 Phone-like object detected")

    return report

def generate_mock_questions():
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "user", "content": "Generate 5 multiple-choice general knowledge questions with 4 options each and indicate the correct option."}
        ],
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=data)
    result = response.json()

    if "error" in result:
        st.error(f"Groq API Error: {result['error']['message']}")
        return []

    return result['choices'][0]['message']['content']

def run_test():
    st.title("🎓 Smart Proctored Mock Test System")
    st.write("This test monitors you in real-time using face, mouth, head, and phone detection. Answer the questions after the timer.")

    mock_questions = generate_mock_questions()
    if not mock_questions:
        return

    st.subheader("📄 AI-Generated Mock Test")
    st.markdown(mock_questions)

    # Start camera
    run_time = st.slider("Test Duration (seconds)", 10, 60, 20)
    stframe = st.empty()
    camera = cv2.VideoCapture(0)
    start_time = time.time()

    while time.time() - start_time < run_time:
        ret, frame = camera.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        detections = detect_features(frame)
        activity_log.extend(detections)

        for d in detections:
            cv2.putText(frame, d, (10, 25 + 30 * detections.index(d)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        stframe.image(frame, channels="BGR")

    camera.release()
    st.success("Test completed. Scroll down to download report.")

    # Show report
    st.subheader("📊 Proctoring Report")
    unique_logs = list(set(activity_log))
    for log in unique_logs:
        st.write(f"- {log}")

    # Generate downloadable report
    report_text = "\n".join(unique_logs)
    with open("proctoring_report.txt", "w") as f:
        f.write(report_text)

    with open("proctoring_report.txt", "rb") as f:
        st.download_button("📥 Download Report", f, file_name="Proctoring_Report.txt")

if __name__ == "__main__" or __name__ == "__streamlit__":
    run_test()
