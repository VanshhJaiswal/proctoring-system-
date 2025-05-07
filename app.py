import streamlit as st
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import requests
import os
from transformers import pipeline
from dotenv import load_dotenv
from datetime import datetime

# Load Groq API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# NLP pipeline for Q&A (Hugging Face)
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

st.title("🛡️ AI Proctored Test System")

# Report holder
log_data = []

# MediaPipe initialization
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def detect_face_and_pose(image):
    result_text = []
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh, \
         mp_pose.Pose(static_image_mode=True) as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face = face_mesh.process(image_rgb)
        pose_result = pose.process(image_rgb)

        # Face detection
        if not face.multi_face_landmarks:
            result_text.append("No face detected")
        elif len(face.multi_face_landmarks) > 1:
            result_text.append("Multiple faces detected")
        else:
            result_text.append("One face detected")

        # Head pose (simple heuristic using eyes and nose)
        if face.multi_face_landmarks:
            landmarks = face.multi_face_landmarks[0].landmark
            nose = landmarks[1]
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            if abs(left_eye.y - right_eye.y) > 0.05:
                result_text.append("Head turned")
            else:
                result_text.append("Head frontal")

        # Mouth opening detection (distance between lips)
        if face.multi_face_landmarks:
            top_lip = face.multi_face_landmarks[0].landmark[13]
            bottom_lip = face.multi_face_landmarks[0].landmark[14]
            mouth_open = abs(top_lip.y - bottom_lip.y) > 0.03
            result_text.append("Mouth open" if mouth_open else "Mouth closed")

        # Phone detection (simplified, checks if hands are near face)
        if pose_result.pose_landmarks:
            landmarks = pose_result.pose_landmarks.landmark
            left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_dist = np.linalg.norm([left_hand.x - nose.x, left_hand.y - nose.y])
            right_dist = np.linalg.norm([right_hand.x - nose.x, right_hand.y - nose.y])
            if left_dist < 0.1 or right_dist < 0.1:
                result_text.append("Possible phone detected near face")
            else:
                result_text.append("No phone detected")

        return ", ".join(result_text)

# Mock test questions using Groq API
def generate_mock_questions():
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [{"role": "user", "content": "Generate 3 general knowledge MCQ questions with 4 options and one correct answer."}],
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    return result['choices'][0]['message']['content']

# Image frame capture
frame = st.camera_input("Proctoring: Show your face and surroundings")

if frame:
    img_array = np.frombuffer(frame.getvalue(), np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    result = detect_face_and_pose(image)
    st.warning(result)
    log_data.append({"timestamp": datetime.now(), "status": result})

# Start test
if st.button("📝 Start Mock Test"):
    questions_text = generate_mock_questions()
    st.session_state.questions = questions_text
    st.session_state.answers = {}
    st.success("Test started. Scroll to see questions.")

# Show questions
if "questions" in st.session_state:
    st.subheader("📚 Mock Test")
    for i, q in enumerate(st.session_state.questions.split("\n\n")):
        st.markdown(f"**Q{i+1}:** {q.splitlines()[0]}")
        options = [line for line in q.splitlines()[1:] if line.startswith(('A)', 'B)', 'C)', 'D)'))]
        answer = st.radio(f"Choose your answer for Q{i+1}", options, key=f"q{i}")
        st.session_state.answers[f"Q{i+1}"] = answer

    if st.button("✅ Submit Test"):
        for q_id, ans in st.session_state.answers.items():
            log_data.append({"timestamp": datetime.now(), "status": f"{q_id}: {ans}"})
        st.success("Test Submitted!")

        # Downloadable report
        df = pd.DataFrame(log_data)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Proctoring Report", csv, "proctor_report.csv", "text/csv")

