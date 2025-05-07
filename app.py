import cv2
import streamlit as st
import mediapipe as mp
import numpy as np
import requests
from transformers import pipeline
import tempfile
import time
import os

# Load Hugging Face QA model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Load environment variable or set manually here
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_key_here")  # Replace with your actual key or use .env

st.set_page_config(layout="wide")
st.title("🧠 AI Proctored Test System")

# ---- Groq API: Generate Questions ----
def generate_mock_questions():
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Create 5 general knowledge multiple choice questions with 4 options and indicate the correct answer."}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        result = response.json()
        st.write("Mock Questions JSON Response:", result)  # Debug only

        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            return "Error: No content received from Groq API."

    except Exception as e:
        return f"Groq API call failed: {str(e)}"

# ---- Mouth Open Detection ----
def detect_mouth_open(image, landmarks):
    top_lip = landmarks.landmark[13]
    bottom_lip = landmarks.landmark[14]
    lip_distance = abs(top_lip.y - bottom_lip.y)
    return lip_distance > 0.03

# ---- Head Pose Estimation ----
def estimate_head_pose(image, landmarks):
    nose_tip = landmarks.landmark[1]
    if nose_tip.x < 0.3:
        return "Looking Right"
    elif nose_tip.x > 0.7:
        return "Looking Left"
    else:
        return "Looking Center"

# ---- Proctoring: Face, Mouth, Pose, Phone ----
def run_proctoring():
    stframe = st.empty()
    mp_face = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    face_mesh = mp_face.FaceMesh()

    cap = cv2.VideoCapture(0)

    violations = []

    start_time = time.time()
    while time.time() - start_time < 30:  # 30 seconds test
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_face.FACEMESH_CONTOURS)

                if detect_mouth_open(frame, landmarks):
                    violations.append("Mouth Open")

                head_pose = estimate_head_pose(frame, landmarks)
                if head_pose != "Looking Center":
                    violations.append(f"Head Pose: {head_pose}")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        phone_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mobilephone.xml')
        phones = phone_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in phones:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            violations.append("Phone Detected")

        frame = cv2.resize(frame, (640, 480))
        stframe.image(frame, channels="BGR")

    cap.release()
    return list(set(violations))

# ---- Run the App ----
if st.button("🧪 Start Proctored Test"):
    with st.spinner("Proctoring in progress..."):
        violations = run_proctoring()

    st.success("✅ Proctoring complete.")
    st.write("🔍 Violations detected:", violations)

    st.markdown("---")
    st.subheader("📄 AI-Generated Mock Questions")

    questions_text = generate_mock_questions()
    st.text_area("📝 Questions", questions_text, height=250)

    st.markdown("---")
    st.subheader("✅ Answer Any Question")

    context = st.text_area("Paste the full question text here:")
    question = st.text_input("Type your answer question here (e.g., 'What is the capital of India?')")

    if st.button("Check Answer"):
        try:
            result = qa_pipeline(question=question, context=context)
            st.write("Answer:", result['answer'])
        except Exception as e:
            st.error(f"Error: {e}")

    # Download Result
    st.markdown("---")
    st.subheader("📥 Download Test Report")
    if st.button("📤 Download Result as .txt"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
            tmp.write(f"Violations Detected:\n{violations}\n\nMock Questions:\n{questions_text}".encode())
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            st.download_button("Download Result", f, file_name="proctored_test_result.txt")

