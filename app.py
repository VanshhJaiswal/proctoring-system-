import streamlit as st
import cv2
import tempfile
import os
import time
import json
import random
import csv
import numpy as np
from datetime import datetime
from groq import Groq
from ultralytics import YOLO
import dlib

# Load models
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
model = YOLO("yolov8n.pt")  # For mobile detection

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(page_title="AI Proctoring System", layout="wide")
st.title("🧠 AI Proctoring System with Random Test and Alerts")

# Session state init
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "logs" not in st.session_state:
    st.session_state.logs = []
if "questions" not in st.session_state:
    st.session_state.questions = []
if "current_q" not in st.session_state:
    st.session_state.current_q = 0
if "score" not in st.session_state:
    st.session_state.score = 0

# Fallback questions
def sample_questions(n):
    return [{
        "question": f"Sample Question {i+1}?",
        "options": ["A", "B", "C", "D"],
        "answer": "A"
    } for i in range(n)]

# Groq question generator
def generate_mcqs(topic, num_questions):
    prompt = f"""
    Generate {num_questions} unique multiple choice questions on {topic}.
    Strictly output JSON array like:
    [
        {{
            "question": "What is Python?",
            "options": ["Language", "Snake", "Tool", "IDE"],
            "answer": "Language"
        }},
        ...
    ]
    Only output valid JSON array. No explanation.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content.strip()

        if not content.startswith("["):
            raise ValueError("Groq API did not return a JSON array.")

        data = json.loads(content)
        if isinstance(data, list) and len(data) == num_questions:
            return data
        else:
            st.warning(f"⚠️ Groq returned {len(data)} questions. Falling back to sample.")
            return sample_questions(num_questions)
    except Exception as e:
        st.error(f"❌ Could not fetch questions: {str(e)}. Using fallback.")
        return sample_questions(num_questions)

# Face absent detection
def detect_absent(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    return len(faces) == 0

# Gaze detection
def eye_gaze_direction(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        nose = landmarks.part(30)
        left_eye = landmarks.part(36)
        right_eye = landmarks.part(45)
        if nose.x < left_eye.x:
            return "Looking Right"
        elif nose.x > right_eye.x:
            return "Looking Left"
        else:
            return "Looking Center"
    return "No Face"

# Mobile detection
def detect_mobile(frame):
    results = model(frame, verbose=False)[0]
    for r in results.boxes.cls:
        if int(r) == 67:  # COCO class for cell phone
            return True
    return False

# Main webcam feed
def run_proctor():
    cap = cv2.VideoCapture(0)
    alert_placeholder = st.empty()
    frame_display = st.empty()
    start = time.time()
    absent_counter = 0
    while time.time() - start < duration * 60:
        ret, frame = cap.read()
        if not ret:
            break

        # Checks
        if detect_mobile(frame):
            alert_placeholder.error("📱 Mobile phone detected!")
            st.session_state.logs.append((datetime.now(), "Mobile detected"))
        elif detect_absent(frame):
            absent_counter += 1
            if absent_counter >= 5:
                alert_placeholder.error("🚫 Candidate absent for >5s!")
                st.session_state.logs.append((datetime.now(), "Absent"))
        else:
            direction = eye_gaze_direction(frame)
            if direction != "Looking Center":
                alert_placeholder.warning(f"👀 {direction}")
                st.session_state.logs.append((datetime.now(), direction))
            else:
                alert_placeholder.empty()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_display.image(frame, channels="RGB")

    cap.release()
    frame_display.empty()
    alert_placeholder.success("✅ Proctoring Complete")

# Random test
with st.sidebar:
    st.header("📝 Test Settings")
    topic = st.text_input("Test Topic", "Python")
    num_q = st.slider("No. of Questions", 5, 25, 10)
    duration = st.slider("Test Duration (mins)", 1, 15, 5)
    if st.button("Start Proctoring + Test"):
        st.session_state.questions = generate_mcqs(topic, num_q)
        st.session_state.start_time = time.time()
        st.session_state.current_q = 0
        st.session_state.score = 0
        run_proctor()

# Test UI
if st.session_state.start_time:
    if st.session_state.current_q < len(st.session_state.questions):
        q = st.session_state.questions[st.session_state.current_q]
        st.subheader(f"Q{st.session_state.current_q+1}: {q['question']}")
        choice = st.radio("Options", q['options'], key=f"q{st.session_state.current_q}")
        if st.button("Next"):
            if choice == q['answer']:
                st.session_state.score += 1
            st.session_state.current_q += 1
    else:
        st.success(f"🎉 Test Completed! Score: {st.session_state.score}/{len(st.session_state.questions)}")

        # Download logs
        csv_data = "timestamp,event\n" + "\n".join([f"{ts},{ev}" for ts, ev in st.session_state.logs])
        st.download_button("Download Logs CSV", csv_data, "logs.csv", "text/csv")
