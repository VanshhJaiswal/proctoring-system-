import streamlit as st
import os
import cv2
import time
import base64
import json
import requests
import tempfile
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="AI Proctoring System", layout="wide")
st.title("📹 Smart AI Proctoring System")

# Functions

def capture_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    return None

def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode()

def detect_faces_google_vision(base64_img):
    url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
    headers = {"Content-Type": "application/json"}
    data = {
        "requests": [
            {
                "image": {"content": base64_img},
                "features": [
                    {"type": "FACE_DETECTION"},
                    {"type": "OBJECT_LOCALIZATION"}
                ]
            }
        ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()

def analyze_google_response(response):
    faces = response['responses'][0].get('faceAnnotations', [])
    objects = response['responses'][0].get('localizedObjectAnnotations', [])
    mobile_detected = any("phone" in obj['name'].lower() for obj in objects)
    return len(faces), mobile_detected

def trigger_random_quiz(num_questions=5, duration_minutes=5):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"Generate a {num_questions}-question multiple choice quiz on general knowledge. Each question should have 4 options and the correct answer indicated."
    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
    return response.json()['choices'][0]['message']['content']

def log_alert(event, logs):
    logs.append({"timestamp": datetime.now().isoformat(), "event": event})

def download_logs(logs):
    df = json.dumps(logs, indent=4)
    with open("proctoring_log.json", "w") as f:
        f.write(df)
    return "proctoring_log.json"

# Main
if "logs" not in st.session_state:
    st.session_state.logs = []

interval = st.slider("Quiz Interval (mins)", 1, 15, 5)
num_questions = st.slider("Number of MCQs", 5, 25, 10)
start_btn = st.button("Start Proctoring")

if start_btn:
    st.info("Proctoring started. Monitoring via webcam...")
    quiz_trigger_time = time.time() + interval * 60

    while True:
        frame = capture_frame()
        if frame is None:
            st.error("Camera not accessible")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(rgb_frame, channels="RGB")

        base64_img = image_to_base64(frame)
        response = detect_faces_google_vision(base64_img)
        num_faces, mobile_detected = analyze_google_response(response)

        if num_faces == 0:
            st.warning("❌ Candidate absent")
            log_alert("Candidate absent", st.session_state.logs)
        elif num_faces > 1:
            st.warning("⚠️ Multiple faces detected")
            log_alert("Multiple faces detected", st.session_state.logs)

        if mobile_detected:
            st.warning("📱 Mobile phone detected!")
            log_alert("Mobile phone detected", st.session_state.logs)

        if time.time() >= quiz_trigger_time:
            quiz = trigger_random_quiz(num_questions=num_questions)
            st.info("📝 Random Quiz:")
            st.markdown(quiz)
            quiz_trigger_time = time.time() + interval * 60

        if st.button("Stop Proctoring"):
            st.success("Proctoring session ended.")
            log_file = download_logs(st.session_state.logs)
            with open(log_file, "rb") as f:
                st.download_button("📥 Download Logs", f, file_name="proctoring_logs.json")
            break

        time.sleep(5)
