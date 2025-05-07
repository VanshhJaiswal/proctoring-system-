import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import os
import requests
from PIL import Image
from transformers import pipeline

# Load Groq API Key from .env or Streamlit secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

# Initialize Mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Proctoring metrics
face_present_count = 0
mouth_open_count = 0
head_movement_count = 0
window_switch_count = 0
total_frames = 0

activity_log = []

def detect_features(image):
    global face_present_count, mouth_open_count, head_movement_count, total_frames
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = []
    total_frames += 1

    face_results = face_mesh.process(img_rgb)
    if face_results.multi_face_landmarks:
        detections.append("✅ Face detected")
        face_present_count += 1

        for landmarks in face_results.multi_face_landmarks:
            upper_lip = landmarks.landmark[13]
            lower_lip = landmarks.landmark[14]
            lip_distance = abs(upper_lip.y - lower_lip.y)
            if lip_distance > 0.03:
                detections.append("⚠️ Mouth open")
                mouth_open_count += 1

    else:
        detections.append("❌ Face not detected")

    pose_results = pose.process(img_rgb)
    if pose_results.pose_landmarks:
        nose_y = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y
        left_ear_y = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y
        if abs(nose_y - left_ear_y) > 0.05:
            detections.append("⚠️ Head tilted")
            head_movement_count += 1

    return detections

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
    global window_switch_count
    st.title("🎓 Smart Proctored Mock Test")
    st.write("Please allow camera access. The system will monitor face, mouth, head movements and window focus.")

    duration_option = st.selectbox("Select Test Duration", ["1 min", "3 min", "10 min", "20 min"])
    duration_seconds = {"1 min":60, "3 min":180, "10 min":600, "20 min":1200}[duration_option]

    if st.button("Start Test"):
        mock_questions = generate_mock_questions()
        if not mock_questions:
            return

        st.subheader("📄 AI-Generated Mock Test")
        st.markdown(mock_questions)

        stframe = st.empty()
        camera = cv2.VideoCapture(0)
        start_time = time.time()

        st.info("⚠️ Don't switch windows during the test! Window switch will be counted.")
        st.warning("⚠️ Keep face visible for proctoring.")

        # JavaScript to detect window blur
        window_event_script = """
            <script>
            let count = 0;
            window.onblur = () => { fetch('/?window_switch='+ (++count)); };
            </script>
        """
        st.components.v1.html(window_event_script)

        while time.time() - start_time < duration_seconds:
            ret, frame = camera.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)

            detections = detect_features(frame)
            activity_log.extend(detections)

            for idx, d in enumerate(detections):
                cv2.putText(frame, d, (10, 30 + idx*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            stframe.image(frame, channels="BGR")

            # Check query params for window switch
            query_params = st.experimental_get_query_params()
            if "window_switch" in query_params:
                window_switch_count = int(query_params["window_switch"][0])

        camera.release()
        st.success("✅ Test completed.")

        # Report generation
        st.subheader("📊 Proctoring Report")
        face_time_percent = (face_present_count / total_frames) * 100 if total_frames else 0

        report_lines = [
            f"👤 Face visible {face_time_percent:.2f}% of the time",
            f"🙃 Head movements detected: {head_movement_count}",
            f"😮 Mouth open instances: {mouth_open_count}",
            f"🖥️ Window switches detected: {window_switch_count}",
            "",
            "Activity Log:",
            *list(set(activity_log))
        ]
        for line in report_lines:
            st.write("- " + line)

        report_text = "\n".join(report_lines)
        with open("proctoring_report.txt", "w") as f:
            f.write(report_text)
        with open("proctoring_report.txt", "rb") as f:
            st.download_button("📥 Download Report", f, file_name="Proctoring_Report.txt")

if __name__ == "__main__" or __name__ == "__streamlit__":
    run_test()
