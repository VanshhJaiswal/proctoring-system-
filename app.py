
import streamlit as st
import cv2
import numpy as np
import time
import base64
import os
import requests
from PIL import Image
from io import BytesIO

# Load GROQ API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Title
st.set_page_config(page_title="AI Proctoring + Quiz", layout="centered")
st.title("🎓 Smart AI Proctored Quiz")

# Session state init
if "quiz_started" not in st.session_state:
    st.session_state.quiz_started = False
if "questions" not in st.session_state:
    st.session_state.questions = []
if "answers" not in st.session_state:
    st.session_state.answers = []
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "proctoring_logs" not in st.session_state:
    st.session_state.proctoring_logs = []

# Utility: Quiz generation
def generate_quiz():
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mixtral-8x7b-32768",
        "messages": [
            {"role": "system", "content": "Generate a 5-question multiple-choice quiz on general knowledge. Each question should have 4 options labeled A-D, and provide the correct answer as 'Answer: A/B/C/D'."}
        ]
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)

    if response.status_code != 200:
        st.error(f"Quiz generation failed. Status code: {response.status_code}")
        st.text(response.text)
        return []

    try:
        content = response.json()["choices"][0]["message"]["content"]
        questions = []
        for q in content.strip().split("\n\n"):
            lines = q.strip().split("\n")
            if len(lines) >= 5:
                question_text = lines[0]
                options = lines[1:5]
                answer_line = lines[5] if len(lines) > 5 else ""
                correct_answer = answer_line.split(":")[-1].strip() if ":" in answer_line else ""
                questions.append({
                    "question": question_text,
                    "options": options,
                    "answer": correct_answer
                })
        return questions
    except Exception as e:
        st.error(f"Parsing quiz content failed: {e}")
        return []

# Utility: Detect faces
def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(faces), faces

# Step 1: Camera snapshot
if st.session_state.quiz_started:
    st.subheader("👁️ Proctoring in Progress: Stay Focused")
    uploaded_image = st.camera_input("📸 Snapshot (Required)")
    if uploaded_image:
        image = Image.open(uploaded_image)
        img_array = np.array(image)
        face_count, faces = detect_face(img_array)
        log = {
            "timestamp": time.time(),
            "face_count": face_count,
        }
        if face_count == 0:
            st.warning("⚠️ No face detected!")
        elif face_count > 1:
            st.warning("⚠️ Multiple faces detected!")
        st.session_state.proctoring_logs.append(log)

# Step 2: Quiz UI
if st.session_state.quiz_started and st.session_state.questions:
    st.subheader("📝 Quiz Time!")
    for i, q in enumerate(st.session_state.questions):
        st.write(f"**Q{i+1}. {q['question']}**")
        for opt in q["options"]:
            st.radio("", options=q["options"], key=f"q_{i}")

    if st.button("🧾 Submit Quiz"):
        st.session_state.quiz_started = False
        st.session_state.end_time = time.time()

        # Evaluate answers
        score = 0
        results = []
        for i, q in enumerate(st.session_state.questions):
            selected = st.session_state.get(f"q_{i}")
            correct = q["answer"]
            is_correct = selected.startswith(correct)
            results.append((q["question"], selected, correct, is_correct))
            if is_correct:
                score += 1

        duration = round(st.session_state.end_time - st.session_state.start_time, 2)
        st.success(f"✅ Quiz Completed! Score: {score}/5 | Time: {duration}s")

        st.subheader("📋 Result Summary:")
        for q_text, sel, ans, correct in results:
            st.write(f"- **{q_text}**")
            st.write(f"  - Your Answer: {sel}")
            st.write(f"  - Correct Answer: {ans}")
            st.write(f"  - {'✅ Correct' if correct else '❌ Incorrect'}")

        st.subheader("🔒 Proctoring Report:")
        total_logs = len(st.session_state.proctoring_logs)
        no_face = sum(1 for log in st.session_state.proctoring_logs if log["face_count"] == 0)
        multi_face = sum(1 for log in st.session_state.proctoring_logs if log["face_count"] > 1)
        st.write(f"- Snapshots analyzed: {total_logs}")
        st.write(f"- No face detected: {no_face}")
        st.write(f"- Multiple faces detected: {multi_face}")

        st.session_state.questions = []
        st.session_state.proctoring_logs = []

# Step 3: Start Quiz
if not st.session_state.quiz_started:
    st.subheader("🚀 Launch Quiz")
    if st.button("Start Quiz"):
        questions = generate_quiz()
        if questions:
            st.session_state.questions = questions
            st.session_state.quiz_started = True
            st.session_state.start_time = time.time()
            st.experimental_rerun()
        else:
            st.error("Quiz could not be generated. Please check your API key or try again later.")
