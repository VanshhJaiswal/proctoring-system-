import streamlit as st
import random
import time
import os
import requests
from dotenv import load_dotenv

# ---------- ENV SETUP ----------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------- SESSION STATE INIT ----------
st.set_page_config(page_title="AI Proctored Quiz", layout="centered")

if "quiz_started" not in st.session_state:
    st.session_state.quiz_started = False
if "questions" not in st.session_state:
    st.session_state.questions = []
if "tab_switch_count" not in st.session_state:
    st.session_state.tab_switch_count = 0
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# ---------- JAVASCRIPT: Tab Switch Detection ----------
st.markdown("""
<script>
let hidden, visibilityChange; 
if (typeof document.hidden !== "undefined") {
  hidden = "hidden";
  visibilityChange = "visibilitychange";
}
document.addEventListener(visibilityChange, function() {
  if (document[hidden]) {
    fetch("/tab_switch", { method: "POST" });
  }
});
</script>
""", unsafe_allow_html=True)

# ---------- JAVASCRIPT: Webcam Access ----------
st.markdown("""
<h5>📸 Webcam Access Required for Proctoring</h5>
<script>
navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
    const video = document.createElement('video');
    video.srcObject = stream;
    video.play();
}).catch(err => {
    alert("Webcam permission is required to continue!");
});
</script>
""", unsafe_allow_html=True)

# ---------- FRONT PAGE ----------
st.title("🧠 AI Proctored Quiz App")
st.markdown("A smart, secure quiz system using Groq + Streamlit with tab-switch monitoring.")

quiz_type = st.selectbox("📚 Select Quiz Type", ["Python", "Machine Learning", "HTML", "General Knowledge"])
num_questions = st.slider("❓ Number of Questions", 3, 10, 5)
duration_minutes = st.slider("⏰ Time Duration (in minutes)", 1, 10, 3)

if st.button("✅ Start Quiz"):
    st.session_state.quiz_started = True
    st.session_state.tab_switch_count = 0
    st.session_state.questions = []
    st.session_state.submitted = False

    with st.spinner("Generating questions using LLaMA3..."):
        prompt = f"Generate {num_questions} multiple choice questions on {quiz_type}. Each question must have 4 options (A, B, C, D) and one correct answer. Format: Question, options, and Answer."
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama3-70b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        res_text = response.json()["choices"][0]["message"]["content"]

        blocks = res_text.split("Answer:")
        questions = []
        for block in blocks[:-1]:
            lines = block.strip().split("\n")
            if len(lines) >= 5:
                q = lines[0].strip()
                options = [opt.strip("ABCD. ") for opt in lines[1:5]]
                correct = blocks[blocks.index(block)+1].strip().split("\n")[0]
                questions.append({"question": q, "options": options, "answer": correct.upper()})
        st.session_state.questions = questions[:num_questions]
        st.success("Questions generated! Please answer below.")

# ---------- DISPLAY QUIZ ----------
if st.session_state.quiz_started and st.session_state.questions and not st.session_state.submitted:
    st.warning(f"🚨 Tab Switches Detected: {st.session_state.tab_switch_count}")
    answers = {}

    with st.form("quiz_form"):
        for idx, q in enumerate(st.session_state.questions):
            st.subheader(f"Q{idx+1}: {q['question']}")
            answers[idx] = st.radio("Choose one:", q['options'], key=f"q_{idx}")

        submitted = st.form_submit_button("🚀 Submit Quiz")
        if submitted:
            score = 0
            for i, q in enumerate(st.session_state.questions):
                if answers[i].strip().upper().startswith(q["answer"]):
                    score += 1
            st.session_state.submitted = True
            st.session_state.score = score

# ---------- DISPLAY RESULTS ----------
if st.session_state.submitted:
    st.success("✅ Quiz Submitted!")
    st.markdown(f"🎯 Your Score: **{st.session_state.score}/{len(st.session_state.questions)}**")
    st.markdown(f"📉 Tab Switches Detected: **{st.session_state.tab_switch_count}**")
    if st.session_state.tab_switch_count > 2:
        st.error("⚠️ Multiple tab switches detected. This attempt may be flagged.")

