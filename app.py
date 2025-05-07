import streamlit as st
import random
import time
import os
import requests

# ---------- ENVIRONMENT VARIABLES ----------
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------- INITIALIZATION ----------
st.set_page_config(page_title="AI Proctored Quiz", layout="centered")
if 'quiz_started' not in st.session_state:
    st.session_state.quiz_started = False
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
if 'selected_answers' not in st.session_state:
    st.session_state.selected_answers = []
if 'tab_switch_count' not in st.session_state:
    st.session_state.tab_switch_count = 0

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
    const currentCount = parseInt(localStorage.getItem("tabSwitchCount") || "0");
    localStorage.setItem("tabSwitchCount", currentCount + 1);
    const streamlitEvent = new Event("tabSwitchDetected");
    window.dispatchEvent(streamlitEvent);
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

# ---------- QUIZ CONFIG ----------
st.title("🧠 AI Proctored Quiz App")
st.markdown("Secure, smart, and monitored quiz system using Groq LLaMA3 and Streamlit.")

quiz_type = st.selectbox("📚 Select Quiz Type", ["Python", "Machine Learning", "HTML", "General Knowledge"])
num_questions = st.slider("❓ Number of Questions", 3, 10, 5)
duration_minutes = st.slider("⏰ Time Duration (in minutes)", 1, 10, 3)

if st.button("✅ Start Quiz"):
    st.session_state.quiz_started = True
    st.session_state.tab_switch_count = 0
    st.session_state.score = 0
    st.session_state.current_question = 0
    st.session_state.selected_answers = []

    with st.spinner("🧠 Generating questions using LLaMA3..."):
        prompt = f"Generate {num_questions} multiple-choice questions on {quiz_type}. Format: Question, 4 options, and correct answer."
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

        # Simple parser (assuming format is Question\nA.\nB.\nC.\nD.\nAnswer: X)
        questions = []
        blocks = res_text.split("Answer:")
        for block in blocks[:-1]:
            lines = block.strip().split("\n")
            if len(lines) >= 5:
                q = lines[0].strip()
                options = [opt.strip("ABCD. ") for opt in lines[1:5]]
                correct = blocks[blocks.index(block)+1].strip().split("\n")[0]
                questions.append({"question": q, "options": options, "answer": correct.upper()})
        st.session_state.questions = questions[:num_questions]
        st.success("Questions loaded. Quiz will begin below.")

# ---------- QUIZ TIMER + QUESTIONS ----------
if st.session_state.quiz_started:
    st.warning(f"🚨 Tab Switches Detected: {st.session_state.tab_switch_count}")
    total_seconds = duration_minutes * 60
    start_time = time.time()

    while st.session_state.current_question < len(st.session_state.questions):
        elapsed = int(time.time() - start_time)
        remaining = total_seconds - elapsed
        if remaining <= 0:
            st.error("⏰ Time's up!")
            break

        mins, secs = divmod(remaining, 60)
        st.info(f"⏱️ Time Left: {mins:02}:{secs:02}")

        q_data = st.session_state.questions[st.session_state.current_question]
        st.subheader(f"Q{st.session_state.current_question + 1}: {q_data['question']}")
        answer = st.radio("Choose your answer:", q_data["options"], key=st.session_state.current_question)

        if st.button("Next Question"):
            st.session_state.selected_answers.append(answer)
            if answer.upper() == q_data["answer"]:
                st.session_state.score += 1
            st.session_state.current_question += 1
            st.rerun()

    if st.session_state.current_question == len(st.session_state.questions):
        st.success("🎉 Quiz Completed!")
        st.markdown(f"📝 Score: **{st.session_state.score}/{num_questions}**")
        st.markdown(f"🚨 Tab Switches during quiz: **{st.session_state.tab_switch_count}**")
        if st.session_state.tab_switch_count > 2:
            st.error("⚠️ Multiple tab switches detected. This quiz attempt may be flagged.")

