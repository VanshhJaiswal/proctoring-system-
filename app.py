import streamlit as st
import openai
import os
import random
import time
import json
import cv2
import threading

# Load Groq API key from environment
openai.api_key = os.getenv("GROQ_API_KEY")
MODEL_NAME = "mixtral-8x7b-32768"

st.set_page_config(page_title="AI Proctored Quiz App", layout="wide")

# ---- Webcam Access Placeholder ----
def start_webcam():
    cap = cv2.VideoCapture(0)
    while st.session_state.quiz_started and not st.session_state.submitted:
        ret, frame = cap.read()
        if ret:
            # Can process frame here (e.g., face detection)
            pass
        time.sleep(1)
    cap.release()

# ---- Tab Switch Detection ----
tab_switch_script = """
<script>
let count = 0;
document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
        count += 1;
        fetch(`/tab_switch?count=${count}`);
    }
});
</script>
"""

# ---- Streamlit Server Extension ----
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime.server import Server

def get_current_session():
    session_id = get_script_run_ctx().session_id
    server = Server.get_current()
    session_infos = server._session_info_by_id
    return session_infos[session_id].session

# ---- Inject Tab Switch Handler ----
def inject_tab_switch_counter():
    st.markdown(tab_switch_script, unsafe_allow_html=True)

# ---- Generate MCQs from Groq API ----
def generate_mcqs(quiz_type, num_questions):
    prompt = f"Generate {num_questions} multiple-choice questions (with 4 options each and correct answer marked) for a quiz on '{quiz_type}'. Respond in JSON list format with question, options, and answer."
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    content = response['choices'][0]['message']['content']
    try:
        questions = json.loads(content)
    except json.JSONDecodeError:
        st.error("Failed to parse question JSON from Groq response.")
        return []
    return questions

# ---- Session State Defaults ----
if 'quiz_started' not in st.session_state:
    st.session_state.quiz_started = False
if 'submitted' not in st.session_state:
    st.session_state.submitted = False
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'tab_switches' not in st.session_state:
    st.session_state.tab_switches = 0
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}

# ---- Title & Setup ----
st.title("🧠 AI Proctored Quiz App")

if not st.session_state.quiz_started:
    with st.form("setup_form"):
        quiz_type = st.text_input("📚 Quiz Topic (e.g., Python, History)")
        num_questions = st.slider("🔢 Number of Questions", 1, 10, 5)
        duration = st.slider("⏱️ Time Duration (minutes)", 1, 30, 5)
        consent = st.checkbox("✅ I allow webcam and tab monitoring during quiz")
        submitted = st.form_submit_button("Start Quiz")

        if submitted:
            if not consent:
                st.error("You must allow webcam and monitoring to start the quiz.")
            elif not quiz_type:
                st.warning("Please enter a quiz topic.")
            else:
                st.session_state.questions = generate_mcqs(quiz_type, num_questions)
                st.session_state.quiz_started = True
                st.session_state.submitted = False
                st.session_state.tab_switches = 0
                threading.Thread(target=start_webcam).start()
                st.success("Quiz started! Good luck!")

# ---- Quiz Active ----
if st.session_state.quiz_started and st.session_state.questions and not st.session_state.submitted:
    inject_tab_switch_counter()
    st.markdown(f"📵 **Tab Switch Count:** `{st.session_state.tab_switches}`")
    with st.form("quiz_form"):
        answers = {}
        for i, q in enumerate(st.session_state.questions):
            st.write(f"**Q{i+1}. {q['question']}**")
            selected = st.radio("Choose:", q["options"], key=f"q{i}")
            answers[i] = selected

        if st.form_submit_button("✅ Submit Quiz"):
            st.session_state.user_answers = answers
            st.session_state.submitted = True

# ---- Evaluation ----
if st.session_state.submitted:
    score = 0
    st.success("🎉 Quiz Submitted!")
    st.markdown("---")
    for i, q in enumerate(st.session_state.questions):
        user_ans = st.session_state.user_answers.get(i, "No answer")
        correct = q["answer"]
        st.write(f"**Q{i+1}:** {q['question']}")
        st.write(f"🔹 Your Answer: `{user_ans}`")
        st.write(f"✅ Correct Answer: `{correct}`")
        if user_ans == correct:
            score += 1
        st.markdown("---")
    st.info(f"📊 Final Score: **{score} / {len(st.session_state.questions)}**")
    st.warning(f"🕵️ Tabs switched during exam: **{st.session_state.tab_switches}**")

# ---- Simulated API to Count Tab Switch ----
from streamlit.web.server.websocket_headers import _get_websocket_headers
from fastapi import FastAPI, Request

app = FastAPI()

@app.get("/tab_switch")
async def handle_tab_switch(request: Request):
    session = get_current_session()
    session._state["tab_switches"] += 1
    return {"status": "ok"}
