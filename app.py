import streamlit as st
import random
import os
import requests
from dotenv import load_dotenv

# Load env vars
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Page config
st.set_page_config(page_title="AI Proctored Quiz", layout="centered")

# Initialize session state
if "quiz_started" not in st.session_state:
    st.session_state.quiz_started = False
if "questions" not in st.session_state:
    st.session_state.questions = []
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}
if "tab_switches" not in st.session_state:
    st.session_state.tab_switches = 0

# JavaScript for tab-switch detection
st.markdown("""
<script>
let switchCount = 0;

document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        switchCount += 1;
        fetch("/tab_switch_detected?count=" + switchCount);
    }
});
</script>
""", unsafe_allow_html=True)

# Webcam access request
st.markdown("""
<script>
navigator.mediaDevices.getUserMedia({ video: true })
  .then(function(stream) {
    // Permission granted
  })
  .catch(function(err) {
    alert("📸 Webcam access is required for this proctored test.");
  });
</script>
""", unsafe_allow_html=True)

st.title("🎓 AI Proctored Quiz App")
st.markdown("This quiz will monitor tab-switching and require webcam access.")

# Selection
quiz_topic = st.selectbox("🧠 Choose Quiz Topic", ["Python", "Machine Learning", "HTML", "General Knowledge"])
num_questions = st.slider("🔢 Number of Questions", 3, 10, 5)
duration = st.slider("⏰ Duration (Minutes)", 1, 10, 3)

# Start Quiz
if st.button("🚀 Start Quiz"):
    st.session_state.quiz_started = True
    st.session_state.tab_switches = 0
    st.session_state.user_answers = {}
    st.session_state.submitted = False

    with st.spinner("Generating quiz using Groq..."):
        prompt = f"Generate {num_questions} multiple choice questions on {quiz_topic}. Format: Q: Question\\nA. Option1\\nB. Option2\\nC. Option3\\nD. Option4\\nAnswer: A/B/C/D"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama3-70b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        content = res.json()["choices"][0]["message"]["content"]

        raw_blocks = content.split("Q:")
        questions = []

        for block in raw_blocks[1:]:
            lines = block.strip().split("\n")
            q_text = lines[0].strip()
            options = [line[2:].strip() for line in lines[1:5]]
            correct_line = [l for l in lines if l.startswith("Answer:")]
            correct_answer = correct_line[0].split(":")[1].strip() if correct_line else "A"
            questions.append({
                "question": q_text,
                "options": options,
                "correct": correct_answer
            })

        st.session_state.questions = questions

# Display questions
if st.session_state.quiz_started and st.session_state.questions and not st.session_state.submitted:
    st.markdown(f"📵 **Tab Switch Count:** `{st.session_state.tab_switches}`")
    with st.form("quiz_form"):
        for i, q in enumerate(st.session_state.questions):
            st.write(f"**Q{i+1}. {q['question']}**")
            st.session_state.user_answers[i] = st.radio("Choose:", q["options"], key=f"q{i}")
        if st.form_submit_button("✅ Submit Quiz"):
            st.session_state.submitted = True

# Display results
if st.session_state.submitted:
    score = 0
    st.subheader("📊 Results")
    for i, q in enumerate(st.session_state.questions):
        user_ans = st.session_state.user_answers[i]
        correct_index = ord(q["correct"]) - ord("A")
        correct_ans = q["options"][correct_index]

        is_correct = user_ans == correct_ans
        if is_correct:
            score += 1

        st.markdown(f"**Q{i+1}: {q['question']}**")
        st.markdown(f"- ✅ Correct: **{correct_ans}**")
        st.markdown(f"- 🧑 Your Answer: {'✅' if is_correct else '❌'} **{user_ans}**")

    st.success(f"🎯 Your Score: **{score}/{len(st.session_state.questions)}**")
    st.warning(f"🧩 Tab Switches Detected: **{st.session_state.tab_switches}**")
    if st.session_state.tab_switches > 2:
        st.error("⚠️ Your session is flagged due to excessive tab switching!")

# Capture tab switch via Streamlit route (simulated here)
# NOTE: This will not work unless you implement a backend route. For local demo, simulate tab switch manually:
st.markdown("---")
if st.button("🔄 Simulate Tab Switch"):
    st.session_state.tab_switches += 1
