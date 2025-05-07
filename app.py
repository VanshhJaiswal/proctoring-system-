import streamlit as st
import openai
import os
import json

# Load API key
openai.api_key = os.getenv("GROQ_API_KEY")
MODEL_NAME = "mixtral-8x7b-32768"

st.set_page_config(page_title="AI Proctored Quiz", layout="wide")

# JavaScript to track tab switching
tab_switch_js = """
<script>
let tabSwitchCount = 0;
document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
        tabSwitchCount += 1;
        const el = document.getElementById("tab-switches");
        if (el) {
            el.innerText = tabSwitchCount;
        }
        window.parent.postMessage({tabSwitch: tabSwitchCount}, "*");
    }
});
</script>
"""

def inject_tab_switch_js():
    st.markdown(tab_switch_js, unsafe_allow_html=True)
    st.markdown("🔁 Tab Switches: **`<span id='tab-switches'>0</span>`**", unsafe_allow_html=True)

# Function to get MCQs from Groq API
def generate_mcqs(topic, num_questions):
    prompt = f"""
    Generate {num_questions} multiple choice questions on {topic}.
    Format: JSON list like this:
    [
        {{
            "question": "What is Python?",
            "options": ["Language", "Snake", "Tool", "IDE"],
            "answer": "Language"
        }},
        ...
    ]
    """
    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response['choices'][0]['message']['content']
        mcqs = json.loads(content)
        return mcqs
    except Exception as e:
        st.error("❌ Failed to generate questions. Check API key or formatting.")
        st.stop()

# Initialize session state
if 'quiz_started' not in st.session_state:
    st.session_state.quiz_started = False
if 'submitted' not in st.session_state:
    st.session_state.submitted = False
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'answers' not in st.session_state:
    st.session_state.answers = {}
if 'tab_switches' not in st.session_state:
    st.session_state.tab_switches = 0

# Quiz Setup UI
if not st.session_state.quiz_started:
    st.title("🧠 AI Proctored Quiz App")
    with st.form("quiz_setup"):
        topic = st.text_input("📘 Enter Quiz Topic", value="Python")
        num_qs = st.slider("❓ Number of Questions", 1, 10, 5)
        duration = st.slider("⏳ Duration (minutes)", 1, 30, 5)
        agree = st.checkbox("✅ I allow tab monitoring during this quiz.")
        start = st.form_submit_button("🚀 Start Quiz")

        if start:
            if not agree:
                st.error("You must allow tab monitoring to proceed.")
            else:
                st.session_state.questions = generate_mcqs(topic, num_qs)
                st.session_state.quiz_started = True
                st.session_state.tab_switches = 0
                st.query_params["tab_switches"] = "0"

# Active Quiz
if st.session_state.quiz_started and not st.session_state.submitted:
    st.title("📝 Quiz In Progress")
    inject_tab_switch_js()
    with st.form("quiz_form"):
        answers = {}
        for idx, q in enumerate(st.session_state.questions):
            st.subheader(f"Q{idx + 1}: {q['question']}")
            answers[idx] = st.radio("Options:", q["options"], key=f"q{idx}")
        submit = st.form_submit_button("✅ Submit Quiz")
        if submit:
            st.session_state.answers = answers
            st.session_state.submitted = True

# Result and Tab Switch Report
if st.session_state.submitted:
    st.title("📊 Quiz Results")
    score = 0
    for i, q in enumerate(st.session_state.questions):
        user_ans = st.session_state.answers.get(i, "Not answered")
        correct = q["answer"]
        st.write(f"**Q{i+1}: {q['question']}**")
        st.write(f"👉 Your Answer: `{user_ans}`")
        st.write(f"✅ Correct Answer: `{correct}`")
        st.markdown("---")
        if user_ans == correct:
            score += 1

    st.success(f"🎯 Your Score: {score}/{len(st.session_state.questions)}")
    st.warning("⚠️ Tab switch detection is shown above. Excessive tab switches may affect credibility.")
