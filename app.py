import streamlit as st
import openai
import os
import json

# Load API key from environment
openai.api_key = os.getenv("GROQ_API_KEY")
MODEL_NAME = "mixtral-8x7b-32768"

st.set_page_config(page_title="AI Proctored Quiz", layout="wide")

# JS for tab switch detection (injects a counter into the DOM)
tab_switch_script = """
<script>
let tabSwitches = 0;
document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
        tabSwitches += 1;
        const countEl = window.parent.document.getElementById("switch-count");
        if (countEl) countEl.innerText = tabSwitches;
        window.parent.postMessage({ type: 'TAB_SWITCH', count: tabSwitches }, "*");
    }
});
</script>
"""

# State initialization
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

# Inject JS (this runs on every rerun)
st.components.v1.html(tab_switch_script + "<div id='switch-count' style='display:none;'>0</div>", height=0)

# JavaScript message listener to increment tab switch
tab_switch_placeholder = st.empty()
tab_switch_placeholder.markdown("🔁 **Tab Switches: 0**")

def set_tab_switch_count_js():
    st.components.v1.html("""
    <script>
    window.addEventListener("message", (event) => {
        if (event.data.type === "TAB_SWITCH") {
            const count = event.data.count;
            const el = window.parent.document.querySelector('section.main div.block-container div:nth-child(1)');
            if (el) el.innerHTML = "🔁 <b>Tab Switches:</b> " + count;
        }
    });
    </script>
    """, height=0)

set_tab_switch_count_js()

# MCQ generator from Groq
def generate_mcqs(topic, num_questions):
    prompt = f"""
    Generate {num_questions} multiple choice questions on {topic}.
    Format strictly as a valid JSON array like:
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
        return json.loads(content)
    except Exception as e:
        st.error("❌ Could not fetch questions. Using sample fallback.")
        return [{
            "question": "What does CPU stand for?",
            "options": ["Central Processing Unit", "Computer Personal Unit", "Central Performance Unit", "Core Processing Utility"],
            "answer": "Central Processing Unit"
        }]

# Quiz setup form
if not st.session_state.quiz_started:
    st.title("🧠 AI Proctored Quiz")
    with st.form("quiz_form"):
        topic = st.text_input("📘 Topic", value="Python")
        num_qs = st.slider("❓ Number of Questions", 1, 10, 3)
        duration = st.slider("⏱️ Duration (minutes)", 1, 30, 5)
        allow = st.checkbox("✅ I allow tab monitoring for proctoring.")
        start = st.form_submit_button("🚀 Start Quiz")

        if start:
            if not allow:
                st.error("Permission required to proceed.")
            else:
                st.session_state.questions = generate_mcqs(topic, num_qs)
                st.session_state.quiz_started = True
                st.session_state.submitted = False
                st.session_state.answers = {}
                st.session_state.tab_switches = 0

# Show questions
if st.session_state.quiz_started and not st.session_state.submitted:
    st.header("📝 Quiz In Progress")
    st.markdown("⚠️ Tab switching will be monitored and recorded.")
    answers = {}

    with st.form("mcq_form"):
        for idx, q in enumerate(st.session_state.questions):
            st.write(f"**Q{idx+1}:** {q['question']}")
            answers[idx] = st.radio("Options", q["options"], key=f"q_{idx}")
        submit = st.form_submit_button("✅ Submit")

        if submit:
            st.session_state.answers = answers
            st.session_state.submitted = True

# Show results
if st.session_state.submitted:
    st.header("📊 Quiz Results")
    correct = 0
    for i, q in enumerate(st.session_state.questions):
        user_ans = st.session_state.answers.get(i)
        correct_ans = q['answer']
        st.write(f"**Q{i+1}:** {q['question']}")
        st.write(f"✅ Correct: `{correct_ans}`")
        st.write(f"🧑 Your Answer: `{user_ans}`")
        if user_ans == correct_ans:
            correct += 1
        st.markdown("---")

    st.success(f"🎯 Final Score: {correct}/{len(st.session_state.questions)}")
    st.warning("📉 Tab switches may affect evaluation credibility.")
