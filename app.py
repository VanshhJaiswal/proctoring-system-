import streamlit as st
import openai
import os
import json
from streamlit_autorefresh import st_autorefresh

# Load API key from environment
openai.api_key = os.getenv("GROQ_API_KEY")
MODEL_NAME = "mixtral-8x7b-32768"

st.set_page_config(page_title="AI Proctored Quiz", layout="wide")

# Inject JS to track tab switches
tab_switch_script = """
<script>
let tabSwitches = 0;
document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
        tabSwitches += 1;
        const countEl = window.parent.document.getElementById("switch-count");
        if (countEl) countEl.innerText = tabSwitches;
    }
});
</script>
"""

# Initialize session states
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
if 'monitor_permission' not in st.session_state:
    st.session_state.monitor_permission = False

# Inject JS + hidden DOM elements
st.components.v1.html(
    tab_switch_script + """
    <div id='switch-count' style='display:none;'>0</div>
    <div id='switch-count-value' style='display:none;'></div>
    <script>
    setInterval(() => {
        const val = window.parent.document.getElementById('switch-count')?.innerText || "0";
        document.getElementById('switch-count-value').innerText = val;
    }, 500);
    </script>
    """,
    height=0
)

# Auto refresh every 1 second to fetch tab count
count = st_autorefresh(interval=1000, limit=None, key="refresh")

# Read tab count from hidden div
switch_html = st.components.v1.html("""
<div id="reader" style="display:none;">
<script>
const val = window.parent.document.getElementById('switch-count-value')?.innerText || "0";
document.getElementById("reader").innerText = val;
</script>
</div>
""", height=0)

# Update session state
try:
    st.session_state.tab_switches = int(switch_html or "0")
except:
    st.session_state.tab_switches = 0

# Display tab switch count
tab_switch_placeholder = st.empty()
tab_switch_placeholder.markdown(f"🔁 **Tab Switches: {st.session_state.tab_switches}**")

# Function to generate MCQs
def generate_mcqs(topic, num_questions):
    prompt = f"""
    Generate exactly {num_questions} multiple choice questions on {topic}.
    Format STRICTLY as a JSON array like:
    [
        {{
            "question": "What is Python?",
            "options": ["Language", "Snake", "Tool", "IDE"],
            "answer": "Language"
        }},
        ...
    ]
    Return only valid JSON array. No explanation, no notes.
    """
    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response['choices'][0]['message']['content']
        data = json.loads(content)
        if isinstance(data, list) and len(data) == num_questions:
            return data
        else:
            st.warning(f"Received {len(data)} questions instead of {num_questions}. Showing fallback.")
            return sample_questions(num_questions)
    except Exception as e:
        st.error("❌ Could not fetch questions. Using fallback.")
        return sample_questions(num_questions)

def sample_questions(n):
    return [{
        "question": "What does CPU stand for?",
        "options": ["Central Processing Unit", "Computer Personal Unit", "Central Performance Unit", "Core Processing Utility"],
        "answer": "Central Processing Unit"
    }] * n

# Quiz setup
if not st.session_state.quiz_started:
    st.title("🧠 AI Proctored Quiz")
    with st.form("quiz_form"):
        topic = st.text_input("📘 Topic", value="Python")
        num_qs = st.slider("❓ Number of Questions", 1, 10, 3)
        duration = st.slider("⏱️ Duration (minutes)", 1, 30, 5)
        allow = st.checkbox("✅ I allow tab monitoring for proctoring (Required)")
        start = st.form_submit_button("🚀 Start Quiz")

        if start:
            if not allow:
                st.error("❌ Permission required to proceed.")
            else:
                st.session_state.monitor_permission = True
                st.session_state.questions = generate_mcqs(topic, num_qs)
                st.session_state.quiz_started = True
                st.session_state.submitted = False
                st.session_state.answers = {}
                st.session_state.tab_switches = 0

# Quiz in progress
if st.session_state.quiz_started and not st.session_state.submitted:
    st.header("📝 Quiz In Progress")
    if st.session_state.monitor_permission:
        st.markdown("⚠️ Tab switching is being monitored. Please stay focused on this tab!")
    else:
        st.error("⚠️ Monitoring permission not granted.")

    answers = {}
    with st.form("mcq_form"):
        for idx, q in enumerate(st.session_state.questions):
            st.write(f"**Q{idx+1}:** {q['question']}")
            answers[idx] = st.radio("Options", q["options"], key=f"q_{idx}")
        submit = st.form_submit_button("✅ Submit")
        if submit:
            st.session_state.answers = answers
            st.session_state.submitted = True

# Results
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
    st.warning(f"🔁 You switched tabs **{st.session_state.tab_switches}** times during the quiz.")
    if st.session_state.tab_switches > 0:
        st.warning("📉 Tab switching may affect evaluation credibility.")
    else:
        st.success("✅ No tab switches detected. Great focus!")

