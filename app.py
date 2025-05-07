import streamlit as st
import openai
import os
import json
import time

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
        window.parent.postMessage({ type: 'TAB_SWITCH', count: tabSwitches }, "*");
    }
});
</script>
"""

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
if 'monitor_permission' not in st.session_state:
    st.session_state.monitor_permission = False

# Inject JS + hidden counter
st.components.v1.html(
    tab_switch_script + """
    <div id='switch-count' style='display:none;'>0</div>
    """,
    height=0
)

# Helper: read tab count from frontend (JS → DOM → Streamlit)
def read_tab_switch_count():
    count_html = st.components.v1.html("""
    <script>
    const val = window.parent.document.getElementById('switch-count')?.innerText || "0";
    document.body.innerText = val;
    </script>
    """, height=0)
    try:
        count = int(count_html)
    except:
        count = st.session_state.tab_switches
    return count

# Update tab switch count every refresh
st.session_state.tab_switches = read_tab_switch_count()

tab_switch_placeholder = st.empty()
tab_switch_placeholder.markdown(f"🔁 **Tab Switches: {st.session_state.tab_switches}**")

# Generate MCQs
def generate_mcqs(topic, num_questions):
    prompt = f"""
    Generate {num_questions} unique multiple choice questions on {topic}.
    Ensure that each question is different.
    Format STRICTLY as a JSON array like:
    [
        {{
            "question": "What is Python?",
            "options": ["Language", "Snake", "Tool", "IDE"],
            "answer": "Language"
        }},
        ...
    ]
    Only output valid JSON array. No explanation, no notes.
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
            st.warning(f"⚠️ Received {len(data)} questions instead of {num_questions}. Showing fallback.")
            return sample_questions(num_questions)
    except Exception as e:
        st.error(f"❌ Could not fetch questions: {e}. Using fallback.")
        return sample_questions(num_questions)

def sample_questions(n):
    return [
        {
            "question": f"What does CPU stand for? (variant {i+1})",
            "options": ["Central Processing Unit", "Computer Personal Unit", "Central Performance Unit", "Core Processing Utility"],
            "answer": "Central Processing Unit"
        }
        for i in range(n)
    ]

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
                st.experimental_rerun()  # refresh page after start

# Quiz in progress
if st.session_state.quiz_started and not st.session_state.submitted:
    st.header("📝 Quiz In Progress")
    if st.session_state.monitor_permission:
        st.markdown("⚠️ Tab switching is being monitored. Please stay focused!")
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
            st.experimental_rerun()  # refresh to results

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

