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

# Initialize states
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

# Inject JS
st.components.v1.html(tab_switch_script + "<div id='switch-count' style='display:none;'>0</div>", height=0)

# JavaScript message listener to capture tab switches
def set_tab_switch_count_js():
    st.components.v1.html("""
    <script>
    window.addEventListener("message", (event) => {
        if (event.data.type === "TAB_SWITCH") {
            const count = event.data.count;
            const el = window.parent.document.querySelector('section.main div.block-container div:nth-child(1)');
            if (el) el.innerHTML = "🔁 <b>Tab Switches:</b> " + count;
            window.parent.postMessage({ type: 'SAVE_TAB_SWITCH', count: count }, "*");
        }
    });
    </script>
    """, height=0)

set_tab_switch_count_js()

# Capture tab switches (listen for postMessage and store in session_state)
tab_switch_placeholder = st.empty()
tab_switch_placeholder.markdown(f"🔁 **Tab Switches: {st.session_state.tab_switches}**")

# JS bridge simulation for updating session_state
st.experimental_data_editor({"tab_switches": st.session_state.tab_switches}, disabled=True)

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
        allow = st.checkbox("✅ I allow tab monitoring for proctoring (Required)")

        start = st.form_submit_button("🚀 Start Quiz")
        if start:
            if not allow:
                st.error("❌ Permission is required to start the quiz. Please check the box.")
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
        st.markdown("⚠️ Tab switching is monitored. Please stay on this tab!")
    else:
        st.error("Monitoring permission was not granted. Quiz may be invalid.")

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
    
    # Retrieve tab switches from DOM
    switch_count_str = st.experimental_get_query_params().get('tab_switches', ['0'])[0]
    try:
        recorded_switches = int(switch_count_str)
    except:
        recorded_switches = st.session_state.tab_switches
    
    st.warning(f"📉 You switched tabs **{recorded_switches} times** during the quiz.")
    if recorded_switches > 2:
        st.error("⚠️ High tab switching detected. Exam may be flagged for review.")
    else:
        st.success("✅ Acceptable tab switching behavior.")

