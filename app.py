import streamlit as st
import requests
import base64
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from streamlit.components.v1 import html
from camera_input_live import camera_input_live

# Load environment variables
load_dotenv()
GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Check for API keys
if not GOOGLE_VISION_API_KEY or not GROQ_API_KEY:
    st.error("Missing API keys. Please set GOOGLE_VISION_API_KEY and GROQ_API_KEY in the .env file.")
    st.stop()

# Set page configuration
st.set_page_config(page_title="🎓 Proctored Quiz", page_icon="📝")

# Initialize session state
if "quiz_started" not in st.session_state:
    st.session_state.quiz_started = False
    st.session_state.questions = []
    st.session_state.answers = {}
    st.session_state.proctoring_logs = []
    st.session_state.tab_switches = 0
    st.session_state.start_time = None

# JavaScript for tab switch detection
tab_switch_js = """
<script>
let hidden, visibilityChange; 
if (typeof document.hidden !== "undefined") {
  hidden = "hidden";
  visibilityChange = "visibilitychange";
} else if (typeof document.msHidden !== "undefined") {
  hidden = "msHidden";
  visibilityChange = "msvisibilitychange";
} else if (typeof document.webkitHidden !== "undefined") {
  hidden = "webkitHidden";
  visibilityChange = "webkitvisibilitychange";
}

document.addEventListener(visibilityChange, function() {
  if (document[hidden]) {
    fetch("/tab_switch");
  }
}, false);
</script>
"""

# Display JavaScript
html(tab_switch_js)

# Function to generate quiz questions using GROQ API
def generate_quiz():
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mixtral-8x7b-32768",
        "messages": [
            {"role": "system", "content": "Generate a 5-question multiple-choice quiz on general knowledge. Each question should have 4 options labeled A-D, and provide the correct answer."}
        ]
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
    if response.status_code == 200:
        content = response.json()["choices"][0]["message"]["content"]
        # Parse the content into questions and options
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
    else:
        st.error("Failed to generate quiz questions.")
        return []

# Function to analyze image using Google Vision API
def analyze_image(image_bytes):
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    vision_url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
    vision_payload = {
        "requests": [{
            "image": {"content": image_b64},
            "features": [
                {"type": "FACE_DETECTION"},
                {"type": "OBJECT_LOCALIZATION"}
            ]
        }]
    }
    resp = requests.post(vision_url, json=vision_payload)
    if resp.status_code == 200:
        return resp.json()
    else:
        return {}

# Function to process proctoring data
def process_proctoring_data(result):
    log = {}
    faces = result.get('responses', [{}])[0].get('faceAnnotations', [])
    objects = result.get('responses', [{}])[0].get('localizedObjectAnnotations', [])
    object_names = [obj['name'].lower() for obj in objects]

    log['timestamp'] = datetime.now().isoformat()
    log['num_faces'] = len(faces)
    log['multiple_faces'] = len(faces) > 1
    log['phone_detected'] = any(name in object_names for name in ['cell phone', 'mobile phone', 'telephone'])
    log['extra_persons'] = object_names.count('person') - 1

    if faces:
        face = faces[0]
        log['head_pose'] = {
            'roll': face.get('rollAngle', 0),
            'pan': face.get('panAngle', 0),
            'tilt': face.get('tiltAngle', 0)
        }
        log['eyes_open'] = {
            'left': face.get('leftEyeOpenLikelihood', 'UNKNOWN'),
            'right': face.get('rightEyeOpenLikelihood', 'UNKNOWN')
        }
        log['mouth_open'] = face.get('mouthOpenLikelihood', 'UNKNOWN')
    else:
        log['head_pose'] = {}
        log['eyes_open'] = {}
        log['mouth_open'] = 'UNKNOWN'

    return log

# Start quiz
if not st.session_state.quiz_started:
    st.title("🎓 Proctored Quiz Application")
    if st.button("Start Quiz"):
        st.session_state.questions = generate_quiz()
        st.session_state.quiz_started = True
        st.session_state.start_time = time.time()
        st.experimental_rerun()
else:
    st.title("📝 Quiz in Progress")
    st.write("Please answer the following questions:")

    # Display questions
    for idx, q in enumerate(st.session_state.questions):
        st.write(f"**{q['question']}**")
        for option in q['options']:
            st.radio(f"Question {idx+1}", q['options'], key=f"q{idx}")

    # Live camera input
    image = camera_input_live()
    if image:
        image_bytes = image.getvalue()
        result = analyze_image(image_bytes)
        log = process_proctoring_data(result)
        st.session_state.proctoring_logs.append(log)

    # Submit button
    if st.button("Submit Quiz"):
        st.session_state.quiz_started = False
        st.session_state.end_time = time.time()
        st.experimental_rerun()

# Display report after submission
if not st.session_state.quiz_started and st.session_state.questions:
    st.title("📊 Quiz Report")

    # Calculate score
    score = 0
    for idx, q in enumerate(st.session_state.questions):
        user_answer = st.session_state.get(f"q{idx}", "")
        correct_answer = q['answer']
        if user_answer and user_answer.startswith(correct_answer):
            score += 1
    st.write(f"**Your Score: {score} out of {len(st.session_state.questions)}**")

    # Proctoring summary
    st.subheader("Proctoring Summary")
    total_logs = len(st.session_state.proctoring_logs)
    multiple_faces = sum(1 for log in st.session_state.proctoring_logs if log['multiple_faces'])
    phones_detected = sum(1 for log in st.session_state.proctoring_logs if log['phone_detected'])
    extra_persons = sum(log['extra_persons'] for log in st.session_state.proctoring_logs)

    st.write(f"Total Monitoring Instances: {total_logs}")
    st.write(f"Instances with Multiple Faces: {multiple_faces}")
    st.write(f"Instances with Phone Detected: {phones_detected}")
    st.write(f"Total Extra Persons Detected: {extra_persons}")
    st.write(f"Tab Switches Detected: {st.session_state.tab_switches}")

    # Download report
    report = f"""
    Quiz Score: {score}/{len(st.session_state.questions)}
    Total Monitoring Instances: {total_logs}
    Instances with Multiple Faces: {multiple_faces}
    Instances with Phone Detected: {phones_detected}
    Total Extra Persons Detected: {extra_persons}
    Tab Switches Detected: {st.session_state.tab_switches}
    """
    st.download_button("Download Report", report, file_name="proctoring_report.txt")
