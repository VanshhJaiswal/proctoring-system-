import streamlit as st
import requests
import base64
import os
import groq
import json
from datetime import datetime

# Load API keys
GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
groq_client = groq.Client(api_key=GROQ_API_KEY)

st.title("🎥 AI-Based Proctoring System with Quiz + Tab Switch Detection")

# Initialize logs
if 'logs' not in st.session_state:
    st.session_state.logs = []

def log_event(event_type, details=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.logs.append({"time": timestamp, "event": event_type, "details": details})

# Inject JavaScript for tab-switch detection
st.components.v1.html("""
<script>
    document.addEventListener("visibilitychange", function() {
        if (document.hidden) {
            fetch("/_stcore/streamlit/message?tab_switch=true", {method: "POST"});
        }
    });
</script>
""", height=0)

if st.experimental_get_query_params().get("tab_switch"):
    log_event("Tab Switch Detected", "User switched away from the exam tab!")
    st.warning("⚠️ Tab switch detected! Please stay on this exam tab.")

# =========================
# 1️⃣ Live Webcam Capture
# =========================

captured_image = st.camera_input("📸 Capture live photo for proctoring")

if captured_image is not None:
    st.image(captured_image, caption="Captured Image", use_column_width=True)

    # Encode image to base64
    image_content = captured_image.getvalue()
    encoded_image = base64.b64encode(image_content).decode('utf-8')

    # Google Vision API endpoint
    vision_api_url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"

    # Prepare request payload
    request_payload = {
        "requests": [
            {
                "image": {"content": encoded_image},
                "features": [{"type": "FACE_DETECTION"}]
            }
        ]
    }

    with st.spinner("🔍 Analyzing image with Google Vision API..."):
        response = requests.post(vision_api_url, json=request_payload)
        result = response.json()

    try:
        face_annotations = result["responses"][0].get("faceAnnotations", [])
        num_faces = len(face_annotations)
        alerts = []

        if num_faces == 0:
            alerts.append("⚠️ No Face Detected")
        elif num_faces > 1:
            alerts.append("⚠️ Multiple Faces Detected")
        else:
            alerts.append("✅ Single Face Detected")

        emotions = []
        for face in face_annotations:
            likelihood_map = {
                "VERY_LIKELY": 5,
                "LIKELY": 4,
                "POSSIBLE": 3,
                "UNLIKELY": 2,
                "VERY_UNLIKELY": 1
            }
            if likelihood_map.get(face.get("joyLikelihood", "VERY_UNLIKELY"), 0) >= 3:
                emotions.append("Joy")
            if likelihood_map.get(face.get("angerLikelihood", "VERY_UNLIKELY"), 0) >= 3:
                emotions.append("Anger")
            if likelihood_map.get(face.get("sorrowLikelihood", "VERY_UNLIKELY"), 0) >= 3:
                emotions.append("Sorrow")
            if likelihood_map.get(face.get("surpriseLikelihood", "VERY_UNLIKELY"), 0) >= 3:
                emotions.append("Surprise")

        st.subheader("✅ Proctoring Analysis Results")
        st.write(f"**Faces Detected:** {num_faces}")
        st.write(f"**Alerts:** {', '.join(alerts)}")
        st.write(f"**Emotions Detected:** {', '.join(emotions) if emotions else 'None'}")

        log_event("Proctoring Analysis", f"Faces: {num_faces}, Alerts: {alerts}, Emotions: {emotions}")

    except Exception as e:
        st.error(f"❌ Error processing response: {e}")
        st.json(result)  # for debugging

# =========================
# 2️⃣ Groq Quiz Generation
# =========================

if st.button("🎯 Generate AI Quiz Question"):
    with st.spinner("🤖 Generating quiz using Groq..."):
        prompt_text = "Generate one multiple-choice question on Python programming with 4 options and specify the correct answer."

        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt_text}],
            model="llama3-8b-8192"
        )
        quiz_content = chat_completion.choices[0].message.content
        st.session_state.quiz_question = quiz_content
        log_event("Quiz Generated", quiz_content)

if "quiz_question" in st.session_state:
    st.subheader("📝 AI-Generated Quiz")
    st.markdown(st.session_state.quiz_question)
    user_answer = st.text_input("Your Answer:")
    if st.button("Submit Answer"):
        log_event("Quiz Answer", f"User answered: {user_answer}")
        st.success("✅ Answer submitted and logged!")

# =========================
# 3️⃣ Download Logs
# =========================

st.subheader("📄 Session Logs")
st.json(st.session_state.logs)

log_json = json.dumps(st.session_state.logs, indent=2)
st.download_button("⬇️ Download Logs as JSON", log_json, file_name="proctoring_logs.json", mime="application/json")
