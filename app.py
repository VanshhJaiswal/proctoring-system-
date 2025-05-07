import streamlit as st
import requests
import base64
import os
import time
from dotenv import load_dotenv
from datetime import datetime

# Load API Key
load_dotenv()
GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY")

if not GOOGLE_VISION_API_KEY:
    st.error("Google Vision API Key missing. Please set it in .env file.")
    st.stop()

# Streamlit page setup
st.set_page_config(page_title="🎓 Live Proctoring System", page_icon="🎥")
st.title("🎓 Online Proctoring System with Google Vision API (Live Snapshot)")

# Refresh interval in seconds
REFRESH_INTERVAL = 5

# Session state variables
if "last_run" not in st.session_state:
    st.session_state.last_run = time.time()
if "image_counter" not in st.session_state:
    st.session_state.image_counter = 0

# Trigger rerun after interval
now = time.time()
if now - st.session_state.last_run > REFRESH_INTERVAL:
    st.session_state.last_run = now
    st.session_state.image_counter += 1
    st.experimental_rerun()

# Capture image with dynamic key
camera_key = f"camera_{st.session_state.image_counter}"
uploaded_image = st.camera_input("📸 Live Snapshot (auto-refresh every 5 sec)", key=camera_key)

if uploaded_image is not None:
    st.info("Analyzing image...")

    # Encode image
    image_bytes = uploaded_image.getvalue()
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')

    vision_url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
    vision_payload = {
        "requests": [{
            "image": { "content": image_b64 },
            "features": [
                { "type": "FACE_DETECTION" },
                { "type": "OBJECT_LOCALIZATION" }
            ]
        }]
    }

    # Call Google Vision API
    response = requests.post(vision_url, json=vision_payload)
    result = response.json()

    # Logging setup
    log_lines = [f"{datetime.now()} - Proctoring Log"]

    ### FACE DETECTION ###
    faces = result.get('responses', [{}])[0].get('faceAnnotations', [])
    num_faces = len(faces)

    st.image(uploaded_image, caption=f"Detected {num_faces} face(s)")
    log_lines.append(f"Faces Detected: {num_faces}")

    if num_faces == 0:
        st.error("⚠️ No face detected!")
    elif num_faces > 1:
        st.warning(f"⚠️ Multiple faces detected ({num_faces})! Possible unauthorized person.")
    else:
        st.success("✅ 1 face detected.")

        face = faces[0]
        roll = face.get('rollAngle', 0)
        pan = face.get('panAngle', 0)
        tilt = face.get('tiltAngle', 0)

        st.write(f"Head Pose: Roll={roll:.2f}, Pan={pan:.2f}, Tilt={tilt:.2f}")
        log_lines.append(f"Head Pose: Roll={roll:.2f}, Pan={pan:.2f}, Tilt={tilt:.2f}")

        if abs(pan) > 20 or abs(tilt) > 20:
            st.warning("⚠️ Head turned away from screen.")

        left_eye = face.get('leftEyeOpenLikelihood', 'UNKNOWN')
        right_eye = face.get('rightEyeOpenLikelihood', 'UNKNOWN')
        mouth = face.get('mouthOpenLikelihood', 'UNKNOWN')

        st.write(f"Eyes Open Likelihood: Left={left_eye}, Right={right_eye}")
        log_lines.append(f"Eyes Open Likelihood: Left={left_eye}, Right={right_eye}")

        if left_eye in ['VERY_UNLIKELY', 'UNLIKELY'] and right_eye in ['VERY_UNLIKELY', 'UNLIKELY']:
            st.warning("⚠️ Eyes possibly closed.")

        st.write(f"Mouth Open Likelihood: {mouth}")
        log_lines.append(f"Mouth Open Likelihood: {mouth}")

        if mouth in ['LIKELY', 'VERY_LIKELY']:
            st.warning("⚠️ Mouth is open (possible talking).")

    ### OBJECT DETECTION ###
    objects = result.get('responses', [{}])[0].get('localizedObjectAnnotations', [])
    object_names = [obj['name'].lower() for obj in objects]

    st.write(f"Detected Objects: {object_names}")
    log_lines.append(f"Objects Detected: {object_names}")

    extra_persons = object_names.count('person') - 1
    if extra_persons > 0:
        st.warning(f"⚠️ {extra_persons} extra person(s) detected!")
    else:
        st.success("✅ No extra person detected.")

    if any(name in object_names for name in ['cell phone', 'mobile phone', 'telephone']):
        st.warning("⚠️ Phone detected in frame!")
    else:
        st.success("✅ No phone detected.")

    # Save and download log
    log_text = "\n".join(log_lines)
    with open("proctoring_log.txt", "a") as f:
        f.write(log_text + "\n")

    st.download_button("📥 Download Log", log_text, file_name="proctoring_log.txt")
