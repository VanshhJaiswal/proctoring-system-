import streamlit as st
import requests
import os
import base64
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

st.set_page_config(page_title="🎓 Proctoring System", page_icon="🎥")
st.title("🎓 Online Proctoring System with Face, Head, Eye, Mouth, Phone Detection")

# Get API keys
FACEPP_API_KEY = os.environ.get("FACEPP_API_KEY")
FACEPP_API_SECRET = os.environ.get("FACEPP_API_SECRET")
GOOGLE_VISION_API_KEY = os.environ.get("GOOGLE_VISION_API_KEY")

if not FACEPP_API_KEY or not FACEPP_API_SECRET or not GOOGLE_VISION_API_KEY:
    st.error("API Keys missing in environment variables. Please check .env or config.")
    st.stop()

uploaded_image = st.camera_input("📸 Take a snapshot for analysis")

if uploaded_image is not None:
    st.info("Analyzing image...")

    image_bytes = uploaded_image.getvalue()

    # Send to Face++ API
    facepp_params = {
        'api_key': FACEPP_API_KEY,
        'api_secret': FACEPP_API_SECRET,
        'return_landmark': 1,
        'return_attributes': 'headpose,eye_status,mouthstatus,gender,age'
    }
    facepp_files = {'image_file': image_bytes}

    facepp_resp = requests.post('https://api-us.faceplusplus.com/facepp/v3/detect', data=facepp_params, files=facepp_files)
    facepp_result = facepp_resp.json()

    if 'error_message' in facepp_result:
        st.error(f"Face++ API Error: {facepp_result['error_message']}")
        st.stop()

    faces = facepp_result.get('faces', [])
    num_faces = len(faces)

    st.image(uploaded_image, caption=f"Detected {num_faces} face(s)")

    # Log
    log_lines = [f"{datetime.now()} - Faces Detected: {num_faces}"]

    if num_faces == 0:
        st.error("⚠️ No face detected!")
    elif num_faces > 1:
        st.warning(f"⚠️ Multiple faces detected ({num_faces})! Possible unauthorized person.")
    else:
        st.success("✅ 1 face detected.")

        attrs = faces[0]['attributes']

        # Head Pose
        headpose = attrs['headpose']
        yaw, pitch, roll = headpose['yaw_angle'], headpose['pitch_angle'], headpose['roll_angle']
        st.write(f"Head Pose: Yaw={yaw:.2f}, Pitch={pitch:.2f}, Roll={roll:.2f}")
        if abs(yaw) > 20 or abs(pitch) > 20:
            st.warning("⚠️ Head turned away from screen")
        log_lines.append(f"HeadPose: Yaw={yaw:.2f}, Pitch={pitch:.2f}, Roll={roll:.2f}")

        # Eye Status
        eye_status = attrs['eye_status']
        left_eye_open = eye_status['left_eye_status']['no_glass_eye_open']
        right_eye_open = eye_status['right_eye_status']['no_glass_eye_open']
        st.write(f"Eyes Open: Left={left_eye_open:.1f}%, Right={right_eye_open:.1f}%")
        if left_eye_open < 50 and right_eye_open < 50:
            st.warning("⚠️ Eyes possibly closed")
        log_lines.append(f"Eye Open: Left={left_eye_open:.1f}%, Right={right_eye_open:.1f}%")

        # Mouth Status
        mouth_open = attrs['mouthstatus']['open']
        st.write(f"Mouth Open: {mouth_open}%")
        if mouth_open > 50:
            st.warning("⚠️ Mouth is open (possible talking)")
        log_lines.append(f"Mouth Open: {mouth_open}%")

    ### --- Google Vision API for phone detection ---
    st.write("🔍 Running object detection for phone/person...")

    # Prepare image as base64
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    vision_url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"

    vision_payload = {
        "requests": [{
            "image": { "content": image_b64 },
            "features": [{ "type": "OBJECT_LOCALIZATION" }]
        }]
    }

    vision_resp = requests.post(vision_url, json=vision_payload)
    vision_result = vision_resp.json()

    objects = []
    if 'responses' in vision_result and 'localizedObjectAnnotations' in vision_result['responses'][0]:
        objects = vision_result['responses'][0]['localizedObjectAnnotations']

    object_names = [obj['name'].lower() for obj in objects]
    st.write(f"Detected Objects: {object_names}")
    log_lines.append(f"Objects Detected: {object_names}")

    if 'mobile phone' in object_names or 'cell phone' in object_names or 'telephone' in object_names:
        st.warning("⚠️ Phone detected in frame!")
    else:
        st.success("✅ No phone detected.")

    extra_persons = object_names.count('person') - 1  # subtract main person
    if extra_persons > 0:
        st.warning(f"⚠️ {extra_persons} extra person(s) detected!")
    else:
        st.success("✅ No extra person detected.")

    # Save log
    log_text = "\n".join(log_lines)
    with open("proctoring_log.txt", "a") as f:
        f.write(log_text + "\n")

    st.download_button("📥 Download Log", log_text, file_name="proctoring_log.txt")

