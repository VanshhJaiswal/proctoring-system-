import streamlit as st
import requests
import base64
import os
from dotenv import load_dotenv
from datetime import datetime
import time

load_dotenv()

GOOGLE_VISION_API_KEY = os.environ.get("GOOGLE_VISION_API_KEY")

if not GOOGLE_VISION_API_KEY:
    st.error("Google Vision API Key missing. Please set it in .env file.")
    st.stop()

st.set_page_config(page_title="🎓 Proctoring System", page_icon="🎥")
st.title("🎓 Online Proctoring System with Google Vision API")

# Use a button to start live detection or set a timer
if st.button("Start Live Detection"):
    st.info("Live detection is now active. Click to stop.")
    capture_interval = 3  # seconds, set the time interval between image captures

    while True:
        uploaded_image = st.camera_input("📸 Take a snapshot for analysis (this will happen every few seconds)")

        if uploaded_image is not None:
            st.info("Analyzing image...")

            # Convert the image to base64
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

            resp = requests.post(vision_url, json=vision_payload)
            result = resp.json()

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

                # Eye open/close detection
                left_eye_open = face.get('leftEyeOpenLikelihood', 'UNKNOWN')
                right_eye_open = face.get('rightEyeOpenLikelihood', 'UNKNOWN')

                st.write(f"Eyes Open Likelihood: Left={left_eye_open}, Right={right_eye_open}")
                log_lines.append(f"Eyes Open Likelihood: Left={left_eye_open}, Right={right_eye_open}")

                if left_eye_open in ['VERY_UNLIKELY', 'UNLIKELY'] and right_eye_open in ['VERY_UNLIKELY', 'UNLIKELY']:
                    st.warning("⚠️ Eyes possibly closed.")

                # Mouth open/close detection
                mouth_open = face.get('mouthOpenLikelihood', 'UNKNOWN')
                st.write(f"Mouth Open Likelihood: {mouth_open}")
                log_lines.append(f"Mouth Open Likelihood: {mouth_open}")

                if mouth_open in ['LIKELY', 'VERY_LIKELY']:
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

            log_text = "\n".join(log_lines)

            with open("proctoring_log.txt", "a") as f:
                f.write(log_text + "\n")

            st.download_button("📥 Download Log", log_text, file_name="proctoring_log.txt")

        time.sleep(capture_interval)  # Wait for the next capture cycle
