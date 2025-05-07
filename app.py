import streamlit as st
import requests
import base64
import os
from groq import Groq
import json

GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

st.title("AI-Based Proctoring System (Google Vision API + Groq API)")

uploaded_file = st.file_uploader("Upload an image for proctoring", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image")

    # Read and encode image
    content = uploaded_file.read()
    encoded_image = base64.b64encode(content).decode('utf-8')

    vision_api_url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"

    # Prepare Google Vision API request
    request_payload = {
        "requests": [
            {
                "image": {"content": encoded_image},
                "features": [{"type": "FACE_DETECTION"}]
            }
        ]
    }

    with st.spinner("Analyzing image with Google Vision API..."):
        response = requests.post(vision_api_url, json=request_payload)
        result = response.json()

    # Handle Vision API response
    try:
        face_annotations = result["responses"][0].get("faceAnnotations", [])
        num_faces = len(face_annotations)
        alerts = []

        if num_faces == 0:
            alerts.append("No Face Detected")
        elif num_faces > 1:
            alerts.append("Multiple Faces Detected")
        else:
            alerts.append("Single Face Detected")

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

        st.write(f"**Faces Detected:** {num_faces}")
        st.write(f"**Alerts:** {', '.join(alerts)}")
        st.write(f"**Emotions Detected:** {', '.join(emotions) if emotions else 'None'}")

        # Prepare prompt for Groq
        prompt_text = f"There are {num_faces} face(s) detected. Emotions detected: {emotions if emotions else 'None'}. Write a proctoring report and flag any suspicious behavior."

        with st.spinner("Generating AI interpretation with Groq API..."):
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt_text}],
                model="llama3-8b-8192"
            )
            interpretation = chat_completion.choices[0].message.content

        st.subheader("AI Interpretation")
        st.write(interpretation)

    except Exception as e:
        st.error(f"Error processing API response: {e}")
        st.json(result)  # Show full response for debugging
