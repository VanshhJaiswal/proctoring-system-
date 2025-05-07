import os
import io
import pandas as pd
import numpy as np
import streamlit as st
from google.cloud import vision
from google.cloud.vision import types
from google.oauth2 import service_account
from PIL import Image

# Load credentials for Google Cloud Vision API
credentials = service_account.Credentials.from_service_account_file('path_to_your_google_cloud_service_account_key.json')
client = vision.ImageAnnotatorClient(credentials=credentials)

# Helper function for processing images using Google Vision API
def detect_faces(image_path):
    """Detect faces in an image using Google Vision API"""
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.face_detection(image=image)
    faces = response.face_annotations

    return faces

def detect_labels(image_path):
    """Detect labels (e.g., mobile phones) in an image"""
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.label_detection(image=image)
    labels = response.label_annotations

    detected_labels = [label.description for label in labels]
    return detected_labels

# Streamlit app layout
st.title("AI Proctoring System")

# Upload image for proctoring analysis
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Show the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save image temporarily to process with Google Vision API
    image_path = '/tmp/uploaded_image.jpg'
    image.save(image_path)

    # Run Face Detection
    faces = detect_faces(image_path)
    st.write(f"Number of faces detected: {len(faces)}")
    
    # Run Mobile Phone Detection (via label detection)
    labels = detect_labels(image_path)
    if any(label in labels for label in ['mobile', 'phone', 'cellphone']):
        st.warning("Mobile phone detected! Please ensure no mobile phone usage during the exam.")

    # Display detected faces and any other relevant information
    if faces:
        st.write("Faces Detected:")
        for i, face in enumerate(faces):
            st.write(f"Face {i + 1}: Joy: {face.joy_likelihood}, Sorrow: {face.sorrow_likelihood}, Anger: {face.anger_likelihood}")

    # Log results for download (if needed)
    log_data = {
        "Detected Faces": len(faces),
        "Detected Labels": labels,
    }
    log_df = pd.DataFrame([log_data])
    st.download_button(
        label="Download Proctoring Log",
        data=log_df.to_csv(index=False),
        file_name="proctoring_log.csv",
        mime="text/csv"
    )

# Optionally, other proctoring features can be added, such as tracking absences or face movements.
