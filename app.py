import streamlit as st
import cv2
import dlib
import numpy as np
from imutils import face_utils
import time
from scipy.spatial import distance as dist
from flask import Flask, Response
import threading
import os

# Flask app for video feed
flask_app = Flask(__name__)

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Global variables
tab_change_count = 0
screen_visible = True
head_movement_alert = False
eye_tracking_alert = False
last_head_pose = None
start_time = time.time()

# Eye aspect ratio for blink detection
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Head pose estimation
def get_head_pose(shape):
    image_points = np.array([
        (shape[30][0], shape[30][1]),  # Nose tip
        (shape[8][0], shape[8][1]),   # Chin
        (shape[36][0], shape[36][1]), # Left eye left corner
        (shape[45][0], shape[45][1]), # Right eye right corner
        (shape[48][0], shape[48][1]), # Left mouth corner
        (shape[54][0], shape[54][1])  # Right mouth corner
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    focal_length = 640
    center = (320, 240)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    (_, rotation_vector, _) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    return rotation_vector

# Video capture function
def generate_frames():
    cap = cv2.VideoCapture(0)
    global head_movement_alert, eye_tracking_alert, last_head_pose

    while True:
        success, frame = cap.read()
        if not success:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            # Eye tracking
            left_eye = shape[36:42]
            right_eye = shape[42:48]
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            if ear < 0.21:  # Blink detection threshold
                eye_tracking_alert = True
            else:
                eye_tracking_alert = False

            # Head movement detection
            head_pose = get_head_pose(shape)
            if last_head_pose is not None:
                pose_diff = np.linalg.norm(head_pose - last_head_pose)
                if pose_diff > 0.5:  # Threshold for head movement
                    head_movement_alert = True
                else:
                    head_movement_alert = False
            last_head_pose = head_pose

            # Draw facial landmarks
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@flask_app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Streamlit app
def run_streamlit():
    st.set_page_config(page_title="Online Proctoring System", layout="wide")
    st.title("Online Proctoring System")

    # JavaScript for tab change and visibility detection
    st.markdown("""
    <script>
    let tabChangeCount = 0;
    document.addEventListener('visibilitychange', function() {
        if (document.hidden) {
            tabChangeCount++;
            fetch('/update_tab_count?count=' + tabChangeCount);
            document.getElementById('screen_status').innerText = 'Screen Not Visible';
        } else {
            document.getElementById('screen_status').innerText = 'Screen Visible';
        }
    });
    </script>
    <p>Tab Change Count: <span id='tab_count'>0</span></p>
    <p>Screen Status: <span id='screen_status'>Screen Visible</span></p>
    """, unsafe_allow_html=True)

    # Display video feed
    st.image("http://localhost:5000/video_feed", channels="BGR")

    # Proctoring alerts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Head Movement")
        if head_movement_alert:
            st.error("Suspicious head movement detected!")
        else:
            st.success("No head movement issues.")

    with col2:
        st.subheader("Eye Tracking")
        if eye_tracking_alert:
            st.error("Suspicious eye movement detected!")
        else:
            st.success("No eye movement issues.")

    # Update tab change count
    st.subheader("Tab Change Count")
    st.write(f"Total tab changes: {tab_change_count}")

@flask_app.route('/update_tab_count')
def update_tab_count():
    global tab_change_count
    tab_change_count += 1
    return '', 204

# Run Flask and Streamlit in separate threads
def run_flask():
    flask_app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    # Download shape predictor if not exists
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        st.write("Downloading shape predictor...")
        os.system("wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        os.system("bunzip2 shape_predictor_68_face_landmarks.dat.bz2")

    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()
    run_streamlit()
