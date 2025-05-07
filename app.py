import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize Mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Global activity log
activity_log = []

# FaceMesh & Pose initialization
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.start_time = time.time()
        self.window_switch_count = 0
        self.last_window_time = time.time()
        self.face_time = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        report = []

        # Face landmarks detection
        face_results = face_mesh.process(img_rgb)
        if face_results.multi_face_landmarks:
            report.append("✅ Face detected")
            self.face_time += 1

            for landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(img, landmarks, mp_face_mesh.FACEMESH_CONTOURS)

                # Eye open detection (landmark index for upper/lower eyelid)
                left_eye_top = landmarks.landmark[386]
                left_eye_bottom = landmarks.landmark[374]
                eye_distance = abs(left_eye_top.y - left_eye_bottom.y)
                if eye_distance < 0.01:
                    report.append("⚠️ Left eye closed")

                # Mouth open detection
                upper_lip = landmarks.landmark[13]
                lower_lip = landmarks.landmark[14]
                lip_distance = abs(upper_lip.y - lower_lip.y)
                if lip_distance > 0.03:
                    report.append("⚠️ Mouth open detected")

        else:
            report.append("❌ No face detected")

        # Head pose estimation
        pose_results = pose.process(img_rgb)
        if pose_results.pose_landmarks:
            nose_y = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y
            left_ear_y = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y
            if abs(nose_y - left_ear_y) > 0.05:
                report.append("⚠️ Head tilted")

        # Phone detection (basic rectangle detection as placeholder)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        phones = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
        if phones is not None:
            report.append("📱 Phone-like object detected")

        # Draw report messages on frame
        y_offset = 30
        for r in report:
            cv2.putText(img, r, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30

        # Append to log
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        for r in report:
            activity_log.append(f"[{timestamp}] {r}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.set_page_config(page_title="🖥️ Smart Proctored Mock Test")

st.title("🎓 Smart Proctored Mock Test System")
st.write("👉 This proctored test monitors your face, eyes, head, mouth & phone detection in real-time. Allow webcam access to start.")

duration = st.selectbox("Select test duration (minutes):", [1, 3, 10, 20])
duration_seconds = duration * 60

start_test = st.button("✅ Start Test")

if start_test:
    st.warning("⚠️ Please allow webcam access in browser popup to proceed.")

    processor = webrtc_streamer(
        key="exam-proctor",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

    start_time = time.time()

    while time.time() - start_time < duration_seconds:
        time.sleep(1)

    st.success("🎉 Test completed!")

    # Generate downloadable report
    st.subheader("📊 Proctoring Report")
    unique_logs = list(set(activity_log))
    for log in unique_logs:
        st.write(f"- {log}")

    report_text = "\n".join(unique_logs)
    st.download_button("📥 Download Report", report_text, file_name="Proctoring_Report.txt")

