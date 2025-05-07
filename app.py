import streamlit as st
import cv2
import numpy as np
import time
import datetime
import os
import json
import requests
from PIL import Image
from io import BytesIO
import base64
import platform
import pandas as pd
from dotenv import load_dotenv
import threading
import mediapipe as mp
import streamlit_javascript as st_js
import sys

# Set page configuration
st.set_page_config(
    page_title="AI Proctoring System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Initialize MediaPipe face detection and face mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize Groq client - Check if API key is available
groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = None
GROQ_MODEL = "llama3-8b-8192"  # Using Llama 3 8B model which is free to use

# Import Groq only if API key is available
try:
    from groq import Groq
    if groq_api_key:
        groq_client = Groq(api_key=groq_api_key)
    else:
        st.sidebar.error("GROQ_API_KEY not found. Please add it to your .env file or Streamlit secrets.")
except ImportError:
    st.sidebar.error("The groq package is not installed. Please make sure it's in your requirements.txt file.")

# Session state initialization
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'proctoring_data' not in st.session_state:
    st.session_state.proctoring_data = {
        'face_counts': [],
        'blink_counts': [],
        'tab_switches': 0,
        'time_on_camera': 0,
        'timestamps': []
    }
if 'tab_switch_detected' not in st.session_state:
    st.session_state.tab_switch_detected = False
if 'quiz_data' not in st.session_state:
    st.session_state.quiz_data = None
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False
if 'report_data' not in st.session_state:
    st.session_state.report_data = None
if 'quiz_submitted' not in st.session_state:
    st.session_state.quiz_submitted = False
if 'blink_counter' not in st.session_state:
    st.session_state.blink_counter = 0
if 'last_blink_check' not in st.session_state:
    st.session_state.last_blink_check = time.time()

# CSS for better UI
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .warning {
        color: red;
        font-weight: bold;
    }
    .report-container {
        padding: 20px;
        border-radius: 5px;
        background-color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .header {
        text-align: center;
        color: #2C3E50;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

def generate_quiz(topic, difficulty, num_questions, time_limit):
    """Generate a quiz using Groq API or fallback to demo data if API is unavailable"""
    # Check if Groq client is available
    if groq_client is None:
        st.warning("Groq API key not configured. Using demo quiz data.")
        # Return a demo quiz instead
        return {
            "title": f"Demo Quiz: {topic}",
            "description": f"This is a demo {difficulty.lower()} quiz on {topic}. Add your Groq API key to generate custom quizzes.",
            "time_limit_minutes": time_limit,
            "questions": [
                {
                    "id": 1,
                    "question": "What does AI stand for?",
                    "options": ["Artificial Intelligence", "Automated Information", "Augmented Interface", "Algorithmic Integration"],
                    "correct_answer": "A"
                },
                {
                    "id": 2,
                    "question": "Which of these is not a popular programming language?",
                    "options": ["Python", "JavaScript", "HTML", "MadeUpLang"],
                    "correct_answer": "D"
                },
                {
                    "id": 3,
                    "question": "What is the main purpose of proctoring software?",
                    "options": ["Entertainment", "Monitoring during exams", "Video editing", "Social networking"],
                    "correct_answer": "B"
                },
                {
                    "id": 4,
                    "question": "Which technology is commonly used for face detection?",
                    "options": ["GPS", "Computer Vision", "Blockchain", "5G"],
                    "correct_answer": "B"
                },
                {
                    "id": 5,
                    "question": "What does ML stand for in tech?",
                    "options": ["Multiple Layers", "Machine Learning", "Meta Language", "Mobile Logic"],
                    "correct_answer": "B"
                }
            ]
        }
    
    try:
        prompt = f"""
        Create a quiz on the topic of {topic} with {num_questions} questions.
        Difficulty level: {difficulty}
        Time limit: {time_limit} minutes
        
        Format the questions as a JSON array with the following structure:
        {{
            "title": "Quiz title",
            "description": "Brief description of the quiz",
            "time_limit_minutes": {time_limit},
            "questions": [
                {{
                    "id": 1,
                    "question": "Question text",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_answer": "Correct option letter"
                }},
                ...
            ]
        }}
        
        Ensure that all questions are well-structured and appropriate for the given difficulty level.
        Return only the JSON structure without any additional text.
        """
        
        chat_completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful quiz generator assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.5
        )
        
        response_text = chat_completion.choices[0].message.content
        
        # Extract JSON from response (in case the AI adds any extra text)
        import re
        json_match = re.search(r'```json\n([\s\S]*?)\n```', response_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response_text
            
        # Clean up any non-JSON text
        start_idx = json_str.find('{')
        end_idx = json_str.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            json_str = json_str[start_idx:end_idx]
            
        quiz_data = json.loads(json_str)
        return quiz_data
    
    except Exception as e:
        st.error(f"Error generating quiz: {str(e)}")
        # Return a demo quiz as fallback
        return {
            "title": f"Fallback Quiz: {topic}",
            "description": f"This is a fallback {difficulty.lower()} quiz on {topic} because there was an error with the Groq API.",
            "time_limit_minutes": time_limit,
            "questions": [
                {
                    "id": 1,
                    "question": "What does AI stand for?",
                    "options": ["Artificial Intelligence", "Automated Information", "Augmented Interface", "Algorithmic Integration"],
                    "correct_answer": "A"
                },
                {
                    "id": 2,
                    "question": "Which programming language is often used for data science?",
                    "options": ["Java", "C++", "Python", "Ruby"],
                    "correct_answer": "C"
                },
                {
                    "id": 3,
                    "question": "What is computer vision primarily used for?",
                    "options": ["Image processing", "Audio analysis", "Text generation", "Network security"],
                    "correct_answer": "A"
                }
            ]
        }

def detect_blinks(landmarks, face_oval):
    """Detect eye blinks using facial landmarks"""
    # Get landmarks for left and right eyes
    if landmarks:
        # LEFT_EYE landmarks indices
        left_eye = [landmarks[33], landmarks[160], landmarks[158], landmarks[133], landmarks[153], landmarks[144]]
        
        # RIGHT_EYE landmarks indices
        right_eye = [landmarks[362], landmarks[385], landmarks[387], landmarks[263], landmarks[373], landmarks[380]]
        
        # Calculate Eye Aspect Ratio (EAR)
        def calculate_ear(eye_points):
            # Compute the euclidean distance between vertical eye landmarks
            height_1 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
            height_2 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
            
            # Compute the euclidean distance between horizontal eye landmarks
            width = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
            
            # Calculate the eye aspect ratio
            ear = (height_1 + height_2) / (2.0 * width)
            return ear
        
        # Get EAR for both eyes
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        
        # Average the EAR of both eyes
        ear = (left_ear + right_ear) / 2.0
        
        # Blink detected if EAR is below threshold
        EAR_THRESHOLD = 0.2
        
        if ear < EAR_THRESHOLD:
            current_time = time.time()
            # Add cooldown to avoid counting multiple blinks for the same event
            if current_time - st.session_state.last_blink_check > 1.0:  # 1 second cooldown
                st.session_state.blink_counter += 1
                st.session_state.last_blink_check = current_time
                return True
    
    return False

def process_webcam_feed():
    """Process webcam feed to detect faces and eye blinks"""
    if not st.session_state.monitoring_active:
        return
    
    # Get webcam feed
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam. Please check your camera connection.")
        return
    
    stframe = st.empty()
    
    try:
        while st.session_state.monitoring_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to get frame from webcam.")
                break
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_results = face_detection.process(rgb_frame)
            
            # Process with Face Mesh
            mesh_results = face_mesh.process(rgb_frame)
            
            # Draw face detections
            face_count = 0
            if face_results.detections:
                face_count = len(face_results.detections)
                for detection in face_results.detections:
                    mp_drawing.draw_detection(frame, detection)
            
            # Extract face landmarks and detect blinks
            landmarks_list = []
            if mesh_results.multi_face_landmarks:
                for face_landmarks in mesh_results.multi_face_landmarks:
                    # Convert landmarks to pixel coordinates
                    landmarks = []
                    for landmark in face_landmarks.landmark:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        landmarks.append((x, y))
                    
                    landmarks_list.append(landmarks)
                    
                    # Face oval indices for mesh visualization
                    face_oval = [
                        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
                    ]
                    
                    # Detect blinks
                    blink_detected = detect_blinks(landmarks, face_oval)
                    
                    # Draw face mesh
                    mp_drawing.draw_landmarks(
                        frame, 
                        face_landmarks,
                        mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
                    )
            
            # Record proctoring data
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.proctoring_data['face_counts'].append(face_count)
            st.session_state.proctoring_data['blink_counts'].append(st.session_state.blink_counter)
            st.session_state.proctoring_data['timestamps'].append(timestamp)
            
            # Display the frame with annotations
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Add overlay with proctoring information
            info_text = f"Faces: {face_count} | Blinks: {st.session_state.blink_counter} | Tab Switches: {st.session_state.proctoring_data['tab_switches']}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Warning if multiple faces detected
            if face_count > 1:
                warning_text = "WARNING: Multiple faces detected!"
                cv2.putText(frame, warning_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Warning if no faces detected
            if face_count == 0:
                warning_text = "WARNING: No face detected!"
                cv2.putText(frame, warning_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            stframe.image(frame, channels="RGB", use_column_width=True)
            
            # Update time on camera
            if st.session_state.start_time is not None:
                st.session_state.proctoring_data['time_on_camera'] = time.time() - st.session_state.start_time
            
            time.sleep(0.1)  # Small delay to reduce CPU usage
    
    finally:
        cap.release()

def detect_tab_switch():
    """Detect tab switching using JavaScript"""
    if st.session_state.monitoring_active:
        try:
            # This will inject JavaScript to detect tab/window visibility changes
            js_code = """
            var hidden, visibilityChange;
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
            
            var isHidden = function() {
                return document[hidden];
            };
            
            isHidden();
            """
            
            is_hidden = st_js.st_javascript(js_code)
            
            if is_hidden and not st.session_state.tab_switch_detected:
                st.session_state.tab_switch_detected = True
                st.session_state.proctoring_data['tab_switches'] += 1
            elif not is_hidden:
                st.session_state.tab_switch_detected = False
        except:
            # If JavaScript detection fails, we'll continue without it
            pass

def generate_report():
    """Generate a comprehensive proctoring report"""
    if not st.session_state.proctoring_data:
        return None
    
    # Calculate metrics
    total_time = st.session_state.proctoring_data['time_on_camera']
    face_counts = st.session_state.proctoring_data['face_counts']
    no_face_count = face_counts.count(0)
    multiple_face_instances = sum(1 for count in face_counts if count > 1)
    
    # Calculate time without face visible (in seconds)
    time_per_frame = total_time / len(face_counts) if len(face_counts) > 0 else 0
    time_without_face = no_face_count * time_per_frame
    
    # Calculate percentage of time with face visible
    face_visibility_percentage = (1 - (time_without_face / total_time)) * 100 if total_time > 0 else 0
    
    # Create report data
    report = {
        'total_exam_time_minutes': round(total_time / 60, 2),
        'face_visibility_percentage': round(face_visibility_percentage, 2),
        'no_face_detected_instances': no_face_count,
        'multiple_faces_detected_instances': multiple_face_instances,
        'total_blinks': st.session_state.blink_counter,
        'blink_rate_per_minute': round(st.session_state.blink_counter / (total_time / 60), 2) if total_time > 0 else 0,
        'tab_switches': st.session_state.proctoring_data['tab_switches'],
        'potential_cheating_risk': calculate_cheating_risk(
            face_visibility_percentage, 
            multiple_face_instances,
            st.session_state.proctoring_data['tab_switches'],
            total_time
        )
    }
    
    return report

def calculate_cheating_risk(face_visibility, multiple_faces, tab_switches, total_time):
    """Calculate a simple cheating risk score"""
    # Convert total_time to minutes
    total_time_minutes = total_time / 60
    
    # Base score starts at 0 (low risk)
    risk_score = 0
    
    # Factors that increase risk
    # Face visibility below 90% is concerning
    if face_visibility < 90:
        risk_score += (90 - face_visibility) * 0.5
    
    # Multiple faces detected
    risk_score += multiple_faces * 20
    
    # Tab switches (each switch adds 10 points)
    risk_score += tab_switches * 10
    
    # Normalize by exam duration (shorter exams have less opportunity for issues)
    risk_score = risk_score / max(total_time_minutes, 1)
    
    # Cap at 100
    risk_score = min(risk_score, 100)
    
    # Determine risk level
    if risk_score < 20:
        return "Low"
    elif risk_score < 50:
        return "Medium"
    else:
        return "High"

def main():
    st.markdown("<h1 class='header'>AI-Based Proctoring System with Quiz Generation</h1>", unsafe_allow_html=True)
    
    # Sidebar for navigation and controls
    with st.sidebar:
        st.image("https://i.imgur.com/IkbWiLH.png", use_column_width=True)
        st.header("Navigation")
        page = st.radio("Select a page:", ["Setup Quiz", "Take Quiz", "View Report"])
        
        if page == "Setup Quiz":
            st.subheader("Guide")
            st.info("1. Enter quiz details\n2. Generate quiz\n3. Start the exam")
            
            st.subheader("API Setup Help")
            with st.expander("How to get a Groq API Key"):
                st.markdown("""
                1. Go to [Groq's website](https://console.groq.com/keys)
                2. Sign up for a free account
                3. Navigate to API Keys section
                4. Create a new API key
                5. Copy the key and paste it in the .env file
                """)
        
        elif page == "Take Quiz":
            st.subheader("Proctoring Information")
            if st.session_state.monitoring_active:
                st.success("Proctoring is active")
                if st.button("Stop Proctoring"):
                    st.session_state.monitoring_active = False
                    
                # Show live proctoring metrics
                st.metric("Faces Detected", 
                          st.session_state.proctoring_data['face_counts'][-1] if st.session_state.proctoring_data['face_counts'] else 0)
                st.metric("Total Blinks", st.session_state.blink_counter)
                st.metric("Tab Switches", st.session_state.proctoring_data['tab_switches'])
                time_on_camera = st.session_state.proctoring_data['time_on_camera']
                st.metric("Time (minutes)", round(time_on_camera / 60, 2) if time_on_camera else 0)
            else:
                st.warning("Proctoring is not active")
                if st.button("Start Proctoring"):
                    st.session_state.monitoring_active = True
                    st.session_state.start_time = time.time()
                    # Start the webcam processing in a separate thread
                    threading.Thread(target=process_webcam_feed, daemon=True).start()
        
        elif page == "View Report":
            st.subheader("Report Information")
            if st.session_state.report_data:
                st.success("Report generated successfully")
            else:
                st.warning("No report available yet")

    # Setup Quiz Page
    if page == "Setup Quiz":
        st.subheader("📝 Setup Quiz Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            topic = st.text_input("Quiz Topic", value="Python Programming")
            difficulty = st.select_slider("Difficulty Level", options=["Easy", "Medium", "Hard"])
        
        with col2:
            num_questions = st.number_input("Number of Questions", min_value=1, max_value=20, value=5)
            time_limit = st.number_input("Time Limit (minutes)", min_value=1, max_value=180, value=10)
        
        if st.button("Generate Quiz"):
            with st.spinner("Generating quiz using Groq API..."):
                quiz_data = generate_quiz(topic, difficulty, num_questions, time_limit)
                
                if quiz_data:
                    st.session_state.quiz_data = quiz_data
                    st.session_state.user_answers = {}
                    st.success("Quiz generated successfully! Go to 'Take Quiz' page to start.")
                    
                    # Preview the quiz
                    with st.expander("Preview Quiz"):
                        st.markdown(f"### {quiz_data['title']}")
                        st.markdown(quiz_data['description'])
                        st.markdown(f"**Time Limit:** {quiz_data['time_limit_minutes']} minutes")
                        st.markdown(f"**Number of Questions:** {len(quiz_data['questions'])}")
                        
                        for i, q in enumerate(quiz_data['questions']):
                            st.markdown(f"**Q{i+1}: {q['question']}**")
                            for opt in q['options']:
                                st.markdown(f"- {opt}")
                            st.markdown("---")

    # Take Quiz Page
    elif page == "Take Quiz":
        st.subheader("📝 Take Quiz")
        
        # Detect tab switching
        detect_tab_switch()
        
        if st.session_state.quiz_data:
            quiz_data = st.session_state.quiz_data
            
            # Display quiz information
            st.markdown(f"### {quiz_data['title']}")
            st.markdown(quiz_data['description'])
            
            # Calculate remaining time
            if st.session_state.start_time is not None:
                elapsed_time = time.time() - st.session_state.start_time
                remaining_time = max(0, quiz_data['time_limit_minutes'] * 60 - elapsed_time)
                minutes, seconds = divmod(int(remaining_time), 60)
                st.markdown(f"**Time Remaining:** {minutes:02d}:{seconds:02d}")
                
                # Auto-submit when time is up
                if remaining_time <= 0 and not st.session_state.quiz_submitted:
                    st.warning("Time's up! Quiz has been automatically submitted.")
                    st.session_state.quiz_submitted = True
                    st.session_state.monitoring_active = False
                    st.session_state.report_data = generate_report()
                    st.experimental_rerun()
            
            if not st.session_state.quiz_submitted:
                # Display questions and collect answers
                for i, question in enumerate(quiz_data['questions']):
                    q_id = question['id']
                    st.markdown(f"**Question {i+1}: {question['question']}**")
                    
                    # Use radio buttons for multiple choice
                    options = question['options']
                    st.session_state.user_answers[q_id] = st.radio(
                        f"Select answer for question {i+1}",
                        options=options,
                        key=f"q_{q_id}",
                        index=options.index(st.session_state.user_answers.get(q_id, options[0])) if q_id in st.session_state.user_answers else 0
                    )
                
                # Submit button
                if st.button("Submit Quiz"):
                    st.session_state.quiz_submitted = True
                    st.session_state.monitoring_active = False
                    st.session_state.report_data = generate_report()
                    st.success("Quiz submitted successfully! Go to 'View Report' page to see your results.")
                    st.experimental_rerun()
            else:
                st.success("Quiz has been submitted. Go to 'View Report' page to see your results.")
        else:
            st.warning("No quiz available. Please generate a quiz first.")
    
    # View Report Page
    elif page == "View Report":
        st.subheader("📊 Proctoring Report")
        
        if st.session_state.report_data and st.session_state.quiz_submitted:
            report_data = st.session_state.report_data
            quiz_data = st.session_state.quiz_data
            
            st.markdown("<div class='report-container'>", unsafe_allow_html=True)
            
            # Quiz results
            st.markdown("### Quiz Results")
            
            # Calculate score
            correct_answers = 0
            for q in quiz_data['questions']:
                q_id = q['id']
                if q_id in st.session_state.user_answers:
                    user_answer = st.session_state.user_answers[q_id]
                    correct_option = q['options'][ord(q['correct_answer']) - ord('A')]
                    if user_answer == correct_option:
                        correct_answers += 1
            
            score_percentage = (correct_answers / len(quiz_data['questions'])) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Questions", len(quiz_data['questions']))
            with col2:
                st.metric("Correct Answers", correct_answers)
            with col3:
                st.metric("Score", f"{score_percentage:.1f}%")
            
            # Show detailed answers
            with st.expander("View Detailed Answers"):
                for i, q in enumerate(quiz_data['questions']):
                    q_id = q['id']
                    user_answer = st.session_state.user_answers.get(q_id, "Not answered")
                    correct_option = q['options'][ord(q['correct_answer']) - ord('A')]
                    
                    st.markdown(f"**Q{i+1}: {q['question']}**")
                    st.markdown(f"Your answer: {user_answer}")
                    st.markdown(f"Correct answer: {correct_option}")
                    
                    if user_answer == correct_option:
                        st.success("Correct! ✓")
                    else:
                        st.error("Incorrect! ✗")
                    
                    st.markdown("---")
            
            # Proctoring metrics
            st.markdown("### Proctoring Metrics")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Exam Duration", f"{report_data['total_exam_time_minutes']:.2f} minutes")
                st.metric("Face Visibility", f"{report_data['face_visibility_percentage']:.1f}%")
                st.metric("No Face Instances", report_data['no_face_detected_instances'])
            with col2:
                st.metric("Multiple Faces Detected", report_data['multiple_faces_detected_instances'])
                st.metric("Blink Rate", f"{report_data['blink_rate_per_minute']:.1f} per minute")
                st.metric("Tab Switches", report_data['tab_switches'])
            
            # Risk assessment
            st.markdown("### Risk Assessment")
            risk_level = report_data['potential_cheating_risk']
            
            if risk_level == "Low":
                st.success(f"Potential Cheating Risk: {risk_level}")
                st.markdown("✅ No significant suspicious activity detected during the exam.")
            elif risk_level == "Medium":
                st.warning(f"Potential Cheating Risk: {risk_level}")
                st.markdown("⚠️ Some suspicious activities were detected. Please review the metrics for more information.")
            else:
                st.error(f"Potential Cheating Risk: {risk_level}")
                st.markdown("❌ Significant suspicious activities were detected during the exam!")
            
            # Visualization of proctoring data
            st.markdown("### Proctoring Data Visualization")
            
            # Create dataframe for visualization
            if st.session_state.proctoring_data['timestamps']:
                df = pd.DataFrame({
                    'timestamp': st.session_state.proctoring_data['timestamps'],
                    'face_count': st.session_state.proctoring_data['face_counts']
                })
                
                # Sample the dataframe if it's too large
                if len(df) > 100:
                    df = df.iloc[::len(df) // 100]
                
                # Plot face count over time
                st.line_chart(df.set_index('timestamp')['face_count'])
                st.caption("Face Count Over Time")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        elif st.session_state.quiz_submitted:
            st.info("Generating report... Please wait.")
        else:
            st.warning("No report available. Please complete a quiz first.")
    
    # Add footer
    st.markdown("---")
    st.markdown(
        "Developed by AI Proctoring Systems | © 2025 | "
        "This application uses computer vision for proctoring and AI for quiz generation."
    )

if __name__ == "__main__":
    main()
