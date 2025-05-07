import streamlit as st
import time
import datetime
import os
import json
import requests
import base64
import pandas as pd
from dotenv import load_dotenv
import threading
import sys
import numpy as np
from PIL import Image
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="AI Proctoring System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Initialize global variables for proctoring
face_detection = None
face_mesh = None
mp_drawing = None

# Try to import OpenCV and MediaPipe
try:
    import cv2
    import mediapipe as mp
    
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False
    st.sidebar.warning("OpenCV or MediaPipe couldn't be imported. Camera-based proctoring features will be limited.")

# Try to import streamlit_javascript
try:
    import streamlit_javascript as st_js
    JS_AVAILABLE = True
except ImportError:
    JS_AVAILABLE = False
    st.sidebar.warning("streamlit-javascript couldn't be imported. Tab switching detection will be disabled.")

# Initialize Groq client
groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = None
GROQ_MODEL = "llama3-8b-8192"

try:
    from groq import Groq
    if groq_api_key:
        groq_client = Groq(api_key=groq_api_key)
    else:
        st.sidebar.error("GROQ_API_KEY not found. Please add it to your .env file or Streamlit secrets.")
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
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
if 'blink_state' not in st.session_state:
    st.session_state.blink_state = 'OPEN'

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
    if groq_client is None:
        st.warning("Groq API key not configured. Using demo quiz data.")
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
        
        import re
        json_match = re.search(r'```json\n([\s\S]*?)\n```', response_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response_text
            
        start_idx = json_str.find('{')
        end_idx = json_str.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            json_str = json_str[start_idx:end_idx]
            
        quiz_data = json.loads(json_str)
        return quiz_data
    
    except Exception as e:
        st.error(f"Error generating quiz: {str(e)}")
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
    """Detect eye blinks using facial landmarks with state machine"""
    if not CV_AVAILABLE or not landmarks:
        return False
        
    try:
        # Corrected landmark indices for left and right eyes
        left_eye = [
            landmarks[362],  # p1: left
            landmarks[386],  # p2: top left
            landmarks[387],  # p3: top right
            landmarks[385],  # p4: right
            landmarks[380],  # p5: bottom right
            landmarks[374]   # p6: bottom left
        ]
        
        right_eye = [
            landmarks[33],   # p1: right
            landmarks[159],  # p2: top right
            landmarks[158],  # p3: top left
            landmarks[133],  # p4: left
            landmarks[145],  # p5: bottom left
            landmarks[144]   # p6: bottom right
        ]
        
        def calculate_ear(eye_points):
            height_1 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
            height_2 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
            width = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
            ear = (height_1 + height_2) / (2.0 * width)
            return ear
        
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        
        EAR_THRESHOLD = 0.2
        OPEN_THRESHOLD = 0.3
        
        current_time = time.time()
        
        # State machine for blink detection
        if st.session_state.blink_state == 'OPEN' and left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
            st.session_state.blink_state = 'CLOSED'
            st.session_state.last_blink_check = current_time
        elif st.session_state.blink_state == 'CLOSED' and left_ear > OPEN_THRESHOLD and right_ear > OPEN_THRESHOLD:
            if current_time - st.session_state.last_blink_check < 0.5:  # Ensure blink duration is reasonable
                st.session_state.blink_counter += 1
                st.session_state.blink_state = 'OPEN'
                st.session_state.last_blink_check = current_time
                return True
            st.session_state.blink_state = 'OPEN'
        
        return False
    
    except Exception as e:
        st.error(f"Error in blink detection: {str(e)}")
        return False

def process_webcam_feed():
    """Process webcam feed to detect faces and eye blinks"""
    if not st.session_state.monitoring_active or not CV_AVAILABLE:
        return
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam. Please check your camera connection.")
            return
        
        stframe = st.empty()
        
        while st.session_state.monitoring_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to get frame from webcam.")
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = face_detection.process(rgb_frame)
            mesh_results = face_mesh.process(rgb_frame)
            
            face_count = 0
            if face_results.detections:
                face_count = len(face_results.detections)
                for detection in face_results.detections:
                    mp_drawing.draw_detection(frame, detection)
            
            landmarks_list = []
            if mesh_results.multi_face_landmarks:
                for face_landmarks in mesh_results.multi_face_landmarks:
                    landmarks = []
                    for landmark in face_landmarks.landmark:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        landmarks.append((x, y))
                    
                    landmarks_list.append(landmarks)
                    
                    face_oval = [
                        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
                    ]
                    
                    blink_detected = detect_blinks(landmarks, face_oval)
                    
                    mp_drawing.draw_landmarks(
                        frame, 
                        face_landmarks,
                        mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
                    )
            
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.proctoring_data['face_counts'].append(face_count)
            st.session_state.proctoring_data['blink_counts'].append(st.session_state.blink_counter)
            st.session_state.proctoring_data['timestamps'].append(timestamp)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            info_text = f"Faces: {face_count} | Blinks: {st.session_state.blink_counter} | Tab Switches: {st.session_state.proctoring_data['tab_switches']}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if face_count > 1:
                warning_text = "WARNING: Multiple faces detected!"
                cv2.putText(frame, warning_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if face_count == 0:
                warning_text = "WARNING: No face detected!"
                cv2.putText(frame, warning_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            stframe.image(frame, channels="RGB", use_column_width=True)
            
            if st.session_state.start_time is not None:
                st.session_state.proctoring_data['time_on_camera'] = time.time() - st.session_state.start_time
            
            time.sleep(0.5)  # Reduced frequency to optimize performance
    
    except Exception as e:
        st.error(f"Error in webcam processing: {str(e)}")
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()

def detect_tab_switch():
    """Detect tab switching using JavaScript"""
    if not st.session_state.monitoring_active or not JS_AVAILABLE:
        return
        
    try:
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
            st.warning("Tab switch detected! Please focus on the exam.")
        elif not is_hidden:
            st.session_state.tab_switch_detected = False
    except Exception as e:
        pass

def generate_report():
    """Generate a comprehensive proctoring report"""
    if not st.session_state.proctoring_data:
        return None
    
    total_time = st.session_state.proctoring_data['time_on_camera']
    face_counts = st.session_state.proctoring_data['face_counts']
    no_face_count = face_counts.count(0)
    multiple_face_instances = sum(1 for count in face_counts if count > 1)
    
    time_per_frame = total_time / len(face_counts) if len(face_counts) > 0 else 0
    time_without_face = no_face_count * time_per_frame
    
    face_visibility_percentage = (1 - (time_without_face / total_time)) * 100 if total_time > 0 else 0
    
    # Adjust blink rate to exclude no-face periods
    valid_frames = len([count for count in face_counts if count == 1])
    valid_time = valid_frames * time_per_frame if valid_frames > 0 else 0
    blink_rate = st.session_state.blink_counter / (valid_time / 60) if valid_time > 0 else 0
    
    report = {
        'total_exam_time_minutes': round(total_time / 60, 2),
        'face_visibility_percentage': round(face_visibility_percentage, 2),
        'no_face_detected_instances': no_face_count,
        'multiple_faces_detected_instances': multiple_face_instances,
        'total_blinks': st.session_state.blink_counter,
        'blink_rate_per_minute': round(blink_rate, 2),
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
    total_time_minutes = total_time / 60
    risk_score = 0
    
    if face_visibility < 90:
        risk_score += (90 - face_visibility) * 0.5
    risk_score += multiple_faces * 20
    risk_score += tab_switches * 10
    risk_score = risk_score / max(total_time_minutes, 1)
    risk_score = min(risk_score, 100)
    
    if risk_score < 20:
        return "Low"
    elif risk_score < 50:
        return "Medium"
    else:
        return "High"

def main():
    st.markdown("<h1 class='header'>AI-Based Proctoring System with Quiz Generation</h1>", unsafe_allow_html=True)
    
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
                    threading.Thread(target=process_webcam_feed, daemon=True).start()
        
        elif page == "View Report":
            st.subheader("Report Information")
            if st.session_state.report_data:
                st.success("Report generated successfully")
            else:
                st.warning("No report available yet")

    if page == "Setup Quiz":
        st.subheader("📝 Setup Quiz Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            topic = st.text_input("Quiz Topic", value="Python Programming")
            difficulty = st.select_slider("Difficulty Level", options=["Easy", "Medium", "Hard"])
        
        with col2:
            num_questions = st.number_inputNUMBER OF QUESTIONS", min_value=1, max_value=20, value=5)
            time_limit = st.number_input("Time Limit (minutes)", min_value=1, max_value=180, value=10)
        
        if st.button("Generate Quiz"):
            with st.spinner("Generating quiz using Groq API..."):
                quiz_data = generate_quiz(topic, difficulty, num_questions, time_limit)
                
                if quiz_data:
                    st.session_state.quiz_data = quiz_data
                    st.session_state.user_answers = {}
                    st.success("Quiz generated successfully! Go to 'Take Quiz' page to start.")
                    
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

    elif page == "Take Quiz":
        st.subheader("📝 Take Quiz")
        
        detect_tab_switch()
        
        if st.session_state.quiz_data:
            quiz_data = st.session_state.quiz_data
            
            st.markdown(f"### {quiz_data['title']}")
            st.markdown(quiz_data['description'])
            
            if st.session_state.start_time is not None:
                elapsed_time = time.time() - st.session_state.start_time
                remaining_time = max(0, quiz_data['time_limit_minutes'] * 60 - elapsed_time)
                minutes, seconds = divmod(int(remaining_time), 60)
                st.markdown(f"**Time Remaining:** {minutes:02d}:{seconds:02d}")
                
                if remaining_time <= 0 and not st.session_state.quiz_submitted:
                    st.warning("Time's up! Quiz has been automatically submitted.")
                    st.session_state.quiz_submitted = True
                    st.session_state.monitoring_active = False
                    st.session_state.report_data = generate_report()
                    st.experimental_rerun()
            
            if not st.session_state.quiz_submitted:
                # Paginate questions
                current_question = st.session_state.get('current_question', 0)
                total_questions = len(quiz_data['questions'])
                
                if current_question < total_questions:
                    question = quiz_data['questions'][current_question]
                    q_id = question['id']
                    st.markdown(f"**Question {current_question + 1} of {total_questions}: {question['question']}**")
                    
                    options = question['options']
                    st.session_state.user_answers[q_id] = st.radio(
                        f"Select answer for question {current_question + 1}",
                        options=options,
                        key=f"q_{q_id}",
                        index=options.index(st.session_state.user_answers.get(q_id, options[0])) if q_id in st.session_state.user_answers else 0
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if current_question > 0 and st.button("Previous"):
                            st.session_state.current_question = current_question - 1
                            st.experimental_rerun()
                    with col2:
                        if current_question < total_questions - 1 and st.button("Next"):
                            st.session_state.current_question = current_question + 1
                            st.experimental_rerun()
                        elif current_question == total_questions - 1 and st.button("Submit Quiz"):
                            st.session_state.quiz_submitted = True
