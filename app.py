import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import mediapipe as mp

# Page configuration
st.set_page_config(
    page_title="Moving Object & Hand Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 10px;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)


def get_background(video_path):
    """Calculate median background from video frames."""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Randomly select 50 frames for calculating median
    frame_indices = frame_count * np.random.uniform(size=min(50, frame_count))
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    
    if len(frames) > 0:
        median_frame = np.median(frames, axis=0).astype(np.uint8)
        return median_frame
    return None


def detect_moving_objects(video_path, consecutive_frames=4, progress_bar=None):
    """Detect moving objects in video using frame differencing."""
    cap = cv2.VideoCapture(video_path)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create temporary output file
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_output.name
    temp_output.close()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Get background model
    background = get_background(video_path)
    if background is None:
        cap.release()
        return None
    
    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    
    frame_count = 0
    frame_diff_list = []
    
    # Reset video to beginning
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        orig_frame = frame.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if frame_count % consecutive_frames == 0 or frame_count == 1:
            frame_diff_list = []
        
        # Find difference between current frame and background
        frame_diff = cv2.absdiff(gray, background)
        
        # Thresholding
        ret, thres = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)
        
        # Dilate
        dilate_frame = cv2.dilate(thres, None, iterations=2)
        frame_diff_list.append(dilate_frame)
        
        if len(frame_diff_list) == consecutive_frames:
            sum_frames = sum(frame_diff_list)
            contours, _ = cv2.findContours(sum_frames, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) < 500:
                    continue
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(orig_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            out.write(orig_frame)
        
        # Update progress
        if progress_bar:
            progress_bar.progress(frame_count / total_frames)
    
    cap.release()
    out.release()
    
    return output_path


def detect_hand_fingers(video_path, progress_bar=None):
    """Detect hands and count fingers in video."""
    cap = cv2.VideoCapture(video_path)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create temporary output file
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_output.name
    temp_output.close()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    
    finger_coords = [(8, 6), (12, 10), (16, 14), (20, 18)]
    thumb_coord = (4, 2)
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get landmark positions
                hand_list = []
                for idx, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    hand_list.append((cx, cy))
                
                # Count fingers
                up_count = 0
                for coord in finger_coords:
                    if hand_list[coord[0]][1] < hand_list[coord[1]][1]:
                        up_count += 1
                
                if hand_list[thumb_coord[0]][0] > hand_list[thumb_coord[1]][0]:
                    up_count += 1
                
                # Display count
                cv2.putText(frame, str(up_count), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
        
        out.write(frame)
        
        # Update progress
        if progress_bar:
            progress_bar.progress(frame_count / total_frames)
    
    cap.release()
    out.release()
    hands.close()
    
    return output_path


# Sidebar navigation
st.sidebar.markdown("## 🎯 Navigation")
page = st.sidebar.radio("Select Feature", ["🏠 Home", "🎬 Moving Object Detection", "✋ Hand & Finger Counting"])

if page == "🏠 Home":
    st.markdown('<h1 class="main-header">Moving Object & Hand Detection</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            A computer vision project demonstrating real-time object detection using 
            <strong>OpenCV</strong> and <strong>MediaPipe</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🎬 Moving Object Detection</h3>
            <p>Detects moving objects in videos using background subtraction and frame differencing technique</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>✋ Hand & Finger Counting</h3>
            <p>Tracks hand landmarks and counts the number of raised fingers using MediaPipe</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    ### 🛠️ Technologies Used
    - **OpenCV** - Computer Vision library for image/video processing
    - **MediaPipe** - ML solutions for hand tracking
    - **Streamlit** - Web application framework
    - **NumPy** - Numerical computing
    """)

elif page == "🎬 Moving Object Detection":
    st.markdown('<h1 class="main-header">🎬 Moving Object Detection</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Upload a video to detect and highlight moving objects using **background subtraction** 
    and **frame differencing** technique.
    """)
    
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
    
    col1, col2 = st.columns(2)
    with col1:
        consecutive_frames = st.slider("Consecutive Frames", min_value=2, max_value=10, value=4, 
                                       help="Number of frames to accumulate for detection")
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_input.write(uploaded_file.read())
        temp_input.close()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📥 Original Video")
            st.video(temp_input.name)
        
        if st.button("🚀 Detect Moving Objects", use_container_width=True):
            with col2:
                st.markdown("### 📤 Processed Video")
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("Processing video...")
                
                output_path = detect_moving_objects(temp_input.name, consecutive_frames, progress_bar)
                
                if output_path and os.path.exists(output_path):
                    status_text.text("✅ Processing complete!")
                    st.video(output_path)
                    
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download Processed Video",
                            data=f,
                            file_name="detected_objects.mp4",
                            mime="video/mp4"
                        )
                    
                    # Cleanup
                    os.unlink(output_path)
                else:
                    st.error("Error processing video. Please try again.")
        
        # Cleanup input file
        os.unlink(temp_input.name)

elif page == "✋ Hand & Finger Counting":
    st.markdown('<h1 class="main-header">✋ Hand & Finger Counting</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Upload a video with hand gestures to detect hands and count raised fingers using **MediaPipe**.
    """)
    
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_input.write(uploaded_file.read())
        temp_input.close()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📥 Original Video")
            st.video(temp_input.name)
        
        if st.button("🚀 Detect Hands & Count Fingers", use_container_width=True):
            with col2:
                st.markdown("### 📤 Processed Video")
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("Processing video...")
                
                output_path = detect_hand_fingers(temp_input.name, progress_bar)
                
                if output_path and os.path.exists(output_path):
                    status_text.text("✅ Processing complete!")
                    st.video(output_path)
                    
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download Processed Video",
                            data=f,
                            file_name="hand_detection.mp4",
                            mime="video/mp4"
                        )
                    
                    # Cleanup
                    os.unlink(output_path)
                else:
                    st.error("Error processing video. Please try again.")
        
        # Cleanup input file
        os.unlink(temp_input.name)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; color: #888; font-size: 0.8rem;">
    Made with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)
