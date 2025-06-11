import streamlit as st
import requests
from PIL import Image
import cv2
import tempfile
import pandas as pd
from sqlalchemy import create_engine
import os

st.set_page_config(page_title="YOLO Multi-Task App", layout="centered")
st.title("🧠 YOLO Detection, Segmentation & Pose Estimation")

tabs = ["📁 Image Upload", "📷 Webcam Detection", "📊 Detection History", "📼 Video Upload"]
tab1, tab2, tab3, tab4 = st.tabs(tabs)

def get_endpoint(task_type, is_video=False):
    prefix = "/video" if is_video else ""
    return f"http://localhost:8000{prefix}/" + {
        "Object Detection": "detect",
        "Segmentation": "segment",
        "Pose Estimation": "pose"
    }[task_type]

# ---- Tab 1: Upload Image ----
with tab1:
    st.subheader("Upload an Image")
    task_type = st.selectbox("Select Task", ["Object Detection", "Segmentation", "Pose Estimation"], key="image_task")
    uploaded_file = st.file_uploader("Choose an image (JPG or PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        with st.spinner("Sending image to server..."):
            files = {"file": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}
            try:
                response = requests.post(get_endpoint(task_type), files=files)
                if response.ok:
                    result = response.json()
                    st.success("✅ Task Complete")
                    st.image(result["result_path"], caption="Result Image")
                    st.write("**Detected Classes:**", result["class_names"])
                else:
                    st.error(f"❌ Server error: {response.status_code}")
                    st.code(response.text)
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Connection failed: {e}")

# ---- Tab 2: Webcam Detection ----
with tab2:
    st.subheader("Live Webcam Detection")
    task_type = st.selectbox("Select Task", ["Object Detection", "Segmentation", "Pose Estimation"], key="webcam_task")
    run_webcam = st.button("🎥 Start Webcam")

    if run_webcam:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        st.info("Press 'Stop' to terminate the webcam stream.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("❌ Failed to read from webcam.")
                break

            _, buffer = cv2.imencode(".jpg", frame)
            byte_img = buffer.tobytes()

            try:
                files = {"file": ("frame.jpg", byte_img, "image/jpeg")}
                response = requests.post(get_endpoint(task_type), files=files)
                if response.ok:
                    result = response.json()
                    img_path = result["result_path"]
                    if os.path.exists(img_path):
                        img = cv2.imread(img_path)
                        stframe.image(img, channels="BGR", use_column_width=True)
                    else:
                        st.warning("⚠️ Result image not found.")
                else:
                    st.error("API returned an error: " + response.text)
            except Exception as e:
                st.error(f"Exception: {e}")
                break

        cap.release()

# ---- Tab 3: Detection History ----
with tab3:
    st.subheader("Detection History (SQLite)")
    try:
        engine = create_engine("sqlite:///detections.db")
        df = pd.read_sql("SELECT timestamp, filename, class_names FROM detections ORDER BY timestamp DESC", engine)
        if df.empty:
            st.info("No detections yet.")
        else:
            st.dataframe(df)
    except Exception as e:
        st.error("Database not available or not populated yet.")
        st.code(str(e))

# ---- Tab 4: Upload Video ----
with tab4:
    st.subheader("Upload a Video")
    task_type = st.selectbox("Select Task", ["Object Detection", "Segmentation", "Pose Estimation"], key="video_task")
    video_file = st.file_uploader("Choose a video file (MP4/AVI)", type=["mp4", "avi", "mov"])

    if video_file:
        with st.spinner("Uploading and processing video..."):
            try:
                files = {"file": (video_file.name, video_file.read(), video_file.type)}
                response = requests.post(get_endpoint(task_type, is_video=True), files=files)

                if response.ok:
                    result = response.json()
                    st.success("✅ Video processed.")
                    st.write(f"Frames processed: {result.get('frame_count', '?')}")

                    video_path = result.get("video_path")
                    frame_paths = result.get("frame_paths", [])

                    if video_path and os.path.exists(video_path):
                        with open(video_path, 'rb') as f:
                            st.video(f.read())
                    elif frame_paths:
                        st.write("Displaying processed frames:")
                        for path in frame_paths:
                            if os.path.exists(path):
                                st.image(path, use_column_width=True)
                    else:
                        st.warning("⚠️ No video or frames available.")
                else:
                    st.error("❌ Server returned error.")
                    st.code(response.text)
            except Exception as e:
                st.error("❌ Failed to upload or process video.")
                st.code(str(e))
