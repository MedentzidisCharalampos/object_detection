import streamlit as st
import requests
from PIL import Image
import io
import cv2
import tempfile
import pandas as pd
from sqlalchemy import create_engine
import os
st.set_page_config(page_title="YOLO Object Detection", layout="centered")

st.title("📦 YOLO Object Detection")

tab1, tab2, tab3 = st.tabs(["📁 Image Upload", "📷 Webcam Detection", "📊 Detection History"])

# ---- Tab 1: Upload Image ----
with tab1:
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image (JPG or PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with st.spinner("Sending image to server..."):
            image_bytes = uploaded_file.read()
            files = {"file": ("image.jpg", image_bytes, uploaded_file.type)}
            try:
                response = requests.post("http://localhost:8000/detect", files=files)
                if response.ok:
                    result = response.json()
                    st.success("✅ Detection Complete")
                    st.image(result["result_path"], caption="Detected Image")
                    st.write("**Detected Classes:**", result["class_names"])
                else:
                    st.error(f"❌ Server error: {response.status_code}")
                    st.code(response.text)
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Connection failed: {e}")

# ---- Tab 2: Webcam Detection ----
with tab2:
    st.subheader("Live Webcam Detection")
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

            # Send image to API
            try:
                files = {
                    "file": ("frame.jpg", byte_img, "image/jpeg")
                }
                response = requests.post("http://localhost:8000/detect", files=files)
                if response.ok:
                    result = response.json()
                    # Use the image returned by server
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
