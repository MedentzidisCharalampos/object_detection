import cv2
import os
import uuid
import numpy as np
from backend.models.detector import Detector
from backend.models.segmentor import Segmentor
from backend.models.pose_estimator import PoseEstimator


def detect_from_video(video_path, output_dir, model_path="models/yolov5s.pt"):
    """
    Run YOLO object detection on each frame of the input video.
    """
    detector = Detector(model_path)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    os.makedirs(output_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output_path = os.path.join(output_dir, f"result_{frame_idx}.jpg")
        detector.detect_and_save_array(frame, output_path)  # Modified: detects from array
        frame_idx += 1

    cap.release()
    return f"Processed {frame_idx} frames."


def handle_image(image_bytes: bytes, task: str):
    """
    Handle image processing for detection, segmentation, or pose estimation.
    """
    # Decode image bytes to OpenCV format
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes")

    # Select the appropriate model
    if task == "detect":
        model = Detector("models/yolov5s.pt")
    elif task == "segment":
        model = Segmentor("models/yolov8n-seg.pt")
    elif task == "pose":
        model = PoseEstimator("models/yolov8n-pose.pt")
    else:
        raise ValueError(f"Unsupported task: {task}")

    # Run inference
    annotated, results = model.process(img)

    # Save result
    os.makedirs("results", exist_ok=True)
    output_path = f"results/{uuid.uuid4().hex}.jpg"
    cv2.imwrite(output_path, annotated)

    # Extract class names if available
    class_names = []
    if hasattr(results[0], "boxes") and hasattr(results[0].boxes, "cls"):
        class_names = [str(cls_id) for cls_id in results[0].boxes.cls.tolist()]

    return {
        "result_path": output_path,
        "class_names": class_names
    }
