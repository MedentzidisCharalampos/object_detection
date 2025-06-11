import cv2
import os
from backend.detection.yolov5 import YOLOv5Detector

def detect_from_video(video_path, output_dir, model_path="yolov5s.pt"):
    """
    Run YOLO object detection on each frame of the input video.

    Args:
        video_path (str): Path to video file.
        output_dir (str): Directory to save frame outputs.
        model_path (str): Path to YOLO model.

    Returns:
        str: Message indicating how many frames were processed.
    """
    detector = YOLOv5Detector(model_path)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    os.makedirs(output_dir, exist_ok=True)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(output_dir, f"frame_{frame_idx}.jpg")
        cv2.imwrite(frame_path, frame)

        output_path = os.path.join(output_dir, f"result_{frame_idx}.jpg")
        detector.detect_and_save(frame_path, output_path)

        frame_idx += 1

    cap.release()
    return f"Processed {frame_idx} frames."
