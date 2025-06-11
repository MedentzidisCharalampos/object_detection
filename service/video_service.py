import cv2
import os
from backend.detection.segmentor import Segmentor
from backend.detection.classifier import ImageClassifier
from backend.detection.pose_estimator import PoseEstimator

def process_video_segment(video_path, output_dir, model_path="yolov8n-seg.pt"):
    segmentor = Segmentor(model_path)
    return _process_video(video_path, output_dir, segmentor.segment_and_mask, suffix="segment")

def process_video_classify(video_path, output_dir, model_path="resnet18.pt"):
    classifier = ImageClassifier(model_path)

    def classify_frame(frame):
        _, buffer = cv2.imencode(".jpg", frame)
        label = classifier.classify(buffer.tobytes())
        annotated = cv2.putText(frame.copy(), label, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return annotated, label

    return _process_video(video_path, output_dir, classify_frame, suffix="classify")

def process_video_pose(video_path, output_dir, model_path="yolov8n-pose.pt"):
    estimator = PoseEstimator(model_path)

    def estimate(frame):
        _, buffer = cv2.imencode(".jpg", frame)
        pose_img = estimator.estimate_pose(buffer.tobytes())
        return pose_img, None

    return _process_video(video_path, output_dir, estimate, suffix="pose")

def _process_video(video_path, output_dir, processing_fn, suffix="processed"):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    os.makedirs(output_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result_frame, _ = processing_fn(frame)
        out_path = os.path.join(output_dir, f"{suffix}_{frame_idx}.jpg")
        cv2.imwrite(out_path, result_frame)
        frame_idx += 1

    cap.release()
    return f"{suffix.capitalize()} completed for {frame_idx} frames."
