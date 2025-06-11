from celery import Celery
import os
from backend.service.detection_service import detect_from_video
from backend.service.video_service import (
    process_video_segment,
    process_video_classify,
    process_video_pose
)

# Celery configuration
celery_app = Celery(
    "worker",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
)

@celery_app.task
def async_detect(video_path: str, output_dir: str, model_path: str = "yolov5s.pt"):
    return detect_from_video(video_path, output_dir, model_path)

@celery_app.task
def async_segment(video_path: str, output_dir: str, model_path: str = "yolov8n-seg.pt"):
    return process_video_segment(video_path, output_dir, model_path)

@celery_app.task
def async_classify(video_path: str, output_dir: str, model_path: str = "resnet18.pt"):
    return process_video_classify(video_path, output_dir, model_path)

@celery_app.task
def async_pose(video_path: str, output_dir: str, model_path: str = "yolov8n-pose.pt"):
    return process_video_pose(video_path, output_dir, model_path)
