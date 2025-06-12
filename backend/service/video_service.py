import cv2
import os
import tempfile
import uuid
from fastapi import UploadFile
from backend.models.detector import Detector
from backend.models.segmentor import Segmentor
from backend.models.pose_estimator import PoseEstimator


def process_video_detect(video_path, output_dir, model_path="models/yolov8n.pt"):
    detector = Detector(model_path)

    def detect_fn(frame):
        annotated, _ = detector.process(frame)
        return annotated, None

    return _process_video(video_path, output_dir, detect_fn, suffix="detect")


def process_video_segment(video_path, output_dir, model_path="models/yolov8n-seg.pt"):
    segmentor = Segmentor(model_path)

    def segment_fn(frame):
        annotated, _ = segmentor.segment_and_mask(frame)
        return annotated, None

    return _process_video(video_path, output_dir, segment_fn, suffix="segment")


def process_video_pose(video_path, output_dir, model_path="models/yolov8n-pose.pt"):
    estimator = PoseEstimator(model_path)

    def estimate_fn(frame):
        pose_img, _ = estimator.estimate_pose(frame)
        return pose_img, None

    return _process_video(video_path, output_dir, estimate_fn, suffix="pose")


def _process_video(video_path, output_dir, processing_fn, suffix="processed", show=False):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    os.makedirs(output_dir, exist_ok=True)
    video_filename = f"{suffix}_processed.avi"
    video_output_path = os.path.join(output_dir, video_filename)

    # Video writer init
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result_frame, _ = processing_fn(frame)
        writer.write(result_frame)

        if show:
            cv2.imshow("Result", result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    return {
        "message": f"{suffix.capitalize()} completed for {frame_idx} frames.",
        "frame_count": frame_idx,
        "video_path": video_output_path
    }


async def handle_video(file: UploadFile, task: str):
    suffix = os.path.splitext(file.filename)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_video_path = tmp.name

    output_dir = os.path.join("video_results", uuid.uuid4().hex)
    os.makedirs(output_dir, exist_ok=True)

    if task == "detect":
        return process_video_detect(temp_video_path, output_dir)
    elif task == "segment":
        return process_video_segment(temp_video_path, output_dir)
    elif task == "pose":
        return process_video_pose(temp_video_path, output_dir)
    else:
        return {
            "status": "error",
            "message": f"Unsupported task type: {task}"
        }
