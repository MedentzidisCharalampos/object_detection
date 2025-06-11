from ultralytics import YOLO
import cv2
import numpy as np

class PoseEstimator:
    def __init__(self, model_path="yolov8n-pose.pt", device="cpu"):
        """
        Load YOLOv8 pose estimation model.
        """
        self.model = YOLO(model_path)
        self.device = device

    def estimate_pose(self, img_bytes):
        """
        Estimate human poses from image bytes.

        Args:
            img_bytes (bytes): Input image content

        Returns:
            np.ndarray: Annotated image with pose keypoints
        """
        np_img = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        results = self.model(frame)
        annotated = results[0].plot()
        return annotated