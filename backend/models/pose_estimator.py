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

    def estimate_pose(self, img):
        """
        Estimate human poses from an OpenCV image.

        Args:
            img (np.ndarray): Input OpenCV image

        Returns:
            Tuple[np.ndarray, List]: Annotated image and results
        """
        results = self.model(img)
        annotated = results[0].plot()
        return annotated, results

    def process(self, img):
        """
        Unified method for compatibility with handle_image().
        """
        return self.estimate_pose(img)
