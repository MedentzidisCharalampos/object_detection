from ultralytics import YOLO
import cv2
import torch

class Segmentor:
    def __init__(self, model_path="yolov8n-seg.pt", device="cpu"):
        """
        Initialize the YOLOv8 segmentation model.

        Args:
            model_path (str): Path to the YOLOv8 segmentation model.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        self.device = device
        self.model = YOLO(model_path)
        self.model.to(torch.device(device))

    def segment_and_mask(self, frame):
        """
        Perform instance segmentation and return annotated frame.

        Args:
            frame (np.ndarray): Input image in BGR format.

        Returns:
            tuple: (annotated image, results object)
        """
        # Convert BGR to RGB for model inference
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(rgb_frame)
        annotated = results[0].plot()
        return annotated, results

    def process(self, img):
        """
        Compatibility method for handle_image().
        """
        return self.segment_and_mask(img)
