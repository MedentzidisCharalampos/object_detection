from ultralytics import YOLO
import cv2

class Detector:
    def __init__(self, model_path: str):
        """
        Initialize the YOLOv5 detector using the Ultralytics YOLO wrapper.
        """
        self.model = YOLO(model_path)

    def detect_and_save(self, input_path: str, output_path: str):
        """
        Run detection from an image file path and save the result to disk.
        """
        results = self.model(input_path)
        result = results[0]
        result.save(filename=output_path)

        boxes = result.boxes.xyxy.tolist()
        class_ids = result.boxes.cls.tolist()
        confidences = result.boxes.conf.tolist()
        class_names = [self.model.names[int(c)] for c in class_ids]

        return class_names, confidences, boxes

    def process(self, img):
        """
        Run detection directly on a given OpenCV image and return annotated frame and results.

        Args:
            img (np.ndarray): OpenCV image (BGR format)

        Returns:
            Tuple[np.ndarray, List]: Annotated image and results
        """
        results = self.model(img)
        result = results[0]
        annotated = result.plot()
        return annotated, results

