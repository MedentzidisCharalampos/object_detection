from ultralytics import YOLO
import os

class YOLOv5Detector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_and_save(self, input_path: str, output_path: str):
        results = self.model(input_path)
        result = results[0]
        result.save(filename=output_path)
        boxes = result.boxes.xyxy.tolist()
        class_ids = result.boxes.cls.tolist()
        confidences = result.boxes.conf.tolist()
        class_names = [self.model.names[int(c)] for c in class_ids]
        return class_names, confidences, boxes
