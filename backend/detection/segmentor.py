import torch
import cv2
import numpy as np

class Segmentor:
    def __init__(self, model_path="yolov8n-seg.pt", device="cpu"):
        self.device = device
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path).to(device)

    def segment_and_mask(self, img_np):
        results = self.model(img_np)
        masks = results.masks  # if using YOLOv8-seg or custom logic
        annotated = results.render()[0]  # returns image with masks drawn
        return annotated, results