import os
import uuid
import shutil
import numpy as np
import cv2

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from backend.detection.yolov5 import YOLOv5Detector
from backend.detection.segmentor import Segmentor
from backend.detection.classifier import ImageClassifier
from backend.detection.pose_estimator import PoseEstimator

from backend.db.session import SessionLocal
from backend.db.models import Detection

app = FastAPI(title="YOLO Detection + Segmentation Platform")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
detector = YOLOv5Detector("yolov5s.pt")
segmentor = Segmentor("yolov8n-seg.pt")
classifier = ImageClassifier("resnet18.pt")  # adjust model path accordingly
pose_estimator = PoseEstimator("yolov8n-pose.pt")

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type.startswith("image"):
        raise HTTPException(status_code=400, detail="Invalid image file.")
    image_id = uuid.uuid4()
    img_path = f"data/{image_id}.jpg"
    result_path = f"data/{image_id}_result.jpg"
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    class_names, confs, boxes = detector.detect_and_save(img_path, result_path)
    db = SessionLocal()
    detection = Detection(
        id=image_id,
        filename=img_path,
        result_path=result_path,
        class_names=class_names,
        confidences=confs,
        bboxes=boxes
    )
    db.add(detection)
    db.commit()
    db.close()
    return JSONResponse(content={"result_path": result_path, "class_names": class_names})

@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    img_bytes = await file.read()
    np_img = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    try:
        mask_img, _ = segmentor.segment_and_mask(frame)
        _, buffer = cv2.imencode(".jpg", mask_img)
        return Response(content=buffer.tobytes(), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {e}")

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    img_bytes = await file.read()
    label = classifier.classify(img_bytes)
    return JSONResponse(content={"label": label})

@app.post("/pose")
async def pose(file: UploadFile = File(...)):
    img_bytes = await file.read()
    pose_img = pose_estimator.estimate_pose(img_bytes)
    _, buffer = cv2.imencode(".jpg", pose_img)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

@app.get("/")
def root():
    return {"message": "YOLO Detection + Segmentation API is live."}

@app.get("/health")
def health():
    return {"status": "ok"}
