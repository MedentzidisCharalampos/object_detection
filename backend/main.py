from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.models.detector import Detector
from backend.models.segmentor import Segmentor
from backend.models.pose_estimator import PoseEstimator
from backend.service.detection_service import handle_image
from backend.service.video_service import handle_video
from starlette.responses import JSONResponse
import uuid, os
import shutil

app = FastAPI(title="YOLO Multi-Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = Detector("models/yolov5s.pt")
segmentor = Segmentor("models/yolov8n-seg.pt")
pose_model = PoseEstimator("models/yolov8n-pose.pt")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = handle_image(image_bytes, task="detect")
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = handle_image(image_bytes, task="segment")
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pose")
async def pose_estimate(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = handle_image(image_bytes, task="pose")
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# def save_temp_file(upload_file: UploadFile, suffix: str = ".mp4") -> str:
#     temp_filename = f"temp_{uuid.uuid4().hex}{suffix}"
#     temp_path = os.path.join("temp_uploads", temp_filename)
#     os.makedirs("temp_uploads", exist_ok=True)
#     with open(temp_path, "wb") as buffer:
#         shutil.copyfileobj(upload_file.file, buffer)
#     return temp_path


@app.post("/video/detect")
async def video_detect(file: UploadFile = File(...)):
    try:
        result = await handle_video(file, "detect")  # pass UploadFile
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/video/segment")
async def video_segment(file: UploadFile = File(...)):
    try:
        result = await handle_video(file, "segment")  # pass UploadFile
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/video/pose")
async def video_pose(file: UploadFile = File(...)):
    try:
        result = await handle_video(file, "pose")  # pass UploadFile
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
